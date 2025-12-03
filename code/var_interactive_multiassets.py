import warnings
warnings.filterwarnings("ignore")  # silenzia messaggi runtime non critici

import os
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf


# =========================
# CONFIGURAZIONE GENERALE
# =========================

# Calendario su cui allineare tutti gli asset
# "B" = Business Days (consigliato per VaR 1-day su portafogli misti)
# "D" = Calendar Days (tiene i weekend, per asset non 24/7)
CALENDAR_FREQ = "B"

# Numero di simulazioni Monte Carlo
MC_SIMS = 100_000
RANDOM_SEED = 200  # per riproducibilita'


# =========================
# INPUT HELPERS (interattivi da terminale)
# =========================

def ask_list(prompt: str) -> list:
    """Chiede una lista separata da virgola (es. NVDA,AAPL,BTC-USD)."""
    s = input(prompt).strip()
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        raise ValueError("Lista vuota.")
    return items

def ask_floats(prompt: str, n: int) -> np.ndarray:
    """Chiede n importi (float) separati da virgola, nella valuta base."""
    s = input(prompt).strip()
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if len(vals) != n:
        raise ValueError(f"Servono {n} valori, ne hai dati {len(vals)}.")
    arr = np.array([float(x.replace(",", ".")) for x in vals], dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Tutti gli importi devono essere > 0.")
    return arr

def ask_base_ccy(prompt: str, default: str = "EUR") -> str:
    """Chiede la valuta base (ISO a 3 lettere). Ripete finche' non valida."""
    while True:
        s = input(prompt + f" [{default}]: ").strip().upper()
        s = s if s else default
        if len(s) == 3 and s.isalpha():
            return s
        print("Valuta non valida. Inserisci un codice ISO a 3 lettere (es. EUR, USD, GBP).")


# =========================
# FUNZIONI FX E DOWNLOAD
# =========================

def _fx_symbol(from_ccy: str, to_ccy: str) -> str:
    """
    Yahoo usa BASEQUOTE=X.
    Per convertire prezzi da FROM -> TO, serve TOFROM=X e poi fare PREZZO / FX.
    Esempio: da USD a EUR -> simbolo 'EURUSD=X' (1 EUR = X USD), quindi prezzo_EUR = prezzo_USD / FX.
    """
    return f"{to_ccy}{from_ccy}=X"

def fetch_fx_series(from_ccy: str, to_ccy: str, start, end) -> pd.Series:
    """Scarica la serie FX (TO/FROM). Se mancante, prova l'inversa e inverte."""
    if from_ccy == to_ccy:
        # Serie costante = 1, indicizzata sul calendario richiesto
        idx = pd.date_range(start=start, end=end, freq=CALENDAR_FREQ)
        return pd.Series(1.0, index=idx, name=f"{to_ccy}/{from_ccy}")

    sym = _fx_symbol(from_ccy, to_ccy)
    df = yf.download(sym, start=start, end=end, auto_adjust=True)
    s = df["Close"] if isinstance(df, pd.DataFrame) else df
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]  # type: ignore
    s = pd.Series(s, dtype="float64").dropna()

    if s.empty:
        # prova l'inversa
        inv = f"{from_ccy}{to_ccy}=X"
        dfi = yf.download(inv, start=start, end=end, auto_adjust=True)
        si = dfi["Close"] if isinstance(dfi, pd.DataFrame) else dfi
        if isinstance(si, pd.DataFrame):
            si = si.iloc[:, 0] # type: ignore
        si = pd.Series(si, dtype="float64").dropna()
        if si.empty:
            raise RuntimeError(f"FX non trovata per {from_ccy}->{to_ccy}")
        s = 1.0 / si

    # riallinea al calendario globale (ffill per buche)
    idx = pd.date_range(s.index.min(), s.index.max(), freq=CALENDAR_FREQ)
    s = s.reindex(idx).ffill()
    s.name = f"{to_ccy}/{from_ccy}"
    return s

def fetch_close_in_base(ticker: str, start, end, base_ccy: str) -> pd.Series:
    """
    Scarica i CLOSE di un ticker e li converte nella valuta base.
    Gestisce anche MultiIndex di yfinance e allinea al calendario globale.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"Download fallito per {ticker}")

    # estrae i 'Close' 
    if "Close" in df.columns:
        px = df["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0] # type: ignore
    elif isinstance(df.columns, pd.MultiIndex):
        px = df["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0] # type: ignore
    else:
        raise RuntimeError(f"Colonna Close non trovata per {ticker}")

    px = pd.Series(px, dtype="float64").dropna()
    px.name = ticker

    # valuta di quotazione
    ccy = None
    try:
        ccy = yf.Ticker(ticker).fast_info.get("currency", None)
    except Exception:
        pass

    # riallinea prima al calendario globale
    idx = pd.date_range(px.index.min(), px.index.max(), freq=CALENDAR_FREQ)
    px = px.reindex(idx).ffill()

    # se la valuta e' sconosciuta o uguale alla base, ritorna px
    if ccy is None or ccy == base_ccy:
        return px

    # conversione in valuta base
    fx = fetch_fx_series(from_ccy=ccy, to_ccy=base_ccy, start=px.index.min(), end=px.index.max())
    df_conv = pd.concat([px, fx], axis=1).dropna()
    if df_conv.empty:
        raise RuntimeError(f"Impossibile allineare {ticker} con FX {ccy}->{base_ccy}")
    px_base = df_conv.iloc[:, 0] / df_conv.iloc[:, 1]
    px_base.name = ticker
    return px_base

def get_close_df_any_asset(tickers: list, start, end, base_ccy: str) -> pd.DataFrame:
    """
    Scarica e converte in valuta base una lista di ticker eterogenei,
    allinea tutto al calendario globale e forward-filla i buchi.
    """
    cols = []
    for t in tickers:
        try:
            s = fetch_close_in_base(t, start=start, end=end, base_ccy=base_ccy)
            cols.append(s)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    if not cols:
        raise RuntimeError("Nessuna serie valida scaricata.")
    df = pd.concat(cols, axis=1).astype(float)
    # eventuali colonne completamente vuote (dopo allineamento) vengono rimosse
    df = df.dropna(axis=1, how="all")
    return df


# =========================
# RENDIMENTI E METODI VaR
# =========================

def portfolio_returns(price_data, weights: np.ndarray) -> pd.Series:
    """
    Calcola rendimenti log del portafoglio.
    - Coercizione a DataFrame float
    - Ribilancia equal-weight se manca qualche colonna
    """
    price_df = pd.DataFrame(price_data).astype(float)
    rets = np.log(price_df / price_df.shift(1)).dropna() # type: ignore
    if rets.shape[1] != len(weights):
        w = np.repeat(1.0 / rets.shape[1], rets.shape[1])
    else:
        w = weights
    return rets.dot(w)  # Series

def var_parametric_from_series(port_ret: pd.Series, total_capital: float, alphas=(0.95, 0.99)) -> dict:
    mu, sigma = float(port_ret.mean()), float(port_ret.std())
    out = {}
    for a in alphas:
        z = float(norm.ppf(1 - a))
        out[f"VaR {int(a*100)}%"] = float(-(mu + z * sigma) * total_capital)
    return out

def var_historical_from_series(port_ret: pd.Series, total_capital: float, alphas=(0.95, 0.99)) -> dict:
    out = {}
    for a in alphas:
        q = float(np.percentile(port_ret, (1 - a) * 100))  # quantile coda sinistra
        out[f"VaR {int(a*100)}%"] = float(-q * total_capital)
    return out

def mc_normal_var_from_prices(price_data, weights, total_capital, n_sims=MC_SIMS, alphas=(0.95, 0.99), random_seed=RANDOM_SEED) -> dict:
    """Monte Carlo multivariata normale, preserva correlazioni storiche."""
    if random_seed is not None:
        np.random.seed(random_seed)
    df = pd.DataFrame(price_data).astype(float)
    rets = np.log(df / df.shift(1)).dropna() # type: ignore
    mu_vec = rets.mean().to_numpy()
    cov_mat = rets.cov().to_numpy()
    sims = np.random.multivariate_normal(mean=mu_vec, cov=cov_mat, size=n_sims)
    w = weights if rets.shape[1] == len(weights) else np.repeat(1 / rets.shape[1], rets.shape[1])
    port_sims = sims @ w
    out = {}
    for a in alphas:
        q = float(np.percentile(port_sims, (1 - a) * 100))
        out[f"VaR {int(a*100)}%"] = float(-q * total_capital)
    return out

def estimate_df_from_excess_kurtosis(x, min_df=4.5, max_df=30.0, default_df=8.0) -> float:
    """Stima df per t-Student dalla kurtosi in eccesso (Fisher)."""
    s = pd.Series(x, dtype="float64").dropna()
    if s.empty:
        return float(default_df)
    g2 = float(s.kurtosis()) # type: ignore
    if not np.isfinite(g2):
        return float(default_df)
    g2 = max(g2, 0.3)  # evita esplosioni
    df = 4.0 + 6.0 / g2
    return float(np.clip(df, min_df, max_df))

def mc_t_var_from_prices(price_data, weights, total_capital, n_sims=MC_SIMS, alphas=(0.95, 0.99), df=None, random_seed=RANDOM_SEED) -> dict:
    """Monte Carlo multivariata t-Student (scale mixture) con code piu' pesanti."""
    if random_seed is not None:
        np.random.seed(random_seed)
    dfp = pd.DataFrame(price_data).astype(float)
    rets = np.log(dfp / dfp.shift(1)).dropna() # type: ignore
    mu_vec = rets.mean().to_numpy()
    S = rets.cov().to_numpy()
    k = S.shape[0]
    w = weights if rets.shape[1] == len(weights) else np.repeat(1 / rets.shape[1], rets.shape[1])

    port_hist = (rets.to_numpy() @ w)
    if df is None:
        df = estimate_df_from_excess_kurtosis(port_hist)
    df = float(max(df, 2.1))

    # Z ~ N(0, S) tramite Cholesky
    L = np.linalg.cholesky(S + 1e-12 * np.eye(k))
    Z = np.random.randn(n_sims, k) @ L.T
    # fattore t
    W = np.random.chisquare(df, size=n_sims)
    scale = np.sqrt((df - 2.0) / W).reshape(-1, 1)
    T = Z * scale
    sims = T + mu_vec
    port_sims = sims @ w

    out = {}
    for a in alphas:
        q = float(np.percentile(port_sims, (1 - a) * 100))
        out[f"VaR {int(a*100)}%"] = float(-q * total_capital)
    return out


# =========================
# OUTPUT: TABELLE, TESTI, PLOT
# =========================

def pretty_table(results_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(results_dict).T
    return df.applymap(lambda x: f"€{x:,.2f}") # type: ignore

def explain_messages(header: str, results: dict) -> None:
    print(f"\n{header}:")
    for k, v in results.items():
        perc = int(k.split()[1].replace("%", ""))
        tail = 100 - perc
        print(f"{k}: €{v:,.2f}")
        print(f"-> Nel {tail}% dei casi peggiori potresti perdere piu di €{v:,.2f} in un giorno.")

def plot_with_lines(port_ret: pd.Series, total_capital: float, title: str, lines: dict, style: str, save_path: str, with_normal=False):
    plt.figure(figsize=(9, 5.6))
    plt.hist(port_ret, bins=60, density=True, alpha=0.6)
    if with_normal:
        mu, sigma = float(port_ret.mean()), float(port_ret.std())
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 400)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, linewidth=2, label="Normal fit (mu, sigma)")
    for k, v in lines.items():
        plt.axvline(-v / total_capital, linestyle=style, linewidth=2.0, label=f"{k}: €{v:,.0f}")
    plt.title(title)
    plt.xlabel("Daily return"); plt.ylabel("Density"); plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.show()


# =========================
# MAIN INTERACTIVE
# =========================

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print("=== Interactive VaR - Multi-asset con FX ===")

    tickers = ask_list("Inserisci i tickers separati da virgola (es. NVDA,AAPL,TLT,GLD,CL=F,BTC-USD): ")
    base_ccy = ask_base_ccy("Valuta base del portafoglio", default="EUR")
    n = len(tickers)
    amounts = ask_floats(f"Inserisci {n} importi nella valuta base ({base_ccy}), separati da virgola (stesso ordine dei tickers): ", n)

    total_capital = float(amounts.sum())
    weights = amounts / total_capital

    print("\nRiepilogo input")
    print("  Tickers:", tickers)
    print("  Valuta base:", base_ccy)
    print("  Importi:", [float(x) for x in amounts])
    print("  Totale:", f"{base_ccy} {total_capital:,.2f}".replace(",", "."))
    print("  Pesi:", np.round(weights, 3))

    end_date = dt.date.today()

    # Scarica prezzi e converte in valuta base
    data_recent = get_close_df_any_asset(tickers, start="2024-01-01", end=end_date, base_ccy=base_ccy)
    data_long   = get_close_df_any_asset(tickers, start="2015-01-01", end=end_date, base_ccy=base_ccy)

    # Rendimenti di portafoglio coerenti con il calendario scelto
    port_recent = portfolio_returns(data_recent, weights)
    port_long   = portfolio_returns(data_long,   weights)

    # Calcolo VaR con tutti i metodi
    param_recent = var_parametric_from_series(port_recent, total_capital)
    hist_recent  = var_historical_from_series(port_recent, total_capital)
    mcN_recent   = mc_normal_var_from_prices(data_recent, weights, total_capital)
    mcT_recent   = mc_t_var_from_prices(data_recent, weights, total_capital)

    param_long = var_parametric_from_series(port_long, total_capital)
    hist_long  = var_historical_from_series(port_long, total_capital)
    mcN_long   = mc_normal_var_from_prices(data_long, weights, total_capital)
    mcT_long   = mc_t_var_from_prices(data_long, weights, total_capital)

    # Tabella comparativa
    results = {
        f"Parametric (2024-{end_date.year})": param_recent,
        f"Historical (2024-{end_date.year})":  hist_recent,
        f"MC Normal (2024-{end_date.year})":   mcN_recent,
        f"MC t-Student (2024-{end_date.year})": mcT_recent,
        f"Parametric (2015-{end_date.year})": param_long,
        f"Historical (2015-{end_date.year})":  hist_long,
        f"MC Normal (2015-{end_date.year})":   mcN_long,
        f"MC t-Student (2015-{end_date.year})": mcT_long,
    }
    print("\n[Table] VaR comparison - on {} {:,.0f}".format(base_ccy, total_capital))
    table_fmt = pretty_table(results)
    print(table_fmt)
    pd.DataFrame(results).T.to_csv("outputs/var_comparison_interactive_multiasset.csv")

    # Messaggi esplicativi
    explain_messages(f"Interpretazione - 2024-{end_date.year} - Parametric", param_recent)
    explain_messages(f"Interpretazione - 2024-{end_date.year} - Historical", hist_recent)
    explain_messages(f"Interpretazione - 2024-{end_date.year} - MC Normal", mcN_recent)
    explain_messages(f"Interpretazione - 2024-{end_date.year} - MC t-Student", mcT_recent)
    explain_messages(f"Interpretazione - 2015-{end_date.year} - Parametric", param_long)
    explain_messages(f"Interpretazione - 2015-{end_date.year} - Historical", hist_long)
    explain_messages(f"Interpretazione - 2015-{end_date.year} - MC Normal", mcN_long)
    explain_messages(f"Interpretazione - 2015-{end_date.year} - MC t-Student", mcT_long)

    # Grafici
    plot_with_lines(port_recent, total_capital, f"Portfolio distribution - 2024-{end_date.year} - Parametric VaR", param_recent, "--", f"plots/parametric_2024_{end_date.year}_interactive.png", with_normal=True)
    plot_with_lines(port_recent, total_capital, f"Portfolio distribution - 2024-{end_date.year} - Historical VaR",  hist_recent,  ":",  f"plots/historical_2024_{end_date.year}_interactive.png")
    plot_with_lines(port_recent, total_capital, f"Portfolio distribution - 2024-{end_date.year} - Monte Carlo Normal", mcN_recent, "-.", f"plots/mc_normal_2024_{end_date.year}_interactive.png")
    plot_with_lines(port_recent, total_capital, f"Portfolio distribution - 2024-{end_date.year} - Monte Carlo t-Student", mcT_recent, "-.", f"plots/mc_t_2024_{end_date.year}_interactive.png")

    plot_with_lines(port_long, total_capital, f"Portfolio distribution - 2015-{end_date.year} - Parametric VaR", param_long, "--", f"plots/parametric_2015_{end_date.year}_interactive.png", with_normal=True)
    plot_with_lines(port_long, total_capital, f"Portfolio distribution - 2015-{end_date.year} - Historical VaR",  hist_long,  ":",  f"plots/historical_2015_{end_date.year}_interactive.png")
    plot_with_lines(port_long, total_capital, f"Portfolio distribution - 2015-{end_date.year} - Monte Carlo Normal", mcN_long, "-.", f"plots/mc_normal_2015_{end_date.year}_interactive.png")
    plot_with_lines(port_long, total_capital, f"Portfolio distribution - 2015-{end_date.year} - Monte Carlo t-Student", mcT_long, "-.", f"plots/mc_t_2015_{end_date.year}_interactive.png")

    print("\nFatto. Grafici in ./plots e risultati in ./outputs.")

