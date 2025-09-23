Value-at-Risk (VaR) Portfolio – Multi-Asset & Interactive

Progetto per calcolare e confrontare il VaR giornaliero di un portafoglio multi-asset usando 4 metodi:

Parametric (Normal)

Historical Simulation

Monte Carlo – Normal multivariata

Monte Carlo – t-Student multivariata (code pesanti, df stimato dalla kurtosi)

Supporta azioni, ETF, futures su materie prime, FX e crypto.
Tutti i prezzi vengono convertiti automaticamente nella valuta base scelta (default: EUR).
Analisi su due orizzonti: 2024→oggi e 2015→oggi.

Contenuti del repository
var-portfolio/
├─ code/
│  └─ var_interactive_multiasset.py   # script interattivo (multi-asset + FX + crypto)
├─ plots/                             # grafici generati (PNG)
├─ outputs/
│  └─ var_comparison_interactive_multiasset.csv
└─ README.md

Requisiti

Python 3.9+

Librerie: numpy, pandas, matplotlib, scipy, yfinance

Installazione rapida:

# opzionale: ambiente
# conda create -n var python=3.10 -y && conda activate var

pip install numpy pandas matplotlib scipy yfinance

Avvio rapido (script interattivo)

Esegui:

python code/var_interactive_multiasset.py


Lo script chiede:

Tickers (separati da virgola)

Azioni/ETF: NVDA,AAPL,TLT,GLD

Futures (continuous): CL=F,GC=F,NG=F

FX: EURUSD=X

Crypto: BTC-USD,ETH-USD

Valuta base (default EUR, codice ISO a 3 lettere).

Importi nella valuta base (stesso ordine dei ticker), ad es. 2000,1500,1000,500.

Lo script:

scarica i prezzi (Yahoo Finance, auto-adjusted);

converte tutto nella valuta base via FX di Yahoo;

calcola rendimenti log e VaR 95%/99% per tutti i metodi, su entrambi gli orizzonti;

salva grafici in plots/ e risultati in outputs/.

Output

Tabella comparativa stampata a terminale e salvata in
outputs/var_comparison_interactive_multiasset.csv

Grafici in plots/, uno per metodo/periodo:

parametric_YYYY_YYYY_interactive.png

historical_YYYY_YYYY_interactive.png

mc_normal_YYYY_YYYY_interactive.png

mc_t_YYYY_YYYY_interactive.png

Messaggi interpretativi (es. “Nel 1% dei casi peggiori potresti perdere più di €X in un giorno”).

Interpretazione del VaR (1-day)

VaR 95% ⇒ nel 5% peggiore dei giorni la perdita è almeno pari al VaR.

VaR 99% ⇒ nell’1% peggiore dei giorni la perdita è almeno pari al VaR.

Differenze tra metodi:

Parametric / MC Normal: assumono normalità (tail più “lisci”).

Historical: quantili direttamente dal campione (sensibile a outlier recenti).

MC t-Student: preserva media/covarianza ma con code più pesanti → VaR 99% spesso più alto.

Asset supportati (ticker Yahoo)

Azioni/ETF: AAPL, TLT, EUNL.DE, IWDA.AS, ISP.MI, …

Futures (continuous): CL=F (WTI), GC=F (oro), SI=F, NG=F, ZC=F (mais), …

FX: EURUSD=X, USDJPY=X, …

Crypto: BTC-USD, ETH-USD, SOL-USD, …

Attenzione ai suffissi di borsa su Yahoo: .MI (Borsa Italiana), .L (LSE), .DE (Xetra), .PA (Parigi), .AS (Amsterdam), ecc.

Configurazioni utili (nel file var_interactive_multiasset.py)
CALENDAR_FREQ = "B"   # "B" = Business Days (consigliato), "D" = Calendar Days
MC_SIMS = 100_000     # n. simulazioni Monte Carlo
RANDOM_SEED = 42      # riproducibilità


Crypto sono 24/7; con CALENDAR_FREQ="B" tutto viene allineato a giorni lavorativi (le crypto sono downsampled con ffill).

Se preferisci includere i weekend, usa CALENDAR_FREQ="D" e documentalo.

Esempio
Tickers: NVDA,TLT,GLD,BTC-USD
Valuta base: EUR
Importi: 2000,1500,1000,500


Output (estratto):

[Table] VaR comparison - on EUR 5,000
                          VaR 95%     VaR 99%
Parametric (2024-2025)    €xxx.xx     €yyy.yy
Historical (2024-2025)    €xxx.xx     €yyy.yy
MC Normal (2024-2025)     €xxx.xx     €yyy.yy
MC t-Student (2024-2025)  €xxx.xx     €yyy.yy
...


E relativi grafici in plots/.

Troubleshooting

“FX not found …” → controlla la valuta base e il ticker; se l’asset quota già in EUR non serve FX.

Ticker non trovato → verifica il suffisso (.MI, .L, .DE, …).

Buchi di calendario → CALENDAR_FREQ="B" (consigliato) o "D" se vuoi i weekend.

Dati Yahoo: utili per demo/portfolio; per produzione considera fonti professionali e ulteriori controlli.

Roadmap

Backtest rolling (P/L vs VaR, hit ratio)

Expected Shortfall (ES) 97.5% / 99%

Stress test/scenari

Versione CLI non interattiva con argparse

Note & Crediti

Dati: Yahoo Finance (prezzi auto-adjusted).

Il VaR è una stima statistica: non garantisce che la perdita non venga superata.

Autore: Luigi Di Muzio – LinkedIn / Email (inserisci i link).

---

## 📫 Contact
- LinkedIn: [Luigi Di Muzio](https://linkedin.com/in/luigidimuzio)  
- Email: [luigidimu@gmail.com](mailto:luigidimu@gmail.com)
