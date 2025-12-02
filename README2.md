# Analisi del Rischio di Portafoglio Multi-Asset (VaR)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Quantitative-green)
![Parametric VaR example](plots/parametric_2024_2025_interactive.png)

## 💡 Motivazione del Progetto
Ho sviluppato questo progetto per un'esigenza pratica personale: **analizzare il rischio reale del mio portafoglio investimenti.**

Durante i miei studi in finanza, ho notato che molti modelli teorici "da manuale" non sono validi quando applicati alla realtà, specialmente se si detiene un portafoglio misto con:
1.  Asset in valute diverse (es. Azioni USA in USD su conto in EUR).
2.  Strumenti con orari di trading differenti (es. Crypto 24/7 vs Borse tradizionali).
3.  Rendimenti che non seguono una curva a campana perfetta (code grasse/eventi estremi).

Questo tool colma il divario tra teoria e pratica, automatizzando la "pulizia" dei dati finanziari e applicando modelli statistici avanzati per stimare quanto potrei perdere in un singolo giorno di mercato avverso.

## 🛠 Cosa risolve (La logica tecnica)

Il software gestisce automaticamente due problemi che spesso vengono ignorati:

* **Normalizzazione Valutaria (FX):** Se il portafoglio è in EUR ma contiene NVIDIA (in USD), il codice scarica i tassi di cambio storici e converte le serie temporali *prima* di calcolare le correlazioni.
* **Allineamento Temporale:** Armonizza asset che scambiano sempre (Bitcoin) con asset che scambiano solo nei giorni lavorativi, garantendo che la matrice di correlazione sia matematicamente coerente.

## Modelli Implementati

Il codice calcola il **Value-at-Risk (VaR)** (al 95% e 99%) confrontando quattro metodologie:

1.  **Parametrico (Varianza-Covarianza):** Assume una distribuzione normale. Veloce, ma spesso ottimista.
2.  **Simulazione Storica:** Non fa assunzioni, guarda semplicemente cosa è successo in passato.
3.  **Monte Carlo (Gaussiano):** Simula migliaia di scenari futuri basandosi sulla correlazione storica.
4.  **Monte Carlo (t-Student):** **Il mio metodo preferito per gli stress-test.** Simula scenari assumendo "code grasse" (alta curtosi), catturando meglio il rischio di eventi cigno nero rispetto alla normale gaussiana.

## 🧮 Formule

### 1. Costruzione dei Rendimenti
Il rendimento logaritmico giornaliero del portafoglio è calcolato come somma pesata, aggiustata per il tasso di cambio:

$$r_{p,t} = \sum_{i=1}^{N} w_i \cdot \ln\left(\frac{P_{i,t} \cdot FX_{i,t}}{P_{i,t-1} \cdot FX_{i,t-1}}\right)$$

### 2. Simulazione Monte Carlo (t-Student)
Per modellare i mercati reali (che sono leptocurtici), stimo i gradi di libertà $\nu$ basandomi sulla curtosi del portafoglio e genero scenari di shock:

$$\mathbf{r}_{sim} = \boldsymbol{\mu} + \frac{L \cdot \mathbf{z}}{\sqrt{W/\nu}}$$

Dove $L$ è la decomposizione di Cholesky della matrice di correlazione, $\mathbf{z}$ è un vettore normale standard e $W$ segue una distribuzione Chi-quadro.

---

## Come usarlo

### Prerequisiti
Installa le librerie necessarie:
```
pip install -r requirements.txt
```
### Esecuzione
Lo script è interattivo. Basta lanciarlo da terminale.
```
python code/var_interactive_multiasset.py
```
### Esempio di input:

Tickers: NVDA, AAPL, BTC-USD Valuta Base: EUR Importi investiti: 2000, 3000, 500

Il programma genererà:

- Un report a video con il VaR stimato.

- Grafici della distribuzione delle perdite nella cartella plots.

- Un file CSV dettagliato nella cartella outputs.

---

Autore: Luigi Di Muzio
