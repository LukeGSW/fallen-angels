# 🪽 Fallen Angels Screener

Screener quantamentale per identificare titoli azionari USA con fondamentali solidi in
temporanea dislocazione tecnica ("Fallen Angels"). Sviluppato su framework Kriterion Quant.

---

## Architettura

```
fallen-angels/
├── app.py                          # Streamlit UI (3 tab)
├── scheduler.py                    # Pipeline notturna (GitHub Actions)
├── requirements.txt
├── pipeline/
│   ├── universe.py                 # Costruzione universo Common Stock USA
│   ├── fundamentals.py             # Piotroski F-Score TTM, FCF Yield, ICR
│   ├── technicals.py               # Z-Score, SMA, Volume Anomaly
│   ├── earnings.py                 # Earnings calendar exclusions
│   └── screener.py                 # Combina i 3 motori → candidati
├── src/
│   └── charts.py                   # Plotly dark-theme chart library
├── cache/                          # Parquet pre-computati (aggiornati da GitHub Actions)
│   ├── universe.parquet
│   ├── fundamentals.parquet
│   ├── prices.parquet
│   ├── technicals.parquet
│   ├── screener_results.parquet
│   └── last_update.json
└── .github/workflows/
    └── nightly_pipeline.yml        # Cron Mon–Fri 22:00 EET
```

### Flusso dati

```
GitHub Actions (22:00 EET)
    ↓
scheduler.py
    ├── [Domenica] fetch universo + fondamentali (~50K API calls)
    └── [Ogni sera] update prezzi bulk + tecnici + earnings (~5K API calls)
    ↓
cache/*.parquet   ←  committati nel repo
    ↓
Streamlit Cloud legge la cache → app.py
```

---

## Deploy

### 1. Fork / Clone del repository

```bash
git clone https://github.com/TUO-USERNAME/fallen-angels.git
cd fallen-angels
```

### 2. Configura GitHub Actions Secrets

Vai su **Settings → Secrets and variables → Actions → New repository secret**
e aggiungi i seguenti secrets:

| Secret | Descrizione |
|---|---|
| `EODHD_API_KEY` | Chiave API EODHD (piano Fundamental Data Feed) |
| `TELEGRAM_TOKEN` | Token del bot Telegram (da @BotFather) |
| `TELEGRAM_CHAT_ID` | ID della chat personale Telegram |

Per ottenere il `TELEGRAM_CHAT_ID`: invia un messaggio al tuo bot, poi visita
`https://api.telegram.org/botTOKEN/getUpdates` e leggi il campo `chat.id`.

### 3. Prima esecuzione (bootstrap cache)

Al primo run, la cache è vuota. Esegui manualmente il workflow:

1. Vai su **Actions → Fallen Angels — Nightly Pipeline**
2. Clicca **Run workflow**
3. Seleziona `force_full_refresh: true`
4. Attendi il completamento (~60-90 min per il fetch fondamentali completo)

In alternativa, esegui localmente:

```bash
export EODHD_API_KEY="la-tua-chiave"
python scheduler.py
```

### 4. Deploy su Streamlit Cloud

1. Vai su [share.streamlit.io](https://share.streamlit.io)
2. Connetti il repository GitHub
3. Imposta:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
4. Aggiungi i secrets Streamlit in **Settings → Secrets**:

```toml
EODHD_API_KEY = "la-tua-chiave"
```

5. Clicca **Deploy**

---

## Logica dello screener

### Universo

- NYSE, NASDAQ, NYSE ARCA
- Solo Common Stock (no ETF, no preferred)
- Esclusi: Financials, Financial Services, Real Estate (ICR non applicabile)
- Market Cap ≥ $500M
- ADV (Average Daily Volume 20D) ≥ 500K

### Motore fondamentale (whitelist)

| Filtro | Soglia |
|---|---|
| Piotroski F-Score TTM | ≥ 7/9 |
| FCF Yield | > 5% (media 8Q per settori ciclici) |
| ICR (Interest Coverage Ratio) | Dipende dal settore (vedi tabella sotto) |

**Soglie ICR per settore:**

| Settore | Soglia ICR |
|---|---|
| Utilities | 2× |
| Energy, Materials | 3× |
| Consumer Staples, Industrials, Communication | 4× |
| Technology, Healthcare, Consumer Discretionary | 5× |

### Trigger tecnico (candidati)

Un titolo in whitelist diventa **candidato operativo** se:

- **Z-Score** = (Price − SMA50) / StdDev50 ≤ −2.5 (dislocazione statistica rara, < 1%)
- **Volume Anomaly** = Volume giornaliero ≥ 1.5× ADV20 (capitolazione confermata)
- **Earnings filter**: nessun earnings announcement nei prossimi 14 giorni

### Phase 2 Monitor (stop loss)

Monitora i titoli già in posizione:

- **Time-Stop**: Prezzo < SMA200 per 90 giorni consecutivi di trading
- **Fundamental Stop**: F-Score < 5 OR FCF negativo per 2 trimestri consecutivi

---

## Sviluppo locale

```bash
# Setup ambiente
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Variabili d'ambiente
export EODHD_API_KEY="la-tua-chiave"

# Prima esecuzione pipeline
python scheduler.py

# Avvia l'app Streamlit
streamlit run app.py
```

---

## Note sulle API calls (budget giornaliero EODHD)

| Operazione | Frequenza | Calls stimate |
|---|---|---|
| Fetch fondamentali (10 call/ticker × ~5K ticker) | Domenica | ~50.000 |
| Bulk EOD prezzi (3 exchange) | Ogni sera | ~3 |
| Earnings calendar (1 call) | Ogni sera | 1 |
| **Totale serale feriale** | | ~10 |
| **Totale domenicale** | | ~50.000 |

Piano EODHD consigliato: **100K calls/day** con rate limit 1.000/min.

---

*Kriterion Quant — Fallen Angels Framework v1.0*
