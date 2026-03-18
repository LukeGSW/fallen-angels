"""
pipeline/technicals.py — Motore Tecnico del framework Fallen Angels.

Calcola per ogni ticker nella whitelist:
    - Z-Score = (Price - SMA50) / StdDev50  → trigger <= -2.5
    - Volume Anomaly Ratio = Volume_oggi / ADV_20gg → soglia >= 1.5x
    - SMA200 per il time-stop della Fase 2
    - Consecutive days below SMA200 (per il Phase 2 Monitor)

Fonte dati: cache/prices.parquet (aggiornata nightly dalla pipeline)
Endpoint EODHD per update nightly:
    GET /api/eod-bulk-last-day/{exchange}?api_token={KEY}&fmt=json
    (1 chiamata per exchange, restituisce tutti i prezzi dell'ultima sessione)
    + GET /api/eod/{ticker}?period=d&from=...&to=...  (per storico iniziale)
"""

import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# === CACHE ===
PRICES_CACHE_PATH    = Path("cache/prices.parquet")
TECHNICALS_CACHE_PATH = Path("cache/technicals.parquet")

# === PARAMETRI TECNICI (dal paper) ===
ZSCORE_WINDOW       = 50    # periodi per SMA e StdDev nel calcolo Z-Score
ZSCORE_TRIGGER      = -2.5  # soglia trigger per candidati operativi
VOLUME_WINDOW       = 20    # giorni per il calcolo ADV (Average Daily Volume)
VOLUME_RATIO_MIN    = 1.5   # rapporto volume/ADV minimo per conferma capitolazione
SMA200_WINDOW       = 200   # periodi per SMA200 (time-stop Fase 2)
SMA200_STOP_DAYS    = 90    # giorni consecutivi sotto SMA200 per attivare time-stop
HISTORY_DAYS        = 260   # giorni di storico prezzi da mantenere in cache (~1 anno trading)


# ============================================================
# FETCH PREZZI EODHD
# ============================================================

def fetch_ohlcv(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Scarica i prezzi OHLCV giornalieri per un singolo ticker da EODHD.

    Args:
        ticker:  Simbolo EODHD (es. 'AAPL.US')
        start:   Data inizio YYYY-MM-DD
        end:     Data fine YYYY-MM-DD
        api_key: Chiave API EODHD

    Returns:
        DataFrame con colonne: open, high, low, close, adjusted_close, volume
        Indice: DatetimeIndex giornaliero.
    """
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "from":        start,
        "to":          end,
        "period":      "d",
        "api_token":   api_key,
        "fmt":         "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
        df = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fetch OHLCV {ticker}: {e}")
        return pd.DataFrame()


def fetch_bulk_eod(exchange: str, api_key: str, date: Optional[str] = None) -> pd.DataFrame:
    """
    Scarica i prezzi di chiusura dell'ultimo giorno per TUTTI i ticker di un exchange.
    Una singola chiamata API restituisce migliaia di ticker.

    Args:
        exchange: Codice exchange EODHD (es. 'NYSE', 'NASDAQ', 'AMEX')
        api_key:  Chiave API EODHD
        date:     Data specifica YYYY-MM-DD (default: ultimo giorno disponibile)

    Returns:
        DataFrame con colonne: code, open, high, low, close, adjusted_close, volume, date
    """
    url = f"https://eodhd.com/api/eod-bulk-last-day/{exchange}"
    params = {"api_token": api_key, "fmt": "json"}
    if date:
        params["date"] = date

    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        numeric_cols = ["open", "high", "low", "close", "adjusted_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fetch bulk EOD {exchange}: {e}")
        return pd.DataFrame()


# ============================================================
# GESTIONE CACHE PREZZI
# ============================================================

def load_prices_cache() -> pd.DataFrame:
    """
    Carica la cache prezzi da Parquet.

    Formato: colonne = [ticker, date, open, high, low, close, adjusted_close, volume]
    """
    if PRICES_CACHE_PATH.exists():
        df = pd.read_parquet(PRICES_CACHE_PATH)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def save_prices_cache(df: pd.DataFrame) -> None:
    """Salva la cache prezzi su Parquet."""
    PRICES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICES_CACHE_PATH, index=False)
    logger.info(f"Prezzi salvati: {len(df):,} righe → {PRICES_CACHE_PATH}")


def update_prices_from_bulk(
    tickers_whitelist: list,
    api_key: str,
    exchanges: list = None,
) -> pd.DataFrame:
    """
    Aggiorna la cache prezzi con i dati dell'ultima sessione usando l'endpoint bulk.

    Strategia:
        1. Scarica bulk EOD per ogni exchange → prezzi ultimo giorno
        2. Filtra solo i ticker nella whitelist fondamentale
        3. Appende alla cache esistente
        4. Mantiene solo gli ultimi HISTORY_DAYS giorni per ogni ticker

    Args:
        tickers_whitelist: Lista ticker (formato 'AAPL.US') da tenere in cache
        api_key:           Chiave API EODHD
        exchanges:         Lista exchange da interrogare (default: NYSE, NASDAQ, AMEX)

    Returns:
        DataFrame prezzi aggiornato.
    """
    if exchanges is None:
        exchanges = ["NYSE", "NASDAQ", "AMEX"]

    # Costruisci set per lookup veloce
    whitelist_set = set(tickers_whitelist)

    # Fetch bulk da ogni exchange
    bulk_frames = []
    for exch in exchanges:
        logger.info(f"Bulk EOD fetch: {exch}...")
        df_bulk = fetch_bulk_eod(exch, api_key)
        if df_bulk.empty:
            continue

        # Normalizza il ticker nel formato EODHD (CODE.US)
        if "code" in df_bulk.columns:
            df_bulk["ticker"] = df_bulk["code"] + ".US"
        else:
            continue

        # Filtra solo i ticker della whitelist
        df_bulk = df_bulk[df_bulk["ticker"].isin(whitelist_set)].copy()

        keep_cols = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
        df_bulk = df_bulk[[c for c in keep_cols if c in df_bulk.columns]]
        bulk_frames.append(df_bulk)

    if not bulk_frames:
        logger.warning("Nessun dato bulk ricevuto.")
        return load_prices_cache()

    new_data = pd.concat(bulk_frames, ignore_index=True)

    # Carica cache esistente e appende nuovi dati
    existing = load_prices_cache()
    if not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # Deduplicazione (stesso ticker + stessa data)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Mantieni solo gli ultimi HISTORY_DAYS giorni per ogni ticker
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=HISTORY_DAYS)
    combined = combined[combined["date"] >= cutoff]

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    save_prices_cache(combined)
    logger.info(f"Cache prezzi aggiornata: {len(combined):,} righe, {combined['ticker'].nunique():,} ticker")
    return combined


def initialize_prices_history(
    tickers: list,
    api_key: str,
    lookback_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Popola la cache prezzi da zero scaricando lo storico per ogni ticker.
    Usato solo al primo avvio del sistema.

    ATTENZIONE: con 5.000 ticker richiede ~5.000 chiamate API.
    Eseguire solo domenica notte durante il full refresh.

    Args:
        tickers:      Lista ticker
        api_key:      Chiave API EODHD
        lookback_days: Giorni di storico da scaricare

    Returns:
        DataFrame con storico prezzi.
    """
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    all_frames = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            logger.info(f"Init prezzi: {i}/{total}...")
        df = fetch_ohlcv(ticker, start, end, api_key)
        if not df.empty:
            df = df.reset_index()
            df["ticker"] = ticker
            all_frames.append(df)
        import time
        time.sleep(0.07)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    save_prices_cache(combined)
    return combined


# ============================================================
# CALCOLO INDICATORI TECNICI
# ============================================================

def compute_technicals_for_ticker(
    prices: pd.DataFrame,
    ticker: str,
) -> Optional[dict]:
    """
    Calcola tutti gli indicatori tecnici per un singolo ticker.

    Args:
        prices: DataFrame con colonne [ticker, date, close, adjusted_close, volume]
                pre-filtrato per il ticker specifico e ordinato per data crescente
        ticker: Simbolo del ticker (per logging)

    Returns:
        Dict con indicatori tecnici, o None se dati insufficienti.
    """
    if prices.empty or len(prices) < ZSCORE_WINDOW:
        return None

    # Ordina per data crescente (più vecchio → più recente)
    p = prices.sort_values("date").copy()

    # Usa adjusted_close se disponibile, altrimenti close
    price_col = "adjusted_close" if "adjusted_close" in p.columns else "close"
    close_prices = p[price_col].astype(float)
    volumes      = p["volume"].astype(float)

    # === Z-SCORE (SMA50) ===
    sma50   = close_prices.rolling(ZSCORE_WINDOW).mean()
    std50   = close_prices.rolling(ZSCORE_WINDOW).std()
    z_score = (close_prices - sma50) / std50

    # === SMA200 ===
    sma200  = close_prices.rolling(SMA200_WINDOW).mean()

    # === VOLUME ANOMALY RATIO ===
    adv20        = volumes.rolling(VOLUME_WINDOW).mean()
    volume_ratio = volumes / adv20

    # Valori correnti (ultima riga)
    current_price       = float(close_prices.iloc[-1])
    current_z           = float(z_score.iloc[-1]) if not np.isnan(z_score.iloc[-1]) else None
    current_sma50       = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
    current_sma200      = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
    current_volume_ratio = float(volume_ratio.iloc[-1]) if not np.isnan(volume_ratio.iloc[-1]) else None
    current_adv20       = float(adv20.iloc[-1]) if not np.isnan(adv20.iloc[-1]) else None
    current_volume      = float(volumes.iloc[-1])
    last_date           = p["date"].iloc[-1]

    # === GIORNI CONSECUTIVI SOTTO SMA200 (per time-stop Fase 2) ===
    # Conta dal giorno più recente quanti giorni consecutivi il prezzo è sotto SMA200
    days_below_sma200 = 0
    if current_sma200 is not None:
        below_sma200 = (close_prices < sma200).values[::-1]  # inverti: recente → antico
        for below in below_sma200:
            if below:
                days_below_sma200 += 1
            else:
                break

    # === FLAG TRIGGER OPERATIVO ===
    # Il ticker è un candidato se: Z <= -2.5 AND volume_ratio >= 1.5
    # Il filtro earnings è applicato separatamente in screener.py
    is_trigger = (
        current_z is not None
        and current_z <= ZSCORE_TRIGGER
        and current_volume_ratio is not None
        and current_volume_ratio >= VOLUME_RATIO_MIN
    )

    # === FLAG TIME-STOP (Fase 2) ===
    time_stop_alert = days_below_sma200 >= SMA200_STOP_DAYS

    return {
        "ticker":              ticker,
        "last_date":           last_date,
        "price":               round(current_price, 2),
        "sma50":               round(current_sma50, 2) if current_sma50 else None,
        "sma200":              round(current_sma200, 2) if current_sma200 else None,
        "z_score":             round(current_z, 3) if current_z is not None else None,
        "volume":              int(current_volume),
        "adv20":               int(current_adv20) if current_adv20 else None,
        "volume_ratio":        round(current_volume_ratio, 2) if current_volume_ratio else None,
        "days_below_sma200":   days_below_sma200,
        "is_trigger":          is_trigger,
        "time_stop_alert":     time_stop_alert,
        # Variazione % rispetto a ieri
        "pct_change_1d":       round(float(close_prices.pct_change().iloc[-1]) * 100, 2) if len(close_prices) > 1 else None,
    }


def compute_all_technicals(prices_cache: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola gli indicatori tecnici per tutti i ticker nella cache prezzi.

    Args:
        prices_cache: DataFrame long-format con colonne [ticker, date, close, ...]

    Returns:
        DataFrame con una riga per ticker e tutti gli indicatori tecnici.
    """
    if prices_cache.empty:
        return pd.DataFrame()

    results = []
    tickers = prices_cache["ticker"].unique()

    for ticker in tickers:
        ticker_prices = prices_cache[prices_cache["ticker"] == ticker].copy()
        record = compute_technicals_for_ticker(ticker_prices, ticker)
        if record is not None:
            results.append(record)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"Tecnici calcolati: {len(df):,} ticker")
    return df


# ============================================================
# STORICO Z-SCORE PER GRAFICI (Tab 2 - Ticker Detail)
# ============================================================

def get_zscore_history(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la serie storica dello Z-Score per un singolo ticker.
    Usata dalla Tab 2 per il grafico Z-Score nel tempo.

    Args:
        prices: DataFrame prezzi per un singolo ticker, con colonne [date, adjusted_close/close]
                ordinato per data crescente.

    Returns:
        DataFrame con colonne: date, price, sma50, z_score, sma200, volume, volume_ratio
    """
    if prices.empty or len(prices) < ZSCORE_WINDOW:
        return pd.DataFrame()

    p = prices.sort_values("date").copy()
    price_col = "adjusted_close" if "adjusted_close" in p.columns else "close"

    close  = p[price_col].astype(float)
    volume = p["volume"].astype(float) if "volume" in p.columns else pd.Series(dtype=float)

    sma50        = close.rolling(ZSCORE_WINDOW).mean()
    std50        = close.rolling(ZSCORE_WINDOW).std()
    z_score      = (close - sma50) / std50
    sma200       = close.rolling(SMA200_WINDOW).mean()
    adv20        = volume.rolling(VOLUME_WINDOW).mean() if not volume.empty else None
    volume_ratio = (volume / adv20) if adv20 is not None else None

    result = pd.DataFrame({
        "date":         p["date"].values,
        "price":        close.values,
        "sma50":        sma50.values,
        "sma200":       sma200.values,
        "z_score":      z_score.values,
        "volume":       volume.values if not volume.empty else np.nan,
        "volume_ratio": volume_ratio.values if volume_ratio is not None else np.nan,
    })

    return result.dropna(subset=["z_score"])


# ============================================================
# CACHE TECHNICALS I/O
# ============================================================

def load_technicals_cache() -> pd.DataFrame:
    """Carica la cache indicatori tecnici."""
    if TECHNICALS_CACHE_PATH.exists():
        return pd.read_parquet(TECHNICALS_CACHE_PATH)
    return pd.DataFrame()


def save_technicals_cache(df: pd.DataFrame) -> None:
    """Salva la cache indicatori tecnici."""
    TECHNICALS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TECHNICALS_CACHE_PATH, index=False)
    logger.info(f"Tecnici salvati: {len(df):,} ticker → {TECHNICALS_CACHE_PATH}")
