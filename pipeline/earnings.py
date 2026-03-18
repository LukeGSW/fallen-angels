"""
pipeline/earnings.py — Filtro Earnings Calendar.

Scarica gli earnings imminenti da EODHD Calendar API e costruisce
un set di ticker da escludere dallo screener operativo.

Logica: se un ticker ha earnings nei prossimi EARNINGS_EXCLUSION_DAYS giorni,
viene escluso dalla lista candidati (ma rimane nella whitelist fondamentale).
Vendere put prima di un earnings espone a gap non gestibili dal modello.

Endpoint EODHD:
    GET /api/calendar/earnings?from={today}&to={today+14d}&api_token={KEY}&fmt=json
    Costo: 1 API call (indipendentemente dal numero di ticker restituiti).
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

# === PARAMETRI ===
EARNINGS_EXCLUSION_DAYS = 14   # giorni di buffer pre-earnings
CACHE_PATH = Path("cache/earnings_exclusions.parquet")


def fetch_upcoming_earnings(api_key: str, days_ahead: int = EARNINGS_EXCLUSION_DAYS) -> pd.DataFrame:
    """
    Scarica tutti i ticker con earnings programmati nei prossimi giorni.

    Una singola chiamata con finestra temporale è molto più efficiente
    di N chiamate per singolo ticker (1 call vs ~5.000 call).

    Args:
        api_key:    Chiave API EODHD
        days_ahead: Numero di giorni in avanti da considerare

    Returns:
        DataFrame con colonne: code, report_date, before_after_market
    """
    today    = datetime.today().strftime("%Y-%m-%d")
    end_date = (datetime.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    url = "https://eodhd.com/api/calendar/earnings"
    params = {
        "from":      today,
        "to":        end_date,
        "api_token": api_key,
        "fmt":       "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # La risposta ha struttura: {"type": "Earnings", "earnings": [...]}
        earnings_list = data.get("earnings", []) if isinstance(data, dict) else []

        if not earnings_list:
            logger.info("Nessun earnings imminente trovato.")
            return pd.DataFrame()

        df = pd.DataFrame(earnings_list)
        logger.info(f"Earnings trovati nei prossimi {days_ahead} giorni: {len(df):,} eventi")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fetch earnings calendar: {e}")
        return pd.DataFrame()


def build_earnings_exclusion_set(api_key: str) -> Set[str]:
    """
    Costruisce il set di ticker da escludere per prossimità agli earnings.

    Normalizza i ticker dal formato EODHD (es. 'AAPL.US') al formato
    standard per confronto con la whitelist.

    Args:
        api_key: Chiave API EODHD

    Returns:
        Set di ticker (formato 'AAPL.US') con earnings nei prossimi 14 giorni.
    """
    df = fetch_upcoming_earnings(api_key)

    if df.empty or "code" not in df.columns:
        return set()

    # I ticker nell'earnings calendar hanno già il formato 'AAPL.US'
    exclusion_set = set(df["code"].dropna().unique())

    logger.info(f"Ticker esclusi per earnings imminenti: {len(exclusion_set):,}")
    return exclusion_set


def save_earnings_cache(exclusion_set: Set[str]) -> None:
    """Salva il set di esclusioni su Parquet per consultazione dalla UI."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ticker": sorted(exclusion_set)})
    df["exclusion_date"] = datetime.today().strftime("%Y-%m-%d")
    df.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Earnings exclusions salvate: {len(exclusion_set)} ticker")


def load_earnings_cache() -> Set[str]:
    """Carica il set di esclusioni dalla cache."""
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        return set(df["ticker"].tolist()) if "ticker" in df.columns else set()
    return set()
