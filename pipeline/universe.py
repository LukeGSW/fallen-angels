"""
pipeline/universe.py — Costruzione e filtro dell'universo azionario USA.

Responsabilità:
    - Fetch della lista completa di strumenti quotati su NYSE, NASDAQ, NYSE ARCA
    - Filtro per tipo strumento: solo Common Stock
    - Esclusione settori Financials e Real Estate
    - Filtro Market Cap >= 500M USD e ADV >= 500K azioni
    - Persistenza su cache/universe.parquet

Endpoint EODHD:
    GET /api/exchange-symbol-list/{EXCHANGE}?api_token={KEY}&fmt=json
    Costo: 1 API call per exchange (3 chiamate totali)
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# === COSTANTI ===
CACHE_PATH = Path("cache/universe.parquet")

# Exchange codes EODHD per mercati USA target
TARGET_EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]  # AMEX = NYSE ARCA in EODHD

# Settori esclusi per incompatibilità con il framework (ICR non applicabile o struttura FCF diversa)
EXCLUDED_GICS_SECTORS = {"Financials", "Financial Services", "Real Estate"}

# Filtri quantitativi minimi di liquidità
MIN_MARKET_CAP_USD = 500_000_000   # 500 milioni USD
MIN_AVG_DAILY_VOLUME = 500_000      # 500.000 azioni/giorno


def fetch_exchange_symbols(exchange: str, api_key: str) -> pd.DataFrame:
    """
    Scarica la lista di tutti i simboli quotati su un exchange EODHD.

    Args:
        exchange: Codice exchange (es. 'NYSE', 'NASDAQ', 'AMEX')
        api_key:  Chiave API EODHD

    Returns:
        DataFrame con colonne: Code, Name, Exchange, Currency, Type
    """
    url = f"https://eodhd.com/api/exchange-symbol-list/{exchange}"
    params = {"api_token": api_key, "fmt": "json"}

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            logger.warning(f"Nessun simbolo restituito per exchange: {exchange}")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["Exchange"] = exchange  # normalizza il nome exchange
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fetch simboli {exchange}: {e}")
        return pd.DataFrame()


def build_raw_universe(api_key: str) -> pd.DataFrame:
    """
    Scarica e concatena i simboli dai tre exchange target.
    Applica il filtro tipo strumento (solo Common Stock).

    Args:
        api_key: Chiave API EODHD

    Returns:
        DataFrame con tutti i Common Stock USA (senza filtri quantitativi ancora)
    """
    frames = []
    for exchange in TARGET_EXCHANGES:
        logger.info(f"Fetch simboli da {exchange}...")
        df = fetch_exchange_symbols(exchange, api_key)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # cortesia verso il rate limiter

    if not frames:
        raise RuntimeError("Impossibile scaricare i simboli da nessun exchange.")

    universe = pd.concat(frames, ignore_index=True)

    # Deduplicazione: se un ticker appare su più exchange, teniamo la prima occorrenza
    if "Code" in universe.columns:
        universe = universe.drop_duplicates(subset="Code", keep="first")

    # Filtro tipo strumento: solo azioni ordinarie
    if "Type" in universe.columns:
        universe = universe[universe["Type"] == "Common Stock"].copy()
        logger.info(f"Common Stock dopo filtro tipo: {len(universe):,}")

    return universe.reset_index(drop=True)


def apply_quantitative_filters(
    universe: pd.DataFrame,
    fundamentals_cache: pd.DataFrame,
    prices_cache: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applica filtri quantitativi di liquidità usando dati già cachati.

    Filtri applicati:
        - Market Cap >= 500M USD (da fundamentals_cache)
        - ADV 20gg >= 500K azioni (da prices_cache)
        - Esclusione settori Financials / Real Estate (da fundamentals_cache)

    Args:
        universe:           DataFrame con colonne Code, Name, Exchange, Type
        fundamentals_cache: DataFrame con colonne ticker, market_cap, gic_sector, ...
        prices_cache:       DataFrame con colonne ticker, date, volume

    Returns:
        DataFrame filtrato con colonna 'ticker' (formato EODHD es. 'AAPL.US')
    """
    if universe.empty:
        return pd.DataFrame()

    # Costruisci ticker in formato EODHD (CODE.US)
    universe = universe.copy()
    universe["ticker"] = universe["Code"] + ".US"

    # Merge con fondamentali per Market Cap e settore
    if not fundamentals_cache.empty and "ticker" in fundamentals_cache.columns:
        universe = universe.merge(
            fundamentals_cache[["ticker", "market_cap", "gic_sector"]],
            on="ticker",
            how="left",
        )
        # Filtro Market Cap
        universe = universe[
            universe["market_cap"].notna()
            & (universe["market_cap"] >= MIN_MARKET_CAP_USD)
        ]
        # Filtro settore
        universe = universe[
            ~universe["gic_sector"].isin(EXCLUDED_GICS_SECTORS)
        ]
        logger.info(f"Dopo filtro Market Cap + settore: {len(universe):,}")

    # Merge con prezzi per ADV
    if not prices_cache.empty and "ticker" in prices_cache.columns:
        # Calcola ADV 20gg per ogni ticker
        adv = (
            prices_cache.groupby("ticker")["volume"]
            .apply(lambda x: x.tail(20).mean())
            .reset_index()
            .rename(columns={"volume": "adv_20d"})
        )
        universe = universe.merge(adv, on="ticker", how="left")
        universe = universe[
            universe["adv_20d"].notna()
            & (universe["adv_20d"] >= MIN_AVG_DAILY_VOLUME)
        ]
        logger.info(f"Dopo filtro ADV: {len(universe):,}")

    return universe.reset_index(drop=True)


def load_universe_cache() -> pd.DataFrame:
    """
    Carica l'universo filtrato dalla cache Parquet.

    Returns:
        DataFrame con l'universo, o DataFrame vuoto se la cache non esiste.
    """
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    logger.warning("Cache universo non trovata. Eseguire lo scheduler per popolarla.")
    return pd.DataFrame()


def save_universe_cache(df: pd.DataFrame) -> None:
    """
    Salva l'universo filtrato su cache Parquet.

    Args:
        df: DataFrame universo da salvare
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Universo salvato: {len(df):,} ticker → {CACHE_PATH}")
