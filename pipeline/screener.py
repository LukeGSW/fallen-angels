"""
pipeline/screener.py — Combinazione dei motori Fondamentale e Tecnico.

Produce due output:
    1. Whitelist: tutti i ticker che superano i filtri fondamentali
       (F-Score >= 7, FCF Yield > 5%, ICR > soglia settoriale)
    2. Candidati operativi: subset della whitelist con trigger tecnico attivo
       (Z-Score <= -2.5, Volume Ratio >= 1.5x, nessun earnings imminente)

Entrambi vengono persistiti in cache/screener_results.parquet.

Architettura pre-computata:
    - Questo modulo viene eseguito ogni sera dallo scheduler.py
    - La Streamlit app (app.py) legge SOLO dalla cache, senza ricalcoli
"""

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Set

from pipeline.fundamentals import load_fundamentals_cache
from pipeline.technicals import load_technicals_cache
from pipeline.earnings import load_earnings_cache

logger = logging.getLogger(__name__)

# === CACHE OUTPUT ===
SCREENER_CACHE_PATH = Path("cache/screener_results.parquet")
METADATA_PATH       = Path("cache/last_update.json")

# === SOGLIE FONDAMENTALI ===
FSCORE_MIN     = 7
FCF_YIELD_MIN  = 0.05   # 5%

# === SOGLIE TECNICHE ===
ZSCORE_TRIGGER     = -2.5
VOLUME_RATIO_MIN   = 1.5


def run_screener(
    fundamentals: pd.DataFrame = None,
    technicals: pd.DataFrame   = None,
    earnings_exclusions: Set[str] = None,
) -> dict:
    """
    Esegue lo screener completo combinando i tre motori.

    Se i DataFrame non sono forniti esplicitamente, vengono caricati dalla cache.

    Args:
        fundamentals:        DataFrame fondamentali (da pipeline/fundamentals.py)
        technicals:          DataFrame tecnici (da pipeline/technicals.py)
        earnings_exclusions: Set di ticker da escludere (da pipeline/earnings.py)

    Returns:
        Dict con chiavi:
            'whitelist':   DataFrame ticker con fondamentali qualificati
            'candidates':  DataFrame ticker con trigger tecnico attivo
            'run_date':    Timestamp dell'esecuzione
            'stats':       Dict con statistiche aggregate
    """
    # Carica dalla cache se non forniti
    if fundamentals is None:
        fundamentals = load_fundamentals_cache()
    if technicals is None:
        technicals = load_technicals_cache()
    if earnings_exclusions is None:
        earnings_exclusions = load_earnings_cache()

    if fundamentals.empty:
        logger.error("Cache fondamentali vuota. Eseguire lo scheduler con refresh fondamentali.")
        return {"whitelist": pd.DataFrame(), "candidates": pd.DataFrame(), "stats": {}}

    # ============================================================
    # STEP 1: WHITELIST — filtro fondamentale
    # ============================================================
    whitelist = fundamentals[
        (fundamentals["in_whitelist"] == True)
        & (fundamentals["is_delisted"] != True)
    ].copy()

    # Rinomina colonne per la UI (più leggibili)
    logger.info(f"Whitelist fondamentale: {len(whitelist):,} ticker")

    # ============================================================
    # STEP 2: MERGE CON TECNICI
    # ============================================================
    if not technicals.empty and "ticker" in technicals.columns:
        whitelist = whitelist.merge(technicals, on="ticker", how="left")
    else:
        logger.warning("Cache tecnici vuota — candidati non calcolabili.")
        # Aggiungi colonne tecnici con valori None per consistenza
        tech_cols = ["price", "sma50", "sma200", "z_score", "volume",
                     "adv20", "volume_ratio", "days_below_sma200",
                     "is_trigger", "time_stop_alert", "pct_change_1d", "last_date"]
        for col in tech_cols:
            whitelist[col] = None

    # ============================================================
    # STEP 3: CANDIDATI OPERATIVI — trigger tecnico + esclusione earnings
    # ============================================================
    if "is_trigger" in whitelist.columns and whitelist["is_trigger"].notna().any():
        candidates = whitelist[
            (whitelist["is_trigger"] == True)
            & (~whitelist["ticker"].isin(earnings_exclusions))
        ].copy()

        # Aggiunge flag earnings imminenti (utile per la UI)
        whitelist["earnings_soon"] = whitelist["ticker"].isin(earnings_exclusions)
        candidates["earnings_soon"] = False

    else:
        candidates = pd.DataFrame()
        whitelist["earnings_soon"] = whitelist["ticker"].isin(earnings_exclusions)

    logger.info(f"Candidati operativi: {len(candidates):,} ticker")

    # ============================================================
    # STEP 4: ORDINAMENTO E SELEZIONE COLONNE
    # ============================================================
    # Whitelist ordinata per F-Score DESC, poi FCF Yield DESC
    sort_cols_whitelist = []
    if "f_score" in whitelist.columns:
        sort_cols_whitelist.append(("f_score", False))
    if "fcf_yield" in whitelist.columns:
        sort_cols_whitelist.append(("fcf_yield", False))

    if sort_cols_whitelist:
        whitelist = whitelist.sort_values(
            [c for c, _ in sort_cols_whitelist],
            ascending=[a for _, a in sort_cols_whitelist],
        ).reset_index(drop=True)

    # Candidati ordinati per Z-Score ASC (più basso = più estremo = primo)
    if not candidates.empty and "z_score" in candidates.columns:
        candidates = candidates.sort_values("z_score", ascending=True).reset_index(drop=True)

    # ============================================================
    # STEP 5: STATISTICHE AGGREGATE
    # ============================================================
    stats = {
        "run_date":             datetime.now().isoformat(),
        "universe_size":        len(fundamentals),
        "whitelist_size":       len(whitelist),
        "candidates_count":     len(candidates),
        "earnings_exclusions":  len(earnings_exclusions),
        "avg_fscore_whitelist": round(whitelist["f_score"].mean(), 2) if "f_score" in whitelist.columns and not whitelist.empty else None,
        "avg_fcf_yield":        round(whitelist["fcf_yield"].mean() * 100, 2) if "fcf_yield" in whitelist.columns and not whitelist.empty else None,
    }

    # Breakdown per settore nella whitelist
    if "gic_sector" in whitelist.columns:
        sector_counts = whitelist["gic_sector"].value_counts().to_dict()
        stats["sector_breakdown"] = sector_counts

    logger.info(f"Screener completato: {stats}")
    return {
        "whitelist":  whitelist,
        "candidates": candidates,
        "run_date":   datetime.now(),
        "stats":      stats,
    }


def save_screener_results(results: dict) -> None:
    """
    Salva i risultati dello screener su Parquet e aggiorna i metadata.

    Struttura cache:
        cache/screener_results.parquet → whitelist completa con flag candidati
        cache/last_update.json         → timestamp e statistiche ultima esecuzione

    Args:
        results: Dict output di run_screener()
    """
    import json

    SCREENER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    whitelist  = results.get("whitelist", pd.DataFrame())
    candidates = results.get("candidates", pd.DataFrame())

    if whitelist.empty:
        logger.warning("Whitelist vuota, nessun dato da salvare.")
        return

    # Aggiunge flag is_candidate alla whitelist per la UI
    whitelist = whitelist.copy()
    if not candidates.empty and "ticker" in candidates.columns:
        whitelist["is_candidate"] = whitelist["ticker"].isin(candidates["ticker"])
    else:
        whitelist["is_candidate"] = False

    whitelist.to_parquet(SCREENER_CACHE_PATH, index=False)
    logger.info(f"Screener results salvati: {len(whitelist):,} ticker → {SCREENER_CACHE_PATH}")

    # Salva metadata (stats + timestamp)
    stats = results.get("stats", {})
    # Il sector_breakdown non è JSON-serializable se ha tipi numpy
    if "sector_breakdown" in stats:
        stats["sector_breakdown"] = {k: int(v) for k, v in stats["sector_breakdown"].items()}

    with open(METADATA_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def load_screener_results() -> pd.DataFrame:
    """
    Carica i risultati dello screener dalla cache.

    Returns:
        DataFrame con whitelist completa e flag is_candidate.
        DataFrame vuoto se la cache non esiste.
    """
    if SCREENER_CACHE_PATH.exists():
        return pd.read_parquet(SCREENER_CACHE_PATH)
    logger.warning("Cache screener non trovata. Eseguire lo scheduler.")
    return pd.DataFrame()


def load_screener_metadata() -> dict:
    """
    Carica i metadata dell'ultima esecuzione dello screener.

    Returns:
        Dict con statistiche e timestamp, o dict vuoto.
    """
    import json
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}
