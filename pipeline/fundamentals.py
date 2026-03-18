"""
pipeline/fundamentals.py — Motore Fondamentale del framework Fallen Angels.

Calcola per ogni ticker:
    1. Piotroski F-Score TTM (9 componenti, soglia >= 7)
    2. FCF Yield TTM (soglia > 5%, rolling 8Q per settori ciclici)
    3. Interest Coverage Ratio TTM con soglie settoriali

Fonte dati: EODHD /api/fundamentals/{TICKER}?filter=...
Costo API: 10 call per ticker (fetched settimanalmente, non giornalmente).

Endpoint:
    GET /api/fundamentals/{TICKER}.US
        ?filter=General,Highlights,Financials::Balance_Sheet::quarterly,
                Financials::Income_Statement::quarterly,
                Financials::Cash_Flow::quarterly
        &api_token={KEY}&fmt=json
"""

import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# === CACHE — metriche calcolate ===
CACHE_PATH = Path("cache/fundamentals.parquet")

# === RAW CACHE — dati trimestrali grezzi (separazione acquisition / computation) ===
# Scaricati una volta a settimana. Ricalcolo delle metriche = zero chiamate API.
RAW_IS_CACHE_PATH   = Path("cache/raw_is.parquet")    # Income Statement quarterly
RAW_BS_CACHE_PATH   = Path("cache/raw_bs.parquet")    # Balance Sheet quarterly
RAW_CF_CACHE_PATH   = Path("cache/raw_cf.parquet")    # Cash Flow quarterly
RAW_META_CACHE_PATH = Path("cache/raw_meta.parquet")  # Metadati ticker (sector, mcap)

# Colonne da conservare per sezione (solo quelle usate nei calcoli)
RAW_IS_COLS = ["totalRevenue", "netIncome", "grossProfit", "ebit", "interestExpense"]
RAW_BS_COLS = ["totalAssets", "totalCurrentAssets", "totalCurrentLiabilities",
               "longTermDebt", "commonStockSharesOutstanding"]
RAW_CF_COLS = ["totalCashFromOperatingActivities", "capitalExpenditures"]
RAW_KEEP_QUARTERS = 16   # 4 anni di dati trimestrali

# === SOGLIE ICR PER GICS SECTOR ===
# Settori con alta leva strutturale ricevono soglie più basse (risk-adjusted)
ICR_SECTOR_THRESHOLDS = {
    "Information Technology": 5.0,
    "Technology":             5.0,
    "Health Care":            5.0,
    "Consumer Discretionary": 5.0,
    "Consumer Staples":       4.0,
    "Industrials":            4.0,
    "Communication Services": 4.0,
    "Materials":              3.0,
    "Energy":                 3.0,
    "Utilities":              2.0,
    "Real Estate":            2.0,   # non dovrebbe apparire (escluso dall'universo)
}
ICR_DEFAULT_THRESHOLD = 4.0  # fallback per settori non mappati

# Settori ciclici: FCF Yield calcolato come media su 8 trimestri invece di 4
CYCLICAL_SECTORS = {"Energy", "Materials", "Industrials", "Consumer Discretionary"}

# Rate limiting: pausa tra chiamate successive
REQUEST_DELAY_SEC = 0.07   # ~14 req/sec → sotto il limite di 1000/min


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _safe_float(value) -> Optional[float]:
    """Converte un valore a float, restituisce None se non convertibile."""
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator) -> Optional[float]:
    """Divisione sicura: restituisce None se denominatore è 0 o None."""
    n = _safe_float(numerator)
    d = _safe_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _sum_quarters(df: pd.DataFrame, col: str, start: int, end: int) -> Optional[float]:
    """
    Somma i valori di una colonna per i trimestri [start:end] (indice 0 = più recente).

    Args:
        df:    DataFrame con indice temporale decrescente (più recente prima)
        col:   Nome della colonna da sommare
        start: Indice di inizio (incluso)
        end:   Indice di fine (escluso)

    Returns:
        Somma dei valori, o None se dati insufficienti o tutti NaN.
    """
    if col not in df.columns or len(df) < end:
        return None
    vals = df[col].iloc[start:end].apply(_safe_float)
    if vals.isna().all():
        return None
    return vals.sum(skipna=True)


def _get_quarter_val(df: pd.DataFrame, col: str, idx: int) -> Optional[float]:
    """
    Estrae il valore di una colonna per il trimestre all'indice idx.

    Args:
        df:  DataFrame con indice temporale decrescente
        col: Nome della colonna
        idx: Indice trimestrale (0 = più recente)

    Returns:
        Valore float, o None se non disponibile.
    """
    if col not in df.columns or len(df) <= idx:
        return None
    return _safe_float(df[col].iloc[idx])


# ============================================================
# FETCH FONDAMENTALI DA EODHD
# ============================================================

def fetch_fundamentals(ticker: str, api_key: str) -> Optional[dict]:
    """
    Scarica i dati fondamentali da EODHD usando il filtro sezioni per ridurre il payload.

    Costo: 10 API call per richiesta (indipendentemente dal filtro).

    Args:
        ticker:  Simbolo EODHD (es. 'AAPL.US')
        api_key: Chiave API EODHD

    Returns:
        Dict JSON con sezioni General, Highlights, Financials, o None se errore.
    """
    url = f"https://eodhd.com/api/fundamentals/{ticker}"
    params = {
        "api_token": api_key,
        "fmt": "json",
        # NOTA: il filter va fermato al primo livello per Financials.
        # Specificare sottopath (es. Financials::Balance_Sheet::quarterly) fa sì che
        # EODHD restituisca solo il nodo foglia, distruggendo la struttura annidata
        # che _parse_financial_section si aspetta. Usiamo "Financials" senza sottopath.
        # Usa solo nomi di sezione top-level (senza ::sottopath).
        # EODHD con path specifici (es. Highlights::MarketCapitalization) restituisce
        # il valore piatto senza il wrapper di sezione, rompendo raw.get("Highlights").
        "filter": "General,Highlights,Financials",
    }
    try:
        resp = requests.get(url, params=params, timeout=45)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.debug(f"Ticker non trovato: {ticker}")
        else:
            logger.warning(f"HTTP {e.response.status_code} per {ticker}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore rete per {ticker}: {e}")
        return None


# ============================================================
# PARSING DATI BILANCIO EODHD
# ============================================================

def _parse_financial_section(raw: dict, section: str, period: str = "quarterly") -> pd.DataFrame:
    """
    Estrae e trasforma una sezione del bilancio EODHD in DataFrame.

    La risposta EODHD ha struttura:
        {'Financials': {'Income_Statement': {'quarterly': {'2024-09-30': {...}, ...}}}}

    Args:
        raw:     Dict JSON fondamentali EODHD
        section: 'Income_Statement', 'Balance_Sheet', 'Cash_Flow'
        period:  'quarterly' o 'annual'

    Returns:
        DataFrame con date come indice decrescente e metriche come colonne.
        Colonne numeriche già convertite a float.
    """
    try:
        data = raw["Financials"][section][period]
    except (KeyError, TypeError):
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index(ascending=False)

    # Converti tutte le colonne a numeric (i valori EODHD sono spesso stringhe)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# CALCOLO PIOTROSKI F-SCORE TTM
# ============================================================

def compute_fscore_ttm(
    is_q: pd.DataFrame,   # Income Statement quarterly
    bs_q: pd.DataFrame,   # Balance Sheet quarterly
    cf_q: pd.DataFrame,   # Cash Flow quarterly
) -> Optional[dict]:
    """
    Calcola il Piotroski F-Score su base TTM (Trailing Twelve Months).

    TTM Flow metrics: somma degli ultimi 4 trimestri (indici 0-3)
    Prior TTM: somma dei trimestri 4-7 (confronto year-over-year)
    Stock metrics: valore del trimestre più recente (indice 0) vs 1 anno fa (indice 4)

    Richiede almeno 8 trimestri di dati per il confronto year-over-year.

    Returns:
        Dict con 'f_score' (int 0-9) e tutti i componenti, o None se dati insufficienti.
    """
    # Verifica disponibilità dati minimi
    min_quarters = 8
    if is_q.empty or bs_q.empty or cf_q.empty:
        return None
    if len(is_q) < min_quarters or len(bs_q) < min_quarters or len(cf_q) < min_quarters:
        return None

    # === METRICHE TTM (somma 4 trimestri recenti) ===
    ttm_revenue    = _sum_quarters(is_q, "totalRevenue", 0, 4)
    ttm_net_income = _sum_quarters(is_q, "netIncome", 0, 4)
    ttm_gross_profit = _sum_quarters(is_q, "grossProfit", 0, 4)
    ttm_ebit       = _sum_quarters(is_q, "ebit", 0, 4)
    ttm_cfo        = _sum_quarters(cf_q, "totalCashFromOperatingActivities", 0, 4)
    ttm_capex      = _sum_quarters(cf_q, "capitalExpenditures", 0, 4)

    # === METRICHE PRIOR TTM (trimestri 4-7, anno precedente) ===
    prior_revenue    = _sum_quarters(is_q, "totalRevenue", 4, 8)
    prior_net_income = _sum_quarters(is_q, "netIncome", 4, 8)
    prior_gross_profit = _sum_quarters(is_q, "grossProfit", 4, 8)

    # === BALANCE SHEET (stock metrics, punto nel tempo) ===
    # Q0 = trimestre più recente, Q4 = 1 anno fa, Q8 = 2 anni fa
    total_assets_q0  = _get_quarter_val(bs_q, "totalAssets", 0)
    total_assets_q4  = _get_quarter_val(bs_q, "totalAssets", 4)
    total_assets_q8  = _get_quarter_val(bs_q, "totalAssets", 8) if len(bs_q) > 8 else None

    curr_assets_q0   = _get_quarter_val(bs_q, "totalCurrentAssets", 0)
    curr_assets_q4   = _get_quarter_val(bs_q, "totalCurrentAssets", 4)
    curr_liab_q0     = _get_quarter_val(bs_q, "totalCurrentLiabilities", 0)
    curr_liab_q4     = _get_quarter_val(bs_q, "totalCurrentLiabilities", 4)

    lt_debt_q0       = _get_quarter_val(bs_q, "longTermDebt", 0)
    lt_debt_q4       = _get_quarter_val(bs_q, "longTermDebt", 4)

    # Shares outstanding per verifica diluizione
    shares_q0 = _get_quarter_val(bs_q, "commonStockSharesOutstanding", 0)
    shares_q4 = _get_quarter_val(bs_q, "commonStockSharesOutstanding", 4)

    # === MEDIE ASSET PER ROA (riduce rumore da variazioni trimestrali) ===
    # Media tra inizio e fine del periodo TTM
    avg_assets_ttm   = _safe_div((total_assets_q0 or 0) + (total_assets_q4 or 0), 2) if (total_assets_q0 and total_assets_q4) else total_assets_q0
    avg_assets_prior = _safe_div((total_assets_q4 or 0) + (total_assets_q8 or 0), 2) if (total_assets_q4 and total_assets_q8) else total_assets_q4

    # === CALCOLO COMPONENTI F-SCORE ===

    # --- Gruppo A: Profittabilità ---
    roa_ttm   = _safe_div(ttm_net_income, avg_assets_ttm)
    roa_prior = _safe_div(prior_net_income, avg_assets_prior)

    # F_ROA: ROA positivo
    F_ROA = 1 if (roa_ttm is not None and roa_ttm > 0) else 0

    # F_CFO: Cash Flow Operativo positivo (più difficile da manipolare dell'utile netto)
    F_CFO = 1 if (ttm_cfo is not None and ttm_cfo > 0) else 0

    # F_DROA: ROA in miglioramento YoY
    F_DROA = 1 if (roa_ttm is not None and roa_prior is not None and roa_ttm > roa_prior) else 0

    # F_ACC: Qualità degli accruals — CFO/Assets > ROA (utili "reali", non contabili)
    cfo_to_assets = _safe_div(ttm_cfo, avg_assets_ttm)
    F_ACC = 1 if (cfo_to_assets is not None and roa_ttm is not None and cfo_to_assets > roa_ttm) else 0

    # --- Gruppo B: Leva, Liquidità, Finanziamento ---

    # F_DLEV: Leva finanziaria in calo (LT Debt / Total Assets ridotto YoY)
    lev_q0 = _safe_div(lt_debt_q0, total_assets_q0)
    lev_q4 = _safe_div(lt_debt_q4, total_assets_q4)
    F_DLEV = 1 if (lev_q0 is not None and lev_q4 is not None and lev_q0 < lev_q4) else 0

    # F_DLIQ: Current Ratio in miglioramento (liquidità a breve termine)
    cr_q0 = _safe_div(curr_assets_q0, curr_liab_q0)
    cr_q4 = _safe_div(curr_assets_q4, curr_liab_q4)
    F_DLIQ = 1 if (cr_q0 is not None and cr_q4 is not None and cr_q0 > cr_q4) else 0

    # F_EQ: Nessuna diluizione azionaria (azioni non aumentate nell'ultimo anno)
    # Tolleranza 0.5% per emissioni minori legate a stock option / ESPP
    F_EQ = 1 if (shares_q0 is not None and shares_q4 is not None and shares_q0 <= shares_q4 * 1.005) else 0

    # --- Gruppo C: Efficienza Operativa ---

    # F_DMAR: Gross Margin in miglioramento YoY
    margin_ttm   = _safe_div(ttm_gross_profit, ttm_revenue)
    margin_prior = _safe_div(prior_gross_profit, prior_revenue)
    F_DMAR = 1 if (margin_ttm is not None and margin_prior is not None and margin_ttm > margin_prior) else 0

    # F_DTURN: Asset Turnover in miglioramento YoY (efficienza uso degli asset)
    turn_ttm   = _safe_div(ttm_revenue, avg_assets_ttm)
    turn_prior = _safe_div(prior_revenue, avg_assets_prior)
    F_DTURN = 1 if (turn_ttm is not None and turn_prior is not None and turn_ttm > turn_prior) else 0

    # === SCORE TOTALE ===
    score = F_ROA + F_CFO + F_DROA + F_ACC + F_DLEV + F_DLIQ + F_EQ + F_DMAR + F_DTURN

    return {
        "f_score":        score,
        "F_ROA":          F_ROA,
        "F_CFO":          F_CFO,
        "F_DROA":         F_DROA,
        "F_ACC":          F_ACC,
        "F_DLEV":         F_DLEV,
        "F_DLIQ":         F_DLIQ,
        "F_EQ":           F_EQ,
        "F_DMAR":         F_DMAR,
        "F_DTURN":        F_DTURN,
        "roa_ttm":        round(roa_ttm, 4) if roa_ttm else None,
        "cfo_ttm":        ttm_cfo,
        "current_ratio":  round(cr_q0, 2) if cr_q0 else None,
        "gross_margin":   round(margin_ttm, 4) if margin_ttm else None,
        "asset_turnover": round(turn_ttm, 4) if turn_ttm else None,
    }


# ============================================================
# CALCOLO FCF YIELD
# ============================================================

def compute_fcf_ttm(
    cf_q: pd.DataFrame,
    gic_sector: str = "",
) -> Optional[float]:
    """
    Calcola il Free Cash Flow TTM in valore assoluto (USD).

    Separato da compute_fcf_yield per permettere il caching del valore assoluto.
    Questo consente di ricalcolare fcf_yield aggiornando solo il market_cap
    (da bulk EOD, 3 chiamate) senza re-fetchare tutti i fondamentali.

    Returns:
        FCF annualizzato in USD, o None se dati insufficienti.
    """
    if cf_q.empty:
        return None

    is_cyclical = any(s in gic_sector for s in CYCLICAL_SECTORS)
    window = 8 if is_cyclical else 4
    if len(cf_q) < window:
        window = len(cf_q)

    cfo   = _sum_quarters(cf_q, "totalCashFromOperatingActivities", 0, window)
    capex = _sum_quarters(cf_q, "capitalExpenditures", 0, window)

    if cfo is None:
        return None

    fcf = cfo + (capex if capex is not None else 0)

    # Settori ciclici: annualizza la media biennale
    if is_cyclical and window == 8:
        fcf = fcf / 2

    return fcf


def compute_fcf_yield(
    cf_q: pd.DataFrame,
    market_cap: Optional[float],
    gic_sector: str = "",
) -> Optional[float]:
    """
    Calcola il Free Cash Flow Yield TTM = FCF_TTM / Market Cap.

    Per settori ciclici usa la media 8 trimestri per ridurre la volatilità ciclica.

    Returns:
        FCF Yield come float (es. 0.07 = 7%), o None se dati insufficienti.
    """
    if market_cap is None or market_cap <= 0:
        return None
    fcf = compute_fcf_ttm(cf_q, gic_sector)
    return _safe_div(fcf, market_cap)


# ============================================================
# CALCOLO INTEREST COVERAGE RATIO
# ============================================================

def compute_icr(
    is_q: pd.DataFrame,
    gic_sector: str = "",
) -> dict:
    """
    Calcola l'Interest Coverage Ratio TTM e verifica la soglia settoriale.

    ICR = EBIT_TTM / InterestExpense_TTM
    Usa EBIT (non EBITDA) per non escludere ammortamenti reali.

    Casi speciali:
        - Interest Expense = 0 (no debito): ICR = +inf, score massimo
        - EBIT negativo: ICR = 0 (peggiore scenario)

    Args:
        is_q:       DataFrame Income Statement quarterly
        gic_sector: GICS Sector per determinare la soglia

    Returns:
        Dict con 'icr' (float), 'threshold' (float), 'passes' (bool)
    """
    threshold = ICR_SECTOR_THRESHOLDS.get(gic_sector, ICR_DEFAULT_THRESHOLD)

    if is_q.empty or len(is_q) < 4:
        return {"icr": None, "threshold": threshold, "passes": False}

    ttm_ebit         = _sum_quarters(is_q, "ebit", 0, 4)
    ttm_interest_exp = _sum_quarters(is_q, "interestExpense", 0, 4)

    # Caso: nessun debito (interest expense assente o zero)
    if ttm_interest_exp is None or ttm_interest_exp == 0:
        return {"icr": float("inf"), "threshold": threshold, "passes": True}

    # Caso: EBIT negativo → ICR = 0 (non copre gli interessi)
    if ttm_ebit is None or ttm_ebit <= 0:
        return {"icr": 0.0, "threshold": threshold, "passes": False}

    # InterestExpense in EODHD può essere positivo o negativo a seconda del ticker
    # Usiamo il valore assoluto per uniformità
    icr = abs(ttm_ebit) / abs(ttm_interest_exp)

    return {
        "icr":       round(icr, 2),
        "threshold": threshold,
        "passes":    icr >= threshold,
    }


# ============================================================
# PIPELINE PRINCIPALE: PROCESSA UN SINGOLO TICKER
# ============================================================

def process_ticker_fundamentals(
    ticker: str,
    api_key: str,
    _raw: Optional[dict] = None,
) -> Optional[dict]:
    """
    Calcola tutte le metriche fondamentali per un singolo ticker.

    Args:
        ticker:  Simbolo EODHD (es. 'AAPL.US')
        api_key: Chiave API EODHD
        _raw:    JSON già fetchato (opzionale). Se None viene chiamata fetch_fundamentals.
                 Usato da process_fundamentals_batch per evitare doppia chiamata API.

    Returns:
        Dict con metriche fondamentali, o None se il ticker non è processabile.
    """
    raw = _raw if _raw is not None else fetch_fundamentals(ticker, api_key)
    if raw is None:
        return None

    # === Dati generali ===
    general    = raw.get("General", {}) or {}
    highlights = raw.get("Highlights", {}) or {}

    gic_sector = (
        general.get("GicSector")
        or general.get("Sector")
        or ""
    )
    market_cap = _safe_float(highlights.get("MarketCapitalization"))

    # === Parsing sezioni bilancio ===
    is_q = _parse_financial_section(raw, "Income_Statement", "quarterly")
    bs_q = _parse_financial_section(raw, "Balance_Sheet", "quarterly")
    cf_q = _parse_financial_section(raw, "Cash_Flow", "quarterly")

    if is_q.empty or bs_q.empty or cf_q.empty:
        return None

    # === Calcolo metriche ===
    fscore_data = compute_fscore_ttm(is_q, bs_q, cf_q)
    fcf_ttm_abs = compute_fcf_ttm(cf_q, gic_sector)      # valore assoluto — cachato per patch a basso costo
    fcf_yield   = _safe_div(fcf_ttm_abs, market_cap) if (fcf_ttm_abs is not None and market_cap and market_cap > 0) else None
    icr_data    = compute_icr(is_q, gic_sector)

    if fscore_data is None:
        return None

    record = {
        "ticker":      ticker,
        "name":        general.get("Name", ""),
        "gic_sector":  gic_sector,
        "gic_group":   general.get("GicGroup", ""),
        "market_cap":  market_cap,
        "is_delisted": general.get("IsDelisted", False),
        # F-Score
        **{k: v for k, v in fscore_data.items()},
        # FCF — valore assoluto TTM cachato per permettere patch market_cap a 3 API call
        "fcf_ttm":           round(fcf_ttm_abs) if fcf_ttm_abs is not None else None,
        # FCF Yield
        "fcf_yield":         round(fcf_yield, 4) if fcf_yield is not None else None,
        "fcf_yield_passes":  (fcf_yield is not None and fcf_yield > 0.05),
        # ICR
        "icr":               icr_data["icr"],
        "icr_threshold":     icr_data["threshold"],
        "icr_passes":        icr_data["passes"],
        # Flag whitelist complessivo
        "in_whitelist": (
            fscore_data["f_score"] >= 7
            and (fcf_yield is not None and fcf_yield > 0.05)
            and icr_data["passes"]
        ),
    }
    return record


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_fundamentals_batch(
    tickers: list,
    api_key: str,
    delay: float = REQUEST_DELAY_SEC,
) -> pd.DataFrame:
    """
    Processa un batch di ticker in sequenza con rate limiting.

    Args:
        tickers: Lista di ticker in formato EODHD (es. ['AAPL.US', 'MSFT.US'])
        api_key: Chiave API EODHD
        delay:   Pausa in secondi tra chiamate successive

    Returns:
        DataFrame con una riga per ticker (solo quelli processati con successo).
    """
    records     = []
    raw_is_rows = []
    raw_bs_rows = []
    raw_cf_rows = []
    meta_rows   = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            logger.info(f"Fondamentali: {i}/{total} ticker processati...")

        # === Fetch una volta sola — riuso per raw cache E metriche calcolate ===
        raw = fetch_fundamentals(ticker, api_key)
        if raw is None:
            time.sleep(delay)
            continue

        # --- Accumula dati grezzi per raw cache ---
        general    = raw.get("General", {}) or {}
        highlights = raw.get("Highlights", {}) or {}
        gic_sector = general.get("GicSector") or general.get("Sector") or ""
        market_cap = _safe_float(highlights.get("MarketCapitalization"))

        is_q = _parse_financial_section(raw, "Income_Statement", "quarterly")
        bs_q = _parse_financial_section(raw, "Balance_Sheet",    "quarterly")
        cf_q = _parse_financial_section(raw, "Cash_Flow",        "quarterly")

        if not is_q.empty and not bs_q.empty and not cf_q.empty:
            for date_idx, row in is_q.head(RAW_KEEP_QUARTERS).iterrows():
                r = {"ticker": ticker, "date": date_idx}
                for col in RAW_IS_COLS:
                    v = row.get(col)
                    r[col] = float(v) if v is not None and not pd.isna(v) else None
                raw_is_rows.append(r)

            for date_idx, row in bs_q.head(RAW_KEEP_QUARTERS).iterrows():
                r = {"ticker": ticker, "date": date_idx}
                for col in RAW_BS_COLS:
                    v = row.get(col)
                    r[col] = float(v) if v is not None and not pd.isna(v) else None
                raw_bs_rows.append(r)

            for date_idx, row in cf_q.head(RAW_KEEP_QUARTERS).iterrows():
                r = {"ticker": ticker, "date": date_idx}
                for col in RAW_CF_COLS:
                    v = row.get(col)
                    r[col] = float(v) if v is not None and not pd.isna(v) else None
                raw_cf_rows.append(r)

            meta_rows.append({
                "ticker":     ticker,
                "name":       general.get("Name", ""),
                "gic_sector": gic_sector,
                "gic_group":  general.get("GicGroup", ""),
                "market_cap": market_cap,
                "is_delisted": bool(general.get("IsDelisted", False)),
            })

        # --- Calcola metriche (riusa raw già fetchato) ---
        record = process_ticker_fundamentals(ticker, api_key, _raw=raw)
        if record is not None:
            records.append(record)

        time.sleep(delay)

    # Salva raw cache (dati grezzi per ricalcolo futuro senza API)
    if raw_is_rows:
        save_raw_cache(raw_is_rows, raw_bs_rows, raw_cf_rows, meta_rows)

    if not records:
        logger.warning("Nessun record fondamentale generato.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"Fondamentali processati: {len(df):,} ticker validi su {total:,}")
    return df


# ============================================================
# HISTORICAL F-SCORE PER PHASE 2 MONITOR (tab 3)
# ============================================================

def compute_rolling_fscore_history(ticker: str, api_key: str, n_quarters: int = 8) -> pd.DataFrame:
    """
    Calcola l'F-Score per ogni trimestre degli ultimi n_quarters trimestri.
    Usato dalla Tab 3 (Phase 2 Monitor) per mostrare il trend del deterioramento.

    Per ogni data di riferimento trimestrale, usa i dati disponibili fino a quel momento
    (simulazione point-in-time semplificata: usa i dati del trimestre stesso come anchor).

    Args:
        ticker:     Simbolo EODHD
        api_key:    Chiave API
        n_quarters: Numero di trimestri da visualizzare (default 8 = 2 anni)

    Returns:
        DataFrame con colonne: date, f_score, F_ROA, F_CFO, ..., fcf_yield, icr
    """
    raw = fetch_fundamentals(ticker, api_key)
    if raw is None:
        return pd.DataFrame()

    general    = raw.get("General", {}) or {}
    highlights = raw.get("Highlights", {}) or {}
    gic_sector = general.get("GicSector") or general.get("Sector") or ""
    market_cap = _safe_float(highlights.get("MarketCapitalization"))

    is_q = _parse_financial_section(raw, "Income_Statement", "quarterly")
    bs_q = _parse_financial_section(raw, "Balance_Sheet", "quarterly")
    cf_q = _parse_financial_section(raw, "Cash_Flow", "quarterly")

    if is_q.empty or bs_q.empty or cf_q.empty or len(is_q) < n_quarters + 4:
        return pd.DataFrame()

    # Calcola F-Score per ogni trimestre disponibile (shift rolling)
    history = []
    for q in range(n_quarters):
        # Usa i dati da trimestre q in avanti (simulazione rolling)
        is_slice = is_q.iloc[q:].reset_index(drop=True)
        bs_slice = bs_q.iloc[q:].reset_index(drop=True)
        cf_slice = cf_q.iloc[q:].reset_index(drop=True)

        anchor_date = is_q.index[q]  # data del trimestre
        fscore_data = compute_fscore_ttm(is_slice, bs_slice, cf_slice)
        fcf_yield   = compute_fcf_yield(cf_slice, market_cap, gic_sector)
        icr_data    = compute_icr(is_slice, gic_sector)

        if fscore_data is None:
            continue

        record = {
            "date": anchor_date,
            **{k: v for k, v in fscore_data.items()},
            "fcf_yield": round(fcf_yield, 4) if fcf_yield is not None else None,
            "icr":       icr_data["icr"],
            "icr_threshold": icr_data["threshold"],
        }
        history.append(record)

    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history).sort_values("date")
    return df


# ============================================================
# CACHE I/O
# ============================================================

def load_fundamentals_cache() -> pd.DataFrame:
    """Carica la cache fondamentali da Parquet."""
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    logger.warning("Cache fondamentali non trovata. Eseguire lo scheduler.")
    return pd.DataFrame()


def save_fundamentals_cache(df: pd.DataFrame) -> None:
    """Salva la cache fondamentali su Parquet."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Fondamentali salvati: {len(df):,} ticker → {CACHE_PATH}")


# ============================================================
# RAW CACHE I/O — dati trimestrali grezzi
# ============================================================

def raw_cache_exists() -> bool:
    """Verifica se tutti i file della raw cache sono presenti."""
    return all(p.exists() for p in [
        RAW_IS_CACHE_PATH, RAW_BS_CACHE_PATH,
        RAW_CF_CACHE_PATH, RAW_META_CACHE_PATH,
    ])


def save_raw_cache(
    is_rows:   list,
    bs_rows:   list,
    cf_rows:   list,
    meta_rows: list,
) -> None:
    """Salva i dati trimestrali grezzi in 4 file Parquet separati."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(is_rows).to_parquet(RAW_IS_CACHE_PATH,   index=False)
    pd.DataFrame(bs_rows).to_parquet(RAW_BS_CACHE_PATH,   index=False)
    pd.DataFrame(cf_rows).to_parquet(RAW_CF_CACHE_PATH,   index=False)
    pd.DataFrame(meta_rows).to_parquet(RAW_META_CACHE_PATH, index=False)
    logger.info(f"Raw cache salvata: {len(meta_rows):,} ticker — {RAW_IS_CACHE_PATH.parent}")


def load_raw_cache() -> Optional[dict]:
    """
    Carica la raw cache. Restituisce dict con chiavi 'is', 'bs', 'cf', 'meta',
    o None se uno dei file manca.
    """
    if not raw_cache_exists():
        return None
    return {
        "is":   pd.read_parquet(RAW_IS_CACHE_PATH),
        "bs":   pd.read_parquet(RAW_BS_CACHE_PATH),
        "cf":   pd.read_parquet(RAW_CF_CACHE_PATH),
        "meta": pd.read_parquet(RAW_META_CACHE_PATH),
    }


def recompute_from_raw_cache(
    market_cap_override: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Ricalcola tutte le metriche fondamentali dalla raw cache. ZERO chiamate API.

    Usato per:
      - Correggere bug nelle formule senza re-fetch
      - Aggiornare market_cap da bulk EOD (3 chiamate) e ricalcolare fcf_yield
      - Qualsiasi modifica alla logica di screening senza costi

    Args:
        market_cap_override: dict {ticker_short: market_cap} per aggiornare il
                             market_cap dalla raw cache (es. da bulk EOD).
                             Se None usa il market_cap già in cache.

    Returns:
        DataFrame fondamentali con metriche aggiornate, pronto per save_fundamentals_cache.
    """
    raw = load_raw_cache()
    if raw is None:
        logger.warning("Raw cache non trovata — impossibile ricalcolare senza API.")
        return pd.DataFrame()

    raw_is   = raw["is"]
    raw_bs   = raw["bs"]
    raw_cf   = raw["cf"]
    meta_df  = raw["meta"]

    # Converti date
    for df in (raw_is, raw_bs, raw_cf):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    records = []
    tickers = meta_df["ticker"].tolist()

    for ticker in tickers:
        meta_row   = meta_df[meta_df["ticker"] == ticker].iloc[0]
        gic_sector = str(meta_row.get("gic_sector", "") or "")
        market_cap = meta_row.get("market_cap")

        # Aggiorna market_cap se fornito override (es. da bulk EOD, 3 API call)
        if market_cap_override:
            ticker_short = ticker.split(".")[0]
            override = market_cap_override.get(ticker_short)
            if override and override > 0:
                market_cap = float(override)

        # Ricostruisci DataFrame quarterly da raw cache
        def _to_quarterly(df_raw):
            sub = (df_raw[df_raw["ticker"] == ticker]
                   .drop(columns=["ticker"])
                   .set_index("date")
                   .sort_index(ascending=False))
            for col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            return sub

        is_q = _to_quarterly(raw_is)
        bs_q = _to_quarterly(raw_bs)
        cf_q = _to_quarterly(raw_cf)

        if is_q.empty or bs_q.empty or cf_q.empty:
            continue

        fscore_data = compute_fscore_ttm(is_q, bs_q, cf_q)
        if fscore_data is None:
            continue

        fcf_ttm_abs = compute_fcf_ttm(cf_q, gic_sector)
        fcf_yield   = (_safe_div(fcf_ttm_abs, market_cap)
                       if (fcf_ttm_abs is not None and market_cap and market_cap > 0)
                       else None)
        icr_data    = compute_icr(is_q, gic_sector)

        records.append({
            "ticker":      ticker,
            "name":        str(meta_row.get("name", "") or ""),
            "gic_sector":  gic_sector,
            "gic_group":   str(meta_row.get("gic_group", "") or ""),
            "market_cap":  market_cap,
            "is_delisted": bool(meta_row.get("is_delisted", False)),
            **{k: v for k, v in fscore_data.items()},
            "fcf_ttm":          round(fcf_ttm_abs) if fcf_ttm_abs is not None else None,
            "fcf_yield":        round(fcf_yield, 4) if fcf_yield is not None else None,
            "fcf_yield_passes": (fcf_yield is not None and fcf_yield > 0.05),
            "icr":              icr_data["icr"],
            "icr_threshold":    icr_data["threshold"],
            "icr_passes":       icr_data["passes"],
            "in_whitelist": (
                fscore_data["f_score"] >= 7
                and (fcf_yield is not None and fcf_yield > 0.05)
                and icr_data["passes"]
            ),
        })

    if not records:
        logger.warning("recompute_from_raw_cache: nessun record prodotto.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    wl = int((df["in_whitelist"] == True).sum())
    logger.info(f"Ricalcolo da raw cache: {len(df):,} ticker — whitelist: {wl:,}")
    return df
