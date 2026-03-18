"""
scheduler.py — Pipeline notturna pre-computazione Fallen Angels.

Eseguito ogni sera da GitHub Actions (cron 22:00 EET = 20:00 UTC, lun-ven).
Produce e aggiorna i file Parquet in cache/ che la Streamlit app legge.

Flusso di esecuzione:
    1. Carica configurazione e secrets
    2. Verifica se è domenica (refresh fondamentali completo) o giorno feriale
    3. [Domenica] Fetch universe + fondamentali per tutti i ticker (~50K calls)
    4. [Ogni sera] Update prezzi via bulk EOD + calcolo tecnici (~5K calls)
    5. Fetch earnings calendar per esclusioni (~1 call)
    6. Esegue screener e salva risultati
    7. Invia notifica Telegram (successo / errore)

Variabili d'ambiente richieste (GitHub Actions Secrets):
    EODHD_API_KEY   — chiave API EODHD
    TELEGRAM_TOKEN  — token bot Telegram (da @BotFather)
    TELEGRAM_CHAT_ID — ID chat personale Telegram
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup logging strutturato
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fallen_angels.scheduler")

import numpy as np

# Import pipeline modules
from pipeline.universe import (
    build_raw_universe,
    apply_quantitative_filters,
    save_universe_cache,
    load_universe_cache,
)
from pipeline.fundamentals import (
    process_fundamentals_batch,
    save_fundamentals_cache,
    load_fundamentals_cache,
    raw_cache_exists,
    recompute_from_raw_cache,
)
from pipeline.technicals import (
    update_prices_from_bulk,
    initialize_prices_history,
    compute_all_technicals,
    save_technicals_cache,
    load_prices_cache,
)
from pipeline.earnings import (
    build_earnings_exclusion_set,
    save_earnings_cache,
)
from pipeline.screener import (
    run_screener,
    save_screener_results,
)

# === CONFIGURAZIONE ===
FULL_REFRESH_DAY = 6       # 0=Lunedì, 6=Domenica
EXCHANGES        = ["NYSE", "NASDAQ", "AMEX"]
CACHE_DIR        = Path("cache")


# ============================================================
# TELEGRAM NOTIFICATIONS
# ============================================================

def send_telegram(token: str, chat_id: str, message: str, parse_mode: str = "Markdown") -> bool:
    """
    Invia un messaggio Telegram via Bot API.

    Args:
        token:      Token del bot Telegram
        chat_id:    ID della chat destinatario
        message:    Testo del messaggio (supporta Markdown)
        parse_mode: 'Markdown' o 'HTML'

    Returns:
        True se inviato con successo, False altrimenti.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore invio Telegram: {e}")
        return False


def notify_success(token: str, chat_id: str, stats: dict, elapsed_sec: float) -> None:
    """Notifica Telegram in caso di pipeline completata con successo."""
    run_date = datetime.now().strftime("%d %b %Y — %H:%M")
    candidates = stats.get("candidates_count", 0)
    whitelist_size = stats.get("whitelist_size", 0)
    universe_size = stats.get("universe_size", 0)
    minutes = int(elapsed_sec // 60)
    seconds = int(elapsed_sec % 60)

    # Emoji in base al numero di candidati
    status_emoji = "🟢" if candidates == 0 else "🔵"
    candidate_emoji = "📭" if candidates == 0 else "📬"

    # Mostra candidati se presenti
    candidate_section = f"\n{candidate_emoji} *Candidati operativi: {candidates}*"
    if candidates == 0:
        candidate_section += "\n_(nessun trigger attivo oggi)_"

    message = (
        f"{status_emoji} *Fallen Angels Pipeline — {run_date}*\n"
        f"{'─' * 35}\n"
        f"📊 Universo processato: {universe_size:,} ticker\n"
        f"✅ Whitelist fondamentale: {whitelist_size:,} titoli\n"
        f"🚫 Esclusi per earnings: {stats.get('earnings_exclusions', 0)}\n"
        f"{candidate_section}\n"
        f"{'─' * 35}\n"
        f"⏱ Durata: {minutes}m {seconds}s"
    )
    send_telegram(token, chat_id, message)


def notify_error(token: str, chat_id: str, step: str, error: str, run_url: str = "") -> None:
    """Notifica Telegram in caso di errore nella pipeline."""
    run_date = datetime.now().strftime("%d %b %Y — %H:%M")
    message = (
        f"🔴 *Fallen Angels Pipeline — ERRORE*\n"
        f"{'─' * 35}\n"
        f"📅 {run_date}\n"
        f"⚠️ Step fallito: `{step}`\n"
        f"❌ Errore: `{error[:300]}`\n"
    )
    if run_url:
        message += f"\n🔗 [Vedi log completo]({run_url})"
    send_telegram(token, chat_id, message)


# ============================================================
# PATCH MARKET CAP — recupero a basso costo via bulk EOD
# ============================================================

def _fetch_mcap_from_bulk_eod(api_key: str, exchanges: list = None) -> dict:
    """
    Scarica market_cap per tutti i ticker via bulk EOD. Costo: 3 chiamate API.
    Restituisce dict {ticker_short: market_cap}.
    """
    from pipeline.technicals import fetch_bulk_eod
    if exchanges is None:
        exchanges = EXCHANGES
    mcap_map = {}
    for exchange in exchanges:
        bulk = fetch_bulk_eod(exchange, api_key)
        if bulk.empty or "code" not in bulk.columns:
            continue
        mcap_col = "market_capitalization" if "market_capitalization" in bulk.columns else None
        if mcap_col is None:
            continue
        bulk[mcap_col] = pd.to_numeric(bulk[mcap_col], errors="coerce")
        for _, row in bulk.iterrows():
            code = str(row.get("code", "")).strip()
            mcap = row.get(mcap_col)
            if code and mcap and not np.isnan(float(mcap)) and float(mcap) > 0:
                mcap_map[code] = float(mcap)
    logger.info(f"market_cap da bulk EOD: {len(mcap_map):,} ticker ({len(exchanges)} exchange)")
    return mcap_map


def patch_market_cap_from_bulk_eod(
    fundamentals: pd.DataFrame,
    api_key: str,
    exchanges: list = None,
) -> pd.DataFrame:
    """
    Aggiorna market_cap e ricalcola fcf_yield / in_whitelist usando il bulk EOD.

    Costo: 3 chiamate API (una per exchange).
    NON ri-fetcha fondamentali — usa il valore fcf_ttm già cachato.

    Casi gestiti:
      - fcf_ttm presente nella cache → fcf_yield ricalcolato esattamente
      - fcf_ttm assente (cache corrotta da run precedenti) → whitelist provvisoria
        basata su F-Score >= 7 e ICR (senza FCF), segnalata nei log.
        Il prossimo refresh domenicale popola fcf_ttm correttamente.

    Args:
        fundamentals: DataFrame fondamentali dalla cache
        api_key:      Chiave API EODHD
        exchanges:    Lista exchange da interrogare

    Returns:
        DataFrame con market_cap, fcf_yield e in_whitelist aggiornati.
    """
    from pipeline.technicals import fetch_bulk_eod

    if exchanges is None:
        exchanges = EXCHANGES

    logger.info("Patch market_cap da bulk EOD — 3 chiamate API...")
    mcap_map: dict = {}

    for exchange in exchanges:
        bulk = fetch_bulk_eod(exchange, api_key)
        if bulk.empty or "code" not in bulk.columns:
            continue
        mcap_col = "market_capitalization" if "market_capitalization" in bulk.columns else None
        if mcap_col is None:
            logger.warning(f"market_capitalization assente nel bulk EOD {exchange} — skip")
            continue
        bulk[mcap_col] = pd.to_numeric(bulk[mcap_col], errors="coerce")
        for _, row in bulk.iterrows():
            code = str(row.get("code", "")).strip()
            mcap = row.get(mcap_col)
            if code and mcap and not np.isnan(mcap) and mcap > 0:
                mcap_map[code] = float(mcap)

    if not mcap_map:
        logger.warning("Nessun market_cap da bulk EOD — patch non applicata.")
        return fundamentals

    logger.info(f"market_cap da bulk EOD: {len(mcap_map):,} ticker")

    df = fundamentals.copy()
    has_fcf_ttm = "fcf_ttm" in df.columns
    provisional_count = 0

    for idx, row in df.iterrows():
        ticker_short = str(row.get("ticker", "")).split(".")[0]
        new_mcap = mcap_map.get(ticker_short)
        if not new_mcap:
            continue

        df.at[idx, "market_cap"] = new_mcap

        # Ricalcola FCF yield se abbiamo il valore assoluto in cache
        fcf_ttm = row.get("fcf_ttm") if has_fcf_ttm else None
        if fcf_ttm is not None and not (isinstance(fcf_ttm, float) and np.isnan(fcf_ttm)):
            new_yield = float(fcf_ttm) / new_mcap
            df.at[idx, "fcf_yield"]        = round(new_yield, 4)
            df.at[idx, "fcf_yield_passes"] = new_yield > 0.05
        else:
            # fcf_ttm non disponibile (cache da run con filtro errato):
            # whitelist provvisoria — F-Score + ICR senza verifica FCF.
            # Il prossimo refresh domenicale corregge definitivamente.
            df.at[idx, "fcf_yield_passes"] = True
            provisional_count += 1

        fscore     = int(row.get("f_score", 0) or 0)
        fcf_passes = bool(df.at[idx, "fcf_yield_passes"])
        icr_passes = bool(row.get("icr_passes", False))
        df.at[idx, "in_whitelist"] = (fscore >= 7 and fcf_passes and icr_passes)

    whitelist_count = int((df["in_whitelist"] == True).sum())
    logger.info(f"Patch completata — whitelist: {whitelist_count:,} ticker")
    if provisional_count > 0:
        logger.warning(
            f"{provisional_count:,} ticker con fcf_yield provvisorio (fcf_ttm assente). "
            "Il refresh domenicale ricalcolerà i valori corretti."
        )
    return df


# ============================================================
# STEP INDIVIDUALI DELLA PIPELINE
# ============================================================

def step_universe(api_key: str) -> pd.DataFrame:
    """
    Step 1 (domenicale): Costruisce l'universo raw dei Common Stock USA.
    Senza filtri quantitativi (quelli vengono applicati dopo i fondamentali).
    """
    logger.info("=== STEP: Universe Construction ===")
    universe = build_raw_universe(api_key)
    logger.info(f"Universe raw: {len(universe):,} Common Stock")
    return universe


def step_fundamentals(universe: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Step 2 (domenicale): Fetch e calcolo fondamentali per tutti i ticker.

    Con ~5.000 ticker e 10 call/ticker → ~50.000 call (dentro il budget giornaliero).
    """
    logger.info("=== STEP: Fundamentals Batch Processing ===")
    tickers = (universe["Code"] + ".US").tolist() if "Code" in universe.columns else []

    if not tickers:
        logger.warning("Nessun ticker nell'universo raw.")
        return pd.DataFrame()

    fundamentals = process_fundamentals_batch(tickers, api_key)
    if not fundamentals.empty:
        save_fundamentals_cache(fundamentals)
    return fundamentals


def step_prices_and_technicals(whitelist_tickers: list, api_key: str) -> tuple:
    """
    Step 3 (giornaliero): Aggiorna prezzi via bulk EOD e ricalcola tecnici.

    Args:
        whitelist_tickers: Lista ticker nella whitelist fondamentale
        api_key:           Chiave API EODHD

    Returns:
        Tuple (prices_df, technicals_df)
    """
    logger.info("=== STEP: Prices Update + Technicals ===")

    # Controlla se la cache prezzi esiste (primo avvio vs update incrementale)
    existing_prices = load_prices_cache()

    if existing_prices.empty:
        logger.info("Prima inizializzazione cache prezzi — scarico storico completo...")
        prices = initialize_prices_history(whitelist_tickers, api_key)
    else:
        logger.info("Aggiornamento incrementale prezzi via bulk EOD...")
        prices = update_prices_from_bulk(whitelist_tickers, api_key, EXCHANGES)

    if prices.empty:
        logger.warning("Nessun prezzo disponibile — tecnici non calcolabili.")
        return pd.DataFrame(), pd.DataFrame()

    # Calcolo indicatori tecnici
    technicals = compute_all_technicals(prices)
    if not technicals.empty:
        save_technicals_cache(technicals)

    return prices, technicals


def step_earnings(api_key: str) -> set:
    """Step 4 (giornaliero): Fetch earnings calendar per i prossimi 14 giorni."""
    logger.info("=== STEP: Earnings Calendar ===")
    exclusions = build_earnings_exclusion_set(api_key)
    save_earnings_cache(exclusions)
    return exclusions


def step_screener(
    fundamentals: pd.DataFrame,
    technicals: pd.DataFrame,
    earnings_exclusions: set,
) -> dict:
    """Step 5 (giornaliero): Combina i motori e salva i risultati."""
    logger.info("=== STEP: Screener Run ===")
    results = run_screener(fundamentals, technicals, earnings_exclusions)
    save_screener_results(results)
    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Entry point principale della pipeline notturna."""
    start_time = time.time()

    # === Carica credenziali da variabili d'ambiente ===
    api_key      = os.environ.get("EODHD_API_KEY", "")
    tg_token     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    github_run_url = os.environ.get("GITHUB_RUN_URL", "")

    if not api_key:
        logger.error("EODHD_API_KEY non configurata. Imposta il secret in GitHub Actions.")
        sys.exit(1)

    is_sunday = (datetime.today().weekday() == FULL_REFRESH_DAY)
    logger.info(f"Pipeline avviata | Domenica (full refresh): {is_sunday}")
    CACHE_DIR.mkdir(exist_ok=True)

    current_step = "init"
    try:
        # ============================================================
        # DOMENICA: refresh completo universo + fondamentali
        # ============================================================
        if is_sunday:
            current_step = "universe"
            universe = step_universe(api_key)

            current_step = "fundamentals"
            fundamentals = step_fundamentals(universe, api_key)
        else:
            # Giorni feriali: usa fondamentali cachati
            fundamentals = load_fundamentals_cache()
            if fundamentals.empty:
                logger.warning("Cache fondamentali non trovata — forzo refresh fondamentali...")
                current_step = "universe"
                universe = step_universe(api_key)
                current_step = "fundamentals"
                fundamentals = step_fundamentals(universe, api_key)

        # Whitelist tickers (per ottimizzare il fetch prezzi)
        whitelist_tickers = []
        if not fundamentals.empty and "ticker" in fundamentals.columns:
            if "in_whitelist" in fundamentals.columns:
                whitelist_tickers = fundamentals[
                    fundamentals["in_whitelist"] == True
                ]["ticker"].tolist()

            # Se la whitelist è vuota prova a recuperare senza spendere crediti API.
            # Cascata di costo crescente:
            #   1. Raw cache disponibile → ricalcolo locale (0 chiamate API)
            #   2. Raw cache + bulk EOD → ricalcolo con market_cap fresco (3 chiamate)
            #   3. Nessuna raw cache → patch solo market_cap da bulk EOD (3 chiamate)
            # NON viene mai forzato un re-fetch completo (~66K chiamate) automaticamente.
            if len(whitelist_tickers) == 0 and not fundamentals.empty:
                current_step = "recompute_or_patch"

                if raw_cache_exists():
                    # Ottieni market_cap fresco da bulk EOD (3 chiamate)
                    logger.info("Raw cache trovata — fetch market_cap da bulk EOD (3 chiamate)...")
                    mcap_map = _fetch_mcap_from_bulk_eod(api_key)
                    # Ricalcola tutto dalla raw cache (zero costo computation)
                    fundamentals = recompute_from_raw_cache(market_cap_override=mcap_map or None)
                    if not fundamentals.empty:
                        save_fundamentals_cache(fundamentals)
                else:
                    # Nessuna raw cache: patch solo market_cap con bulk EOD (3 chiamate)
                    logger.warning(
                        "Raw cache assente — patch market_cap da bulk EOD (3 chiamate). "
                        "FCF yield provvisorio fino al prossimo refresh domenicale."
                    )
                    fundamentals = patch_market_cap_from_bulk_eod(fundamentals, api_key)
                    if not fundamentals.empty:
                        save_fundamentals_cache(fundamentals)

                if "in_whitelist" in fundamentals.columns:
                    whitelist_tickers = fundamentals[
                        fundamentals["in_whitelist"] == True
                    ]["ticker"].tolist()

        logger.info(f"Whitelist tickers per price fetch: {len(whitelist_tickers):,}")

        # ============================================================
        # OGNI SERA: update prezzi + tecnici + earnings + screener
        # ============================================================
        current_step = "prices_technicals"
        _, technicals = step_prices_and_technicals(whitelist_tickers, api_key)

        current_step = "earnings"
        earnings_exclusions = step_earnings(api_key)

        current_step = "screener"
        results = step_screener(fundamentals, technicals, earnings_exclusions)

        # ============================================================
        # NOTIFICA TELEGRAM — SUCCESSO
        # ============================================================
        elapsed = time.time() - start_time
        stats   = results.get("stats", {})
        logger.info(f"Pipeline completata in {elapsed:.1f}s")

        if tg_token and tg_chat_id:
            notify_success(tg_token, tg_chat_id, stats, elapsed)
        else:
            logger.info("Telegram non configurato — skip notifica.")

        sys.exit(0)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(f"Pipeline FALLITA allo step '{current_step}': {e}")

        if tg_token and tg_chat_id:
            notify_error(tg_token, tg_chat_id, current_step, str(e), github_run_url)

        sys.exit(1)


if __name__ == "__main__":
    main()
