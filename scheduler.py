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
        if not fundamentals.empty and "ticker" in fundamentals.columns:
            whitelist_tickers = fundamentals[
                fundamentals["in_whitelist"] == True
            ]["ticker"].tolist()
        else:
            whitelist_tickers = fundamentals["ticker"].tolist() if not fundamentals.empty else []

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
