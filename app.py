"""
app.py — Fallen Angels Screener | Kriterion Quant

Dashboard Streamlit a 3 tab per il framework quantamentale Fallen Angels.

Tab 1 — Daily Screener:
    Whitelist fondamentale completa + candidati operativi del giorno
    (Z-Score <= -2.5, Volume 1.5x, nessun earnings imminente)

Tab 2 — Ticker Detail:
    Analisi approfondita per ticker selezionato dalla whitelist.
    Grafici: Price+SMA, Z-Score, F-Score breakdown, FCF Yield, ICR.

Tab 3 — Phase 2 Monitor:
    Inserisci un ticker in posizione aperta (Fase 2 - Covered Call loop).
    Monitora il degradamento quantamentale nel tempo e gli alert stop loss.

Deploy:
    streamlit run app.py
    Secrets necessari: EODHD_API_KEY (solo per Tab 3 live fetch)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURAZIONE PAGINA — deve essere il PRIMO comando Streamlit
# ============================================================
st.set_page_config(
    page_title="Fallen Angels Screener | Kriterion Quant",
    page_icon="🪽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import moduli interni
from pipeline.screener import load_screener_results, load_screener_metadata
from pipeline.fundamentals import compute_rolling_fscore_history
from pipeline.technicals import (
    fetch_ohlcv,
    get_zscore_history,
    ZSCORE_TRIGGER,
    SMA200_STOP_DAYS,
)
from src.charts import (
    build_price_sma_chart,
    build_zscore_chart,
    build_fscore_breakdown,
    build_fcf_yield_trend,
    build_icr_trend,
    build_fscore_history,
    build_fcf_history,
    build_icr_history,
    build_sma200_monitor,
)

# ============================================================
# API KEY (usata solo per Tab 3 — live fetch fondamentali)
# ============================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except Exception:
    EODHD_API_KEY = ""

# ============================================================
# CSS CUSTOM — ottimizza l'aspetto delle tabelle e dei metric
# ============================================================
st.markdown("""
<style>
    /* Header principale */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2196F3;
        margin-bottom: 0;
    }
    /* Badge candidato */
    .badge-candidate {
        background-color: #F44336;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    /* Alert box */
    .alert-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    /* Metriche più grandi */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_screener_cached():
    """Carica i risultati dello screener dalla cache Parquet (TTL 1h)."""
    return load_screener_results(), load_screener_metadata()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_history(ticker: str, api_key: str):
    """Carica lo storico fondamentale per un ticker (Tab 3). Cachato 1h."""
    if not api_key:
        return pd.DataFrame()
    return compute_rolling_fscore_history(ticker, api_key, n_quarters=8)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker_prices(ticker: str, api_key: str):
    """Carica i prezzi storici per un ticker (ultimi 18 mesi). Cachato 1h."""
    if not api_key:
        return pd.DataFrame()
    from datetime import timedelta
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=540)).strftime("%Y-%m-%d")
    return fetch_ohlcv(ticker, start, end, api_key)


def format_pct(val, decimals: int = 2) -> str:
    """Formatta un valore float come percentuale."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:.{decimals}f}%"


def format_float(val, decimals: int = 2, suffix: str = "") -> str:
    """Formatta un valore float con suffisso opzionale."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if val == float("inf"):
        return "∞"
    return f"{val:.{decimals}f}{suffix}"


def colorize_fscore(val) -> str:
    """Restituisce colore CSS per il valore F-Score."""
    if val is None or pd.isna(val):
        return "color: #9E9E9E"
    if val >= 7:
        return "color: #4CAF50; font-weight: bold"
    elif val >= 5:
        return "color: #FFC107"
    else:
        return "color: #F44336"


def style_screener_table(df: pd.DataFrame):
    """Applica stile professionale alla tabella screener."""

    def color_zscore(val):
        if pd.isna(val):
            return ""
        return "color: #F44336; font-weight:bold" if val <= ZSCORE_TRIGGER else ""

    def color_fscore(val):
        if pd.isna(val):
            return ""
        if val >= 7:
            return "color: #4CAF50; font-weight:bold"
        elif val >= 5:
            return "color: #FFC107"
        return "color: #F44336"

    def color_bool(val):
        if val is True:
            return "color: #4CAF50"
        elif val is False:
            return "color: #9E9E9E"
        return ""

    styler = df.style
    if "z_score" in df.columns:
        styler = styler.map(color_zscore, subset=["z_score"])
    if "f_score" in df.columns:
        styler = styler.map(color_fscore, subset=["f_score"])
    if "icr_passes" in df.columns:
        styler = styler.map(color_bool, subset=["icr_passes"])

    return styler.set_properties(**{
        "font-size": "13px",
        "text-align": "right",
    }).set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#2A2A3E"),
            ("color", "#E0E0E0"),
            ("font-weight", "bold"),
            ("padding", "8px 12px"),
            ("border-bottom", "2px solid #4444AA"),
        ]},
        {"selector": "tbody td", "props": [
            ("padding", "6px 12px"),
            ("border-bottom", "1px solid #333355"),
        ]},
        {"selector": "tbody tr:hover", "props": [
            ("background-color", "rgba(33,150,243,0.08)"),
        ]},
    ])


# ============================================================
# HEADER GLOBALE
# ============================================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🪽")
with col_title:
    st.markdown('<p class="main-header">Fallen Angels Screener</p>', unsafe_allow_html=True)
    st.caption("Framework Quantamentale | Kriterion Quant — Piotroski F-Score · FCF Yield · ICR · Z-Score")

st.divider()

# ============================================================
# CARICAMENTO DATI SCREENER (dalla cache)
# ============================================================
with st.spinner("Caricamento dati screener..."):
    screener_df, metadata = load_screener_cached()

cache_available = not screener_df.empty

if not cache_available:
    st.warning(
        "⚠️ **Cache screener non disponibile.** "
        "La pipeline notturna non è ancora stata eseguita. "
        "Esegui `python scheduler.py` localmente per il primo popolamento, "
        "oppure attendi la prima esecuzione programmata da GitHub Actions.",
        icon="⚠️",
    )

# ============================================================
# SIDEBAR — FILTRI E INFO
# ============================================================
with st.sidebar:
    st.header("⚙️ Filtri & Opzioni")
    st.divider()

    # Filtro settore
    if cache_available and "gic_sector" in screener_df.columns:
        all_sectors   = sorted(screener_df["gic_sector"].dropna().unique().tolist())
        selected_sectors = st.multiselect(
            "Settori GICS",
            options=all_sectors,
            default=all_sectors,
            help="Filtra i risultati per settore GICS",
        )
    else:
        selected_sectors = []

    # Filtro F-Score minimo
    fscore_min = st.slider(
        "F-Score minimo",
        min_value=7, max_value=9, value=7,
        help="Soglia minima Piotroski F-Score (default: 7, range sistema: 7-9)",
    )

    # Filtro Z-Score (soglia operativa)
    zscore_threshold = st.slider(
        "Soglia Z-Score",
        min_value=-4.0, max_value=-0.5, value=float(ZSCORE_TRIGGER), step=0.1,
        format="%.1f",
        help=f"Soglia operativa Z-Score (default sistema: {ZSCORE_TRIGGER}σ). "
             "Abbassa per vedere più candidati in esplorazione.",
    )
    if zscore_threshold != ZSCORE_TRIGGER:
        st.caption(f"⚠️ Soglia modificata: {zscore_threshold}σ (sistema: {ZSCORE_TRIGGER}σ)")

    # Mostra solo candidati operativi
    show_only_candidates = st.toggle(
        "Solo candidati operativi",
        value=False,
        help="Mostra solo i ticker con trigger Z-Score attivo alla soglia selezionata",
    )

    st.divider()

    # Info ultima esecuzione
    if metadata:
        run_date = metadata.get("run_date", "")
        if run_date:
            try:
                run_dt   = datetime.fromisoformat(run_date)
                run_str  = run_dt.strftime("%d %b %Y — %H:%M")
            except Exception:
                run_str = run_date
            st.caption(f"🕐 Ultimo aggiornamento: {run_str}")
        st.caption(f"📊 Universo: {metadata.get('universe_size', '—'):,} ticker")
        st.caption(f"✅ Whitelist: {metadata.get('whitelist_size', '—'):,} titoli")
        st.caption(f"🎯 Candidati oggi: {metadata.get('candidates_count', '—')}")
    else:
        st.caption("📡 Dati: EODHD")

    st.divider()
    st.caption("ℹ️ La cache si aggiorna ogni sera (lun-ven) via GitHub Actions.")
    st.caption("🔄 Fondamentali: refresh domenicale completo.")


# ============================================================
# 3 TAB PRINCIPALI
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📋 Daily Screener",
    "🔍 Ticker Detail",
    "🛡️ Phase 2 Monitor",
])


# ============================================================
# TAB 1 — DAILY SCREENER
# ============================================================
with tab1:
    st.subheader("📋 Daily Screener — Whitelist & Candidati Operativi")
    st.markdown("""
    La **whitelist fondamentale** include tutti i titoli che superano simultaneamente
    i tre filtri del Motore Fondamentale: Piotroski F-Score ≥ 7, FCF Yield > 5%, ICR
    superiore alla soglia settoriale. I **candidati operativi** (evidenziati in rosso)
    sono quelli con trigger tecnico attivo oggi: Z-Score ≤ -2.5 e volume ≥ 1.5× ADV20,
    senza earnings nei prossimi 14 giorni.
    """)

    if not cache_available:
        st.info("Nessun dato disponibile. Esegui la pipeline per popolare la cache.")
        st.stop()

    # Applica filtri sidebar
    df_display = screener_df.copy()
    if selected_sectors:
        df_display = df_display[df_display["gic_sector"].isin(selected_sectors)]
    if "f_score" in df_display.columns:
        df_display = df_display[df_display["f_score"] >= fscore_min]

    # Ricalcola is_candidate on-the-fly in base alla soglia Z-Score del sidebar
    # (sovrascrive il flag precompilato nella cache che usa sempre ZSCORE_TRIGGER fisso)
    if "z_score" in df_display.columns and "volume_ratio" in df_display.columns:
        df_display = df_display.copy()
        earnings_col = df_display["earnings_soon"] if "earnings_soon" in df_display.columns else False
        df_display["is_candidate"] = (
            (df_display["z_score"] <= zscore_threshold)
            & (df_display["volume_ratio"] >= 1.5)
            & (~df_display.get("earnings_soon", pd.Series(False, index=df_display.index)).fillna(False))
        )

    if show_only_candidates and "is_candidate" in df_display.columns:
        df_display = df_display[df_display["is_candidate"] == True]

    # === KPI SUMMARY ===
    candidates_today = df_display[df_display["is_candidate"] == True] if "is_candidate" in df_display.columns else pd.DataFrame()
    n_candidates = len(candidates_today)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Whitelist (filtrata)", f"{len(df_display):,}")
    k2.metric("Candidati oggi", n_candidates, delta=None)
    k3.metric(
        "F-Score medio",
        format_float(df_display["f_score"].mean(), 1) if "f_score" in df_display.columns else "—",
    )
    k4.metric(
        "FCF Yield medio",
        format_pct(df_display["fcf_yield"].mean(), 1) if "fcf_yield" in df_display.columns else "—",
    )
    k5.metric(
        "Esclusi earnings",
        metadata.get("earnings_exclusions", "—"),
    )

    st.divider()

    # === ALERT CANDIDATI OPERATIVI ===
    is_exploration = zscore_threshold != ZSCORE_TRIGGER
    if n_candidates > 0:
        threshold_note = f" (soglia esplorazione: {zscore_threshold}σ)" if is_exploration else ""
        st.markdown(
            f'<div class="alert-box">🎯 <strong>{n_candidates} candidato/i operativo/i</strong> '
            f'con trigger Z-Score ≤ {zscore_threshold}σ attivo oggi{threshold_note}. '
            f'{"⚠️ Soglia modificata — verifica con il valore operativo standard prima di procedere." if is_exploration else "Verifica la catena opzioni sul broker prima di procedere."}</div>',
            unsafe_allow_html=True,
        )
    else:
        if is_exploration:
            st.markdown(
                f'<div class="warning-box">📭 Nessun trigger attivo nemmeno alla soglia esplorazione {zscore_threshold}σ. '
                f'Prova ad abbassare ulteriormente lo slider.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warning-box">📭 Nessun trigger tecnico attivo oggi nella whitelist. '
                'Il sistema è in modalità attesa.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # === TABELLA RISULTATI ===
    # Colonne da mostrare nella tabella principale
    display_cols_map = {
        "ticker":        "Ticker",
        "name":          "Azienda",
        "gic_sector":    "Settore",
        "f_score":       "F-Score",
        "fcf_yield":     "FCF Yield",
        "icr":           "ICR",
        "z_score":       "Z-Score",
        "volume_ratio":  "Vol/ADV",
        "price":         "Prezzo",
        "pct_change_1d": "Δ 1g%",
        "earnings_soon": "Earnings⚠",
        "is_candidate":  "🎯 Trigger",
    }

    available_cols = [c for c in display_cols_map if c in df_display.columns]
    df_table = df_display[available_cols].rename(columns=display_cols_map).copy()

    # Formattazione colonne
    if "FCF Yield" in df_table.columns:
        df_table["FCF Yield"] = df_table["FCF Yield"].apply(lambda x: format_pct(x, 1))
    if "Prezzo" in df_table.columns:
        df_table["Prezzo"] = df_table["Prezzo"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    if "Δ 1g%" in df_table.columns:
        df_table["Δ 1g%"] = df_table["Δ 1g%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
    if "ICR" in df_table.columns:
        df_table["ICR"] = df_table["ICR"].apply(lambda x: format_float(x, 1, "x"))
    if "Vol/ADV" in df_table.columns:
        df_table["Vol/ADV"] = df_table["Vol/ADV"].apply(lambda x: format_float(x, 2, "x"))
    if "Z-Score" in df_table.columns:
        df_table["Z-Score"] = df_table["Z-Score"].apply(lambda x: format_float(x, 3))
    if "Earnings⚠" in df_table.columns:
        df_table["Earnings⚠"] = df_table["Earnings⚠"].map({True: "⚠️", False: "", None: ""})
    if "🎯 Trigger" in df_table.columns:
        df_table["🎯 Trigger"] = df_table["🎯 Trigger"].map({True: "🎯", False: "", None: ""})

    st.dataframe(
        df_table,
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    # === BREAKDOWN PER SETTORE ===
    if "gic_sector" in df_display.columns and len(df_display) > 0:
        with st.expander("📊 Breakdown per Settore GICS"):
            sector_stats = (
                df_display.groupby("gic_sector")
                .agg(
                    Titoli=("ticker", "count"),
                    F_Score_Medio=("f_score", "mean"),
                    FCF_Yield_Medio=("fcf_yield", "mean"),
                    Candidati=("is_candidate", "sum") if "is_candidate" in df_display.columns else ("ticker", "count"),
                )
                .reset_index()
                .rename(columns={"gic_sector": "Settore"})
                .sort_values("Titoli", ascending=False)
            )
            sector_stats["F_Score_Medio"] = sector_stats["F_Score_Medio"].apply(lambda x: f"{x:.1f}")
            sector_stats["FCF_Yield_Medio"] = sector_stats["FCF_Yield_Medio"].apply(lambda x: format_pct(x, 1))
            st.dataframe(sector_stats, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Metodologia — Motori Fondamentale e Tecnico"):
        st.markdown("""
        **Motore Fondamentale** (aggiornato settimanalmente):
        - **Piotroski F-Score TTM** ≥ 7/9: 9 componenti su profittabilità, leva/liquidità
          ed efficienza operativa calcolati su Trailing Twelve Months (ultimi 4 trimestri).
        - **FCF Yield** > 5%: Free Cash Flow / Market Cap. Per settori ciclici,
          media rolling su 8 trimestri (2 anni) per ridurre la volatilità ciclica.
        - **ICR** (EBIT / Interest Expense) TTM > soglia settoriale: da 2x (Utilities)
          a 5x (Technology). Settori Financials e Real Estate esclusi dall'universo.

        **Motore Tecnico** (aggiornato ogni sera):
        - **Z-Score** = (Prezzo − SMA50) / StdDev50 ≤ −2.5: evento statisticamente raro
          (< 1% dei giorni in distribuzione normale), segnale di panic selling.
        - **Volume Ratio** ≥ 1.5×: conferma della capitolazione (selling pressure genuino).
        - **Earnings filter**: esclusi ticker con earnings nei prossimi 14 giorni.

        *Dati: EODHD Historical Data API. Aggiornamento nightly via GitHub Actions.*
        """)


# ============================================================
# TAB 2 — TICKER DETAIL
# ============================================================
with tab2:
    st.subheader("🔍 Ticker Detail — Analisi Approfondita")
    st.markdown("""
    Seleziona un ticker dalla whitelist (o inseriscilo manualmente) per visualizzare
    l'analisi completa: andamento tecnico, trigger Z-Score, breakdown F-Score e
    trend dei fondamentali chiave.
    """)

    col_sel1, col_sel2 = st.columns([3, 1])

    with col_sel1:
        if cache_available and "ticker" in screener_df.columns:
            ticker_options = sorted(screener_df["ticker"].dropna().unique().tolist())
        else:
            ticker_options = []

        selected_ticker_detail = st.selectbox(
            "Seleziona ticker dalla whitelist",
            options=[""] + ticker_options,
            index=0,
            help="Seleziona un ticker dalla whitelist fondamentale",
            key="ticker_detail_select",
        )

    with col_sel2:
        manual_ticker = st.text_input(
            "Oppure digita manualmente",
            placeholder="es. AAPL.US",
            key="ticker_detail_manual",
        ).strip().upper()

    # Priorità: manuale > selectbox
    active_ticker = manual_ticker if manual_ticker else selected_ticker_detail

    if not active_ticker:
        st.info("👆 Seleziona o inserisci un ticker per visualizzare l'analisi.")
        st.stop()

    # Normalizza formato ticker
    if not active_ticker.endswith(".US"):
        active_ticker = active_ticker + ".US"

    st.divider()

    # Recupera dati fondamentali dalla cache screener
    ticker_fundamentals = {}
    if cache_available and "ticker" in screener_df.columns:
        row = screener_df[screener_df["ticker"] == active_ticker]
        if not row.empty:
            ticker_fundamentals = row.iloc[0].to_dict()

    # Carica prezzi storici
    if not EODHD_API_KEY:
        st.warning("⚠️ `EODHD_API_KEY` non configurata. I grafici di prezzo non sono disponibili. "
                   "Configura il secret in `.streamlit/secrets.toml` o Streamlit Cloud.")
        prices_df = pd.DataFrame()
    else:
        with st.spinner(f"Caricamento prezzi {active_ticker}..."):
            prices_df = load_ticker_prices(active_ticker, EODHD_API_KEY)

    # === HEADER TICKER ===
    ticker_name   = ticker_fundamentals.get("name", active_ticker)
    ticker_sector = ticker_fundamentals.get("gic_sector", "—")
    is_cand       = ticker_fundamentals.get("is_candidate", False)
    fscore_val    = ticker_fundamentals.get("f_score")
    zscore_val    = ticker_fundamentals.get("z_score")

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        candidate_badge = ' <span class="badge-candidate">🎯 TRIGGER ATTIVO</span>' if is_cand else ""
        st.markdown(
            f"### {active_ticker} — {ticker_name}{candidate_badge}",
            unsafe_allow_html=True,
        )
        st.caption(f"Settore: {ticker_sector}")
    with header_col2:
        if fscore_val is not None:
            score_color = "green" if fscore_val >= 7 else "orange" if fscore_val >= 5 else "red"
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:2.5rem;font-weight:bold;color:{score_color};'>"
                f"{int(fscore_val)}/9</span><br>"
                f"<span style='color:#9E9E9E;font-size:0.75rem;'>F-Score</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # === KPI TICKER ===
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Prezzo",      f"${ticker_fundamentals.get('price', '—'):.2f}" if ticker_fundamentals.get('price') else "—")
    kpi2.metric("Z-Score",     format_float(zscore_val, 3))
    kpi3.metric("Vol/ADV20",   format_float(ticker_fundamentals.get("volume_ratio"), 2, "x"))
    kpi4.metric("FCF Yield",   format_pct(ticker_fundamentals.get("fcf_yield"), 1))
    kpi5.metric("ICR",         format_float(ticker_fundamentals.get("icr"), 1, "x"))
    kpi6.metric("Market Cap",  f"${ticker_fundamentals.get('market_cap', 0) / 1e9:.1f}B" if ticker_fundamentals.get('market_cap') else "—")

    st.divider()

    # === GRAFICO PREZZO + SMA ===
    st.subheader("📈 Prezzo, SMA50 e SMA200")
    st.markdown("""
    Il grafico mostra il prezzo adjusted close con le medie mobili a 50 e 200 giorni.
    La **SMA50** (arancio tratteggiato) è usata per il calcolo dello Z-Score.
    La **SMA200** (grigio) è il riferimento per il time-stop della Fase 2.
    """)

    if not prices_df.empty:
        fig_price = build_price_sma_chart(prices_df.reset_index(), active_ticker)
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Prezzi non disponibili. Configura EODHD_API_KEY.")

    # === GRAFICO Z-SCORE ===
    st.subheader("📉 Z-Score (SMA50 | 50 periodi)")
    st.markdown(f"""
    Lo Z-Score normalizza il prezzo rispetto alla sua media mobile a 50 giorni in unità di
    deviazione standard: Z = (Prezzo − SMA50) / σ50. La soglia operativa è **{ZSCORE_TRIGGER}σ**
    (probabilità statistica < 1% in distribuzione normale). I punti in rosso indicano i giorni
    in cui il trigger era attivo.
    """)

    if not prices_df.empty:
        prices_reset = prices_df.reset_index()
        zscore_history = get_zscore_history(prices_reset)
        if not zscore_history.empty:
            fig_zscore = build_zscore_chart(zscore_history, active_ticker)
            st.plotly_chart(fig_zscore, use_container_width=True)
    else:
        st.info("Prezzi non disponibili.")

    st.divider()

    # === F-SCORE BREAKDOWN ===
    st.subheader("🧮 Breakdown Piotroski F-Score")
    st.markdown("""
    I 9 componenti del F-Score, raggruppati per categoria:
    **A** (Profittabilità: ROA, CFO, ΔROA, Accruals),
    **B** (Leva & Liquidità: ΔLEV, ΔLIQ, no diluizione),
    **C** (Efficienza: ΔMAR, ΔTURN).
    Il verde indica il componente attivo (1 punto), il rosso inattivo (0 punti).
    """)

    fscore_keys = ["F_ROA", "F_CFO", "F_DROA", "F_ACC", "F_DLEV", "F_DLIQ", "F_EQ", "F_DMAR", "F_DTURN", "f_score"]
    fscore_dict = {k: ticker_fundamentals.get(k) for k in fscore_keys if k in ticker_fundamentals}

    if fscore_dict:
        fig_fscore = build_fscore_breakdown(fscore_dict, active_ticker)
        st.plotly_chart(fig_fscore, use_container_width=True)
    else:
        st.info("F-Score non disponibile per questo ticker.")

    # === FCF YIELD + ICR ===
    col_fcf, col_icr = st.columns(2)

    with col_fcf:
        st.subheader("💰 FCF Yield")
        st.markdown(
            "Free Cash Flow / Market Cap TTM. "
            "La soglia operativa è **5%** (linea verde tratteggiata). "
            "Per settori ciclici, calcolato come media su 8 trimestri."
        )
        # Crea DataFrame sintetico da dati screener (punto corrente)
        fcf_val = ticker_fundamentals.get("fcf_yield")
        if fcf_val is not None:
            fcf_df = pd.DataFrame({
                "date":      [pd.Timestamp.today()],
                "fcf_yield": [fcf_val],
            })
            fig_fcf = build_fcf_yield_trend(fcf_df, active_ticker, ticker_sector)
            st.plotly_chart(fig_fcf, use_container_width=True)
            st.caption("💡 Il grafico multi-trimestrale è disponibile nella Tab Phase 2 Monitor.")
        else:
            st.info("FCF Yield non disponibile.")

    with col_icr:
        st.subheader("🏦 Interest Coverage Ratio")
        icr_val = ticker_fundamentals.get("icr")
        icr_thr = ticker_fundamentals.get("icr_threshold", 5.0)
        st.markdown(
            f"EBIT / Interest Expense TTM. "
            f"Soglia settoriale per **{ticker_sector}**: **{icr_thr}x** (linea arancio). "
            f"ICR infinito = azienda senza debito finanziario."
        )
        if icr_val is not None:
            icr_df = pd.DataFrame({
                "date":          [pd.Timestamp.today()],
                "icr":           [min(icr_val, icr_thr * 3) if icr_val != float("inf") else icr_thr * 3],
                "icr_threshold": [icr_thr],
            })
            fig_icr = build_icr_trend(icr_df, active_ticker, ticker_sector, icr_thr)
            st.plotly_chart(fig_icr, use_container_width=True)
        else:
            st.info("ICR non disponibile.")

    with st.expander("ℹ️ Note tecniche — Calcoli fondamentali"):
        st.markdown(f"""
        - **F-Score**: calcolato su base TTM (ultimi 4 trimestri per flussi,
          trimestre più recente per stock). Delta YoY confronta TTM corrente vs TTM 12 mesi fa.
        - **FCF**: CFO − |CapEx|. Dati EODHD quarterly, sezione Cash Flow Statement.
        - **ICR**: usa EBIT (non EBITDA) per non escludere ammortamenti reali.
          InterestExpense: valore assoluto per uniformità tra ticker.
        - **Z-Score**: finestra 50 giorni di borsa (≈ 2.5 mesi calendario).
          P(Z ≤ {ZSCORE_TRIGGER}) ≈ 0.62% in distribuzione normale.
        - Tutti i prezzi: adjusted close (split e dividendi inclusi). Fonte: EODHD EOD API.
        """)


# ============================================================
# TAB 3 — PHASE 2 MONITOR
# ============================================================
with tab3:
    st.subheader("🛡️ Phase 2 Monitor — Monitoring Posizioni in Covered Call Loop")
    st.markdown("""
    Inserisci il ticker di un'azienda su cui stai applicando la **Fase 2** del framework
    (Covered Call loop post-assegnazione). Il monitor analizza il deterioramento dei parametri
    quantamentali nel tempo e segnala l'eventuale attivazione degli stop loss:

    - 🔴 **Stop Loss Fondamentale**: F-Score < 5 oppure FCF negativo per 2 trimestri consecutivi
    - 🟡 **Time-Stop**: prezzo sotto SMA200 per {SMA200_STOP_DAYS}+ giorni consecutivi
    """.format(SMA200_STOP_DAYS=SMA200_STOP_DAYS))

    st.divider()

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        monitor_ticker = st.text_input(
            "Ticker in Fase 2",
            placeholder="es. AAPL.US oppure AAPL",
            key="phase2_ticker",
            help="Inserisci il ticker dell'azienda in posizione aperta (Covered Call loop).",
        ).strip().upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_monitor = st.button("🔍 Analizza", type="primary", use_container_width=True)

    if not monitor_ticker and not run_monitor:
        st.info("👆 Inserisci un ticker e clicca **Analizza** per avviare il monitoraggio.")
        st.stop()

    if monitor_ticker:
        # Normalizza formato
        if not monitor_ticker.endswith(".US"):
            monitor_ticker = monitor_ticker + ".US"

        if not EODHD_API_KEY:
            st.error("❌ `EODHD_API_KEY` non configurata. Impossibile caricare i dati fondamentali.")
            st.stop()

        # Carica dati
        with st.spinner(f"Caricamento dati fondamentali storici per {monitor_ticker}..."):
            history_df = load_ticker_history(monitor_ticker, EODHD_API_KEY)

        with st.spinner(f"Caricamento prezzi storici per {monitor_ticker}..."):
            prices_mon = load_ticker_prices(monitor_ticker, EODHD_API_KEY)

        if history_df.empty:
            st.warning(
                f"⚠️ Dati fondamentali insufficienti per **{monitor_ticker}**. "
                "Potrebbero mancare trimestri storici nell'API EODHD, "
                "oppure il ticker non è nella copertura del Fundamental Data Feed."
            )
            st.stop()

        # === HEADER MONITOR ===
        # Dati più recenti
        latest = history_df.iloc[-1] if not history_df.empty else {}
        current_fscore = latest.get("f_score")
        current_fcf    = latest.get("fcf_yield")
        current_icr    = latest.get("icr")

        # Calcola giorni sotto SMA200 dai prezzi
        days_below_sma200 = 0
        if not prices_mon.empty:
            prices_reset_mon = prices_mon.reset_index() if prices_mon.index.name == "date" else prices_mon
            price_col = "adjusted_close" if "adjusted_close" in prices_reset_mon.columns else "close"
            if price_col in prices_reset_mon.columns:
                close_vals = prices_reset_mon[price_col].astype(float)
                sma200_vals = close_vals.rolling(200).mean()
                below = (close_vals < sma200_vals).values[::-1]
                for b in below:
                    if b:
                        days_below_sma200 += 1
                    else:
                        break

        # === ALERT STOP LOSS ===
        fundamental_stop = (
            (current_fscore is not None and current_fscore < 5)
            or (current_fcf is not None and current_fcf < 0)
        )
        time_stop = days_below_sma200 >= SMA200_STOP_DAYS

        if fundamental_stop:
            st.markdown(
                '<div class="alert-box">🔴 <strong>STOP LOSS FONDAMENTALE ATTIVO</strong> — '
                'F-Score < 5 oppure FCF negativo. La tesi di investimento è invalidata. '
                'Valuta la liquidazione della posizione indipendentemente dal prezzo corrente.</div>',
                unsafe_allow_html=True,
            )
        elif time_stop:
            st.markdown(
                f'<div class="alert-box">🟡 <strong>TIME-STOP ATTIVO</strong> — '
                f'{days_below_sma200} giorni consecutivi sotto SMA200 (soglia: {SMA200_STOP_DAYS}). '
                f'Il titolo è in dead money. Valuta la chiusura per liberare capitale.</div>',
                unsafe_allow_html=True,
            )
        elif days_below_sma200 > 60:
            st.markdown(
                f'<div class="warning-box">⚠️ <strong>Attenzione</strong> — '
                f'{days_below_sma200} giorni sotto SMA200. Time-stop a {SMA200_STOP_DAYS} giorni.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success(f"✅ **{monitor_ticker}** — Nessun alert stop loss attivo. Posizione monitorata.")

        st.divider()

        # === KPI STATO ATTUALE ===
        st.markdown(f"#### Stato Attuale — {monitor_ticker}")
        m1, m2, m3, m4 = st.columns(4)

        fscore_color = "normal" if (current_fscore and current_fscore >= 7) else "inverse"
        m1.metric(
            "F-Score (ultimo trimestre)",
            f"{int(current_fscore)}/9" if current_fscore is not None else "—",
            delta="✅ OK" if (current_fscore and current_fscore >= 7) else ("⚠️ Attenzione" if (current_fscore and current_fscore >= 5) else "🔴 CRITICO"),
        )
        m2.metric(
            "FCF Yield",
            format_pct(current_fcf, 1),
            delta="✅ Positivo" if (current_fcf and current_fcf > 0) else "🔴 NEGATIVO",
        )
        m3.metric(
            "ICR",
            format_float(current_icr, 1, "x"),
            delta=f"Soglia: {latest.get('icr_threshold', '—')}x",
        )
        m4.metric(
            "Giorni sotto SMA200",
            str(days_below_sma200),
            delta=f"Soglia stop: {SMA200_STOP_DAYS}gg",
            delta_color="inverse" if time_stop else "off",
        )

        st.divider()

        # === GRAFICI STORICI ===

        # F-Score trend
        st.subheader("📊 Trend F-Score (ultimi 8 trimestri)")
        st.markdown(
            "Il grafico mostra l'evoluzione del Piotroski F-Score nel tempo. "
            "Un calo sostenuto sotto la soglia **5** è un segnale di deterioramento strutturale "
            "e attiva il **stop loss fondamentale**. La zona rossa indica area di pericolo."
        )
        fig_fsh = build_fscore_history(history_df, monitor_ticker)
        st.plotly_chart(fig_fsh, use_container_width=True)

        # FCF e ICR affiancati
        col_fcf_mon, col_icr_mon = st.columns(2)

        with col_fcf_mon:
            st.subheader("💰 FCF Yield Trend")
            st.markdown(
                "Trend del FCF Yield trimestrale. Se il FCF diventa **negativo** "
                "(barre rosse sotto zero) per 2 trimestri consecutivi, la tesi di "
                "investimento è invalidata indipendentemente dal prezzo."
            )
            fig_fcfh = build_fcf_history(history_df, monitor_ticker)
            st.plotly_chart(fig_fcfh, use_container_width=True)

        with col_icr_mon:
            st.subheader("🏦 ICR Trend")
            icr_threshold = float(history_df["icr_threshold"].iloc[-1]) if "icr_threshold" in history_df.columns else 5.0
            st.markdown(
                f"Trend dell'Interest Coverage Ratio. "
                f"La soglia per il settore è **{icr_threshold}x** (linea arancio). "
                "Un ICR in calo persistente segnala fragilità finanziaria crescente."
            )
            fig_icrh = build_icr_history(history_df, monitor_ticker)
            st.plotly_chart(fig_icrh, use_container_width=True)

        # SMA200 Monitor
        st.subheader(f"📉 Prezzo vs SMA200 — Time-Stop Monitor")
        st.markdown(f"""
        La **SMA200** è il discriminatore universale tra bull market e bear market su singolo titolo.
        Un prezzo che rimane sotto la SMA200 per **{SMA200_STOP_DAYS} giorni consecutivi** (≈ 4.5 mesi)
        attiva il **time-stop**: il capitale è in dead money e il costo opportunità giustifica l'uscita
        anche in perdita.
        """)

        if not prices_mon.empty:
            prices_reset_for_chart = prices_mon.reset_index() if prices_mon.index.name == "date" else prices_mon
            fig_sma200 = build_sma200_monitor(prices_reset_for_chart, monitor_ticker, days_below_sma200)
            st.plotly_chart(fig_sma200, use_container_width=True)
        else:
            st.info("Prezzi non disponibili per il grafico SMA200.")

        with st.expander("ℹ️ Metodologia Stop Loss — Framework Fallen Angels"):
            st.markdown(f"""
            Il framework adotta due meccanismi di uscita **indipendenti** per la Fase 2:

            **Stop Loss Fondamentale** (invalidazione della tesi):
            - F-Score < 5 al prossimo report trimestrale
            - FCF negativo per 2 trimestri consecutivi (l'azienda brucia cassa)
            - In entrambi i casi: liquidazione a mercato sull'open della sessione successiva.

            **Time-Stop** (costo opportunità):
            - Prezzo sotto SMA200 per **{SMA200_STOP_DAYS} giorni di borsa** consecutivi (≈ 4.5 mesi)
            - Non indica insolvenza ma dead money: il capitale non può essere riallocato
            - Si applica anche se l'azienda è fondamentalmente solida

            *Nota: il sistema non usa stop loss percentuali fissi (es. -20%) per non andare
            contro la logica mean reversion su cui si basa la strategia.*
            """)
