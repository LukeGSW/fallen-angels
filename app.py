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


import plotly.graph_objects as go


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    prices_df: pd.DataFrame,
    entry_z: float = -2.0,
    exit_z: float  =  2.0,
    sma_stop: int  = 120,
    z_window: int  =  50,
) -> pd.DataFrame:
    """
    Simula long equity con entry a Z-Score ≤ entry_z (primo crossover dal basso)
    ed exit a Z-Score ≥ exit_z oppure dopo sma_stop gg consecutivi sotto SMA200.
    """
    df = prices_df.copy().sort_values("date").reset_index(drop=True)
    price_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    close  = df[price_col].astype(float)
    dates  = pd.to_datetime(df["date"])
    sma_f  = close.rolling(z_window).mean()
    std_f  = close.rolling(z_window).std()
    zscore = (close - sma_f) / std_f
    sma200 = close.rolling(200).mean()

    trades = []
    in_pos = False
    entry_price = entry_date = entry_z_val = None
    days_below = 0
    can_enter  = True   # richiede recupero sopra entry_z prima di rientrare

    for i in range(1, len(df)):
        z     = zscore.iloc[i]
        z_prv = zscore.iloc[i - 1]
        px    = close.iloc[i]
        dt    = dates.iloc[i]
        s200  = sma200.iloc[i]

        if pd.isna(z) or pd.isna(s200):
            continue

        if not in_pos:
            if z > entry_z:
                can_enter = True
            if can_enter and z <= entry_z and z_prv > entry_z:
                in_pos = True
                entry_price, entry_date, entry_z_val = px, dt, z
                days_below = 0 if px >= s200 else 1
                can_enter  = False
        else:
            days_below = days_below + 1 if px < s200 else 0
            reason = None
            if z >= exit_z:
                reason = f"Z-Score ≥ +{exit_z:.1f} (Take Profit)"
            elif days_below >= sma_stop:
                reason = f"SMA200 {sma_stop}gg (Time Stop)"

            if reason:
                pnl_pct = (px - entry_price) / entry_price * 100
                trades.append({
                    "entry_date":  entry_date.strftime("%Y-%m-%d"),
                    "exit_date":   dt.strftime("%Y-%m-%d"),
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(px, 2),
                    "entry_z":     round(entry_z_val, 2),
                    "exit_z":      round(z, 2),
                    "days_held":   (dt - entry_date).days,
                    "pnl_pct":     round(pnl_pct, 2),
                    "exit_reason": reason,
                    "win":         pnl_pct > 0,
                })
                in_pos = False
                days_below = 0

    return pd.DataFrame(trades)


def bt_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    wins   = trades[trades["win"]]
    losses = trades[~trades["win"]]
    equity = trades["pnl_pct"].cumsum()
    max_dd = (equity - equity.cummax()).min()
    return {
        "n_trades":    len(trades),
        "win_rate":    len(wins) / len(trades) * 100,
        "avg_pnl":     trades["pnl_pct"].mean(),
        "avg_win":     wins["pnl_pct"].mean()   if not wins.empty   else 0.0,
        "avg_loss":    losses["pnl_pct"].mean() if not losses.empty else 0.0,
        "best":        trades["pnl_pct"].max(),
        "worst":       trades["pnl_pct"].min(),
        "avg_days":    trades["days_held"].mean(),
        "total_pnl":   trades["pnl_pct"].sum(),
        "max_dd":      max_dd,
        "n_tp":        trades["exit_reason"].str.contains("Take Profit").sum(),
        "n_stop":      trades["exit_reason"].str.contains("Stop").sum(),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_bt_prices(ticker: str, api_key: str) -> pd.DataFrame:
    """Carica prezzi per backtest: prima dalla cache locale, poi da EODHD."""
    from pipeline.technicals import load_prices_cache
    all_px = load_prices_cache()
    if not all_px.empty and "ticker" in all_px.columns:
        tkr_px = all_px[all_px["ticker"] == ticker].copy()
        if len(tkr_px) > 250:
            return tkr_px
    if not api_key:
        return pd.DataFrame()
    from datetime import timedelta
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=1825)).strftime("%Y-%m-%d")
    px = fetch_ohlcv(ticker, start, end, api_key)
    if px.empty:
        return pd.DataFrame()
    px = px.reset_index()
    px["date"]   = pd.to_datetime(px["date"]).dt.strftime("%Y-%m-%d")
    px["ticker"] = ticker
    return px


def _bt_dark(title="", h=380):
    return dict(
        template="plotly_dark",
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E0E0E0", size=11),
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=50, r=20, t=40, b=40),
        height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
    )


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
        min_value=6, max_value=9, value=7,
        help="Soglia minima Piotroski F-Score (operativo: 7). "
             "Abbassare a 6 espande la pool ma riduce la qualità.",
    )
    if fscore_min < 7:
        st.caption(f"⚠️ F-Score {fscore_min} — soglia esplorazione (operativo: 7)")

    # Filtro FCF Yield minimo
    fcf_yield_min = st.slider(
        "FCF Yield minimo (%)",
        min_value=1, max_value=10, value=5, step=1,
        help="Soglia minima FCF Yield TTM (operativa: 5%). "
             "Abbassare a 3% aggiunge business eccellenti con alta market cap.",
    )
    if fcf_yield_min != 5:
        st.caption(f"⚠️ FCF Yield {fcf_yield_min}% — soglia esplorazione (operativo: 5%)")

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

    # Filtro Volume Ratio minimo
    volume_ratio_min = st.slider(
        "Volume Ratio minimo (x ADV20)",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        format="%.1f",
        help="Conferma di capitolazione (operativo: 1.5x ADV20). "
             "Abbassare a 1.0 rimuove il requisito di volume anomalo.",
    )
    if volume_ratio_min != 1.5:
        st.caption(f"⚠️ Volume {volume_ratio_min}x — soglia esplorazione (operativo: 1.5x)")

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
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Daily Screener",
    "🔍 Ticker Detail",
    "🛡️ Phase 2 Monitor",
    "📊 Backtest",
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
    # Nota: screener_df contiene TUTTI i ticker della whitelist fondamentale (in_whitelist=True,
    # quindi F-Score≥7 e FCF>5%). I filtri sidebar espandono verso il basso rispetto all'operativo.
    df_display = screener_df.copy()

    # Aggiunge ticker con F-Score o FCF Yield inferiori al default se slider abbassati
    if fscore_min < 7 or fcf_yield_min < 5:
        # Modalità esplorazione: carica TUTTI i fondamentali + TUTTA la cache tecnici
        # (non solo i 372 whitelist ticker) per trovare Z-Score anche su ticker extra
        from pipeline.fundamentals import load_fundamentals_cache
        from pipeline.technicals import load_technicals_cache
        all_fundamentals = load_fundamentals_cache()
        all_technicals   = load_technicals_cache()
        if not all_fundamentals.empty:
            base = all_fundamentals.copy()
            # Merge con la cache tecnici completa (include ticker oltre la whitelist stretta)
            if not all_technicals.empty:
                base = base.merge(all_technicals, on="ticker", how="left")
            # Preserva colonne extra dallo screener (earnings_soon, ecc.)
            screener_extra = [c for c in screener_df.columns
                              if c not in base.columns and c != "ticker"]
            if screener_extra:
                base = base.merge(
                    screener_df[["ticker"] + screener_extra], on="ticker", how="left"
                )
            mask = (
                (~base["gic_sector"].isin({"Financials", "Financial Services", "Real Estate"}))
                & (base["f_score"] >= fscore_min)
                & (base["fcf_yield"].notna())
                & (base["fcf_yield"] > fcf_yield_min / 100)
                & (base["fcf_yield"] <= 1.0)
                & (base["icr_passes"] == True)
            )
            df_display = base[mask].copy()
    else:
        # Soglie operative: usa direttamente la whitelist precompilata
        if selected_sectors:
            df_display = df_display[df_display["gic_sector"].isin(selected_sectors)]

    if selected_sectors and (fscore_min < 7 or fcf_yield_min < 5):
        df_display = df_display[df_display["gic_sector"].isin(selected_sectors)]

    # Ricalcola is_candidate on-the-fly in base alle soglie sidebar
    if "z_score" in df_display.columns and "volume_ratio" in df_display.columns:
        df_display = df_display.copy()
        not_earnings = (
            ~df_display["earnings_soon"].astype(bool)
            if "earnings_soon" in df_display.columns
            else pd.Series(True, index=df_display.index)
        )
        df_display["is_candidate"] = (
            (df_display["z_score"] <= zscore_threshold)
            & (df_display["volume_ratio"] >= volume_ratio_min)
            & not_earnings
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
    is_exploration = (zscore_threshold != ZSCORE_TRIGGER) or (fscore_min < 7) or (fcf_yield_min < 5) or (volume_ratio_min != 1.5)
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


# ============================================================
# TAB 4 — BACKTEST
# ============================================================
with tab4:
    st.subheader("📊 Backtest — Long Equity Z-Score Mean Reversion")
    st.markdown("""
    Simula la strategia su prezzi storici: **entry** al primo crossover Z-Score ≤ soglia,
    **exit** quando Z-Score ≥ +2.0 (take profit) oppure dopo N giorni consecutivi sotto
    SMA200 (time-stop). Il backtest usa la cache prezzi locale; per ticker non ancora
    tracciati richiede EODHD_API_KEY.
    """)
    st.warning(
        "⚠️ **Caveat metodologico**: il backtest simula solo i filtri tecnici (Z-Score, SMA200). "
        "Il filtro fondamentale (F-Score ≥ 7, FCF > 5%, ICR) si applica al *momento attuale* "
        "e non è verificabile storicamente — si assume che la qualità fosse comparabile nel passato "
        "(selection bias noto, ineliminabile senza serie storiche fondamentali).",
        icon="⚠️",
    )
    st.divider()

    # ── Ticker input ─────────────────────────────────────────────────────────
    col_bt1, col_bt2 = st.columns([3, 1])
    with col_bt1:
        bt_options = ([""] + sorted(screener_df["ticker"].dropna().unique().tolist())
                      if cache_available and "ticker" in screener_df.columns else [""])
        bt_sel = st.selectbox("Seleziona ticker dalla whitelist",
                              bt_options, key="bt_sel")
    with col_bt2:
        bt_man = st.text_input("Oppure digita manualmente",
                               placeholder="es. AAPL.US", key="bt_man").strip().upper()

    bt_ticker_raw = bt_man if bt_man else bt_sel
    bt_ticker_bt  = (bt_ticker_raw + ".US"
                     if bt_ticker_raw and not bt_ticker_raw.endswith(".US")
                     else bt_ticker_raw)

    # ── Parametri ─────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Parametri")
    p1, p2, p3, p4 = st.columns(4)
    bt_entry_z  = p1.slider("Z-Score Entry ≤", -4.0, -1.0, -2.0, 0.1, format="%.1f", key="bt_ez")
    bt_exit_z   = p2.slider("Z-Score Exit ≥",   0.5,  4.0,  2.0, 0.1, format="%.1f", key="bt_xz")
    bt_sma_stop = p3.slider("SMA200 Stop (gg)", 60, 200, 120, 10, key="bt_sma")
    bt_z_win    = p4.slider("Finestra Z-Score (gg)", 20, 100, 50, 5, key="bt_zw")

    run_bt_btn = st.button("🚀 Esegui Backtest", type="primary",
                           disabled=(not bt_ticker_bt), key="bt_run")

    if not bt_ticker_bt:
        st.info("👆 Seleziona o inserisci un ticker per avviare il backtest.")

    elif run_bt_btn or ("bt_last_ticker" in st.session_state
                        and st.session_state.bt_last_ticker == bt_ticker_bt):

        st.session_state.bt_last_ticker = bt_ticker_bt
        st.divider()
        st.markdown(f"#### Risultati — {bt_ticker_bt}")

        # ── Carica prezzi ─────────────────────────────────────────────────────
        with st.spinner(f"Caricamento prezzi {bt_ticker_bt}..."):
            bt_prices = load_bt_prices(bt_ticker_bt, EODHD_API_KEY)

        if bt_prices.empty:
            st.error(
                f"❌ Prezzi non disponibili per **{bt_ticker_bt}**. "
                "Il ticker potrebbe non essere nella cache locale. "
                "Configura EODHD_API_KEY per caricare dati on-demand."
            )
        else:
            price_col_bt = ("adjusted_close" if "adjusted_close" in bt_prices.columns
                            else "close")
            n_days = len(bt_prices)
            date_range = (f"{pd.to_datetime(bt_prices['date'].min()).strftime('%d/%m/%Y')} → "
                          f"{pd.to_datetime(bt_prices['date'].max()).strftime('%d/%m/%Y')}")
            st.caption(f"📅 Storico: {date_range} — {n_days:,} sessioni")

            if n_days < 250:
                st.warning("⚠️ Storico inferiore a 1 anno — risultati statisticamente limitati.")

            # ── Esegui backtest ───────────────────────────────────────────────
            trades_df = run_backtest(bt_prices, bt_entry_z, bt_exit_z,
                                     bt_sma_stop, bt_z_win)
            stats = bt_stats(trades_df)

            if not stats:
                st.info("ℹ️ Nessun trade simulato nel periodo disponibile con i parametri selezionati. "
                        "Prova ad abbassare la soglia Z-Score Entry o ad ampliare i dati storici.")
            else:
                # ── KPI SUMMARY ───────────────────────────────────────────────
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Trade Totali",     f"{stats['n_trades']}")
                k2.metric("Win Rate",         f"{stats['win_rate']:.1f}%")
                k3.metric("Return Medio",     f"{stats['avg_pnl']:+.2f}%")
                k4.metric("Avg Winner",       f"{stats['avg_win']:+.2f}%",
                          delta=f"vs Avg Loser {stats['avg_loss']:+.2f}%",
                          delta_color="normal")
                k5.metric("Max Drawdown",     f"{stats['max_dd']:.2f}%",
                          delta=f"P&L cumulativo {stats['total_pnl']:+.1f}%",
                          delta_color="off")
                k6.metric("Giorni Medi",      f"{stats['avg_days']:.0f}gg",
                          delta=f"TP: {stats['n_tp']} | Stop: {stats['n_stop']}",
                          delta_color="off")

                st.divider()

                # ── GRAFICO PREZZI + ENTRY/EXIT ───────────────────────────────
                df_bt = bt_prices.copy().sort_values("date").reset_index(drop=True)
                close_bt  = df_bt[price_col_bt].astype(float)
                sma50_bt  = close_bt.rolling(bt_z_win).mean()
                sma200_bt = close_bt.rolling(200).mean()
                std_bt    = close_bt.rolling(bt_z_win).std()
                zscore_bt = (close_bt - sma50_bt) / std_bt
                dates_bt  = pd.to_datetime(df_bt["date"])

                entries_tp   = trades_df[trades_df["exit_reason"].str.contains("Take Profit")]
                entries_stop = trades_df[trades_df["exit_reason"].str.contains("Stop")]

                fig_px = go.Figure()
                fig_px.add_trace(go.Scatter(x=dates_bt, y=close_bt,
                    name="Prezzo", line=dict(color="#90CAF9", width=1.2)))
                fig_px.add_trace(go.Scatter(x=dates_bt, y=sma50_bt,
                    name=f"SMA{bt_z_win}", line=dict(color="#FF9800", width=1, dash="dot")))
                fig_px.add_trace(go.Scatter(x=dates_bt, y=sma200_bt,
                    name="SMA200", line=dict(color="#9E9E9E", width=1, dash="dash")))
                fig_px.add_trace(go.Scatter(
                    x=trades_df["entry_date"], y=trades_df["entry_price"],
                    mode="markers", name="Entry",
                    marker=dict(color="#4CAF50", size=9, symbol="triangle-up")))
                if not entries_tp.empty:
                    fig_px.add_trace(go.Scatter(
                        x=entries_tp["exit_date"], y=entries_tp["exit_price"],
                        mode="markers", name="Exit TP",
                        marker=dict(color="#2196F3", size=9, symbol="triangle-down")))
                if not entries_stop.empty:
                    fig_px.add_trace(go.Scatter(
                        x=entries_stop["exit_date"], y=entries_stop["exit_price"],
                        mode="markers", name="Exit Stop",
                        marker=dict(color="#F44336", size=9, symbol="triangle-down")))
                fig_px.update_layout(**_bt_dark(f"Prezzo + SMA — {bt_ticker_bt}", h=380))
                st.plotly_chart(fig_px, use_container_width=True)

                # ── GRAFICO Z-SCORE ────────────────────────────────────────────
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=dates_bt, y=zscore_bt,
                    name="Z-Score", line=dict(color="#CE93D8", width=1.2), fill="tozeroy",
                    fillcolor="rgba(206,147,216,0.08)"))
                for lvl, col, lbl in [
                    (bt_entry_z, "#F44336", f"Entry {bt_entry_z:.1f}σ"),
                    (bt_exit_z,  "#4CAF50", f"Exit +{bt_exit_z:.1f}σ"),
                    (0.0,        "#9E9E9E", "Media"),
                ]:
                    fig_z.add_hline(y=lvl, line_color=col, line_dash="dot",
                                    annotation_text=lbl, annotation_font_color=col)
                fig_z.add_trace(go.Scatter(
                    x=trades_df["entry_date"],
                    y=trades_df["entry_z"],
                    mode="markers", name="Entry",
                    marker=dict(color="#4CAF50", size=8, symbol="triangle-up")))
                fig_z.add_trace(go.Scatter(
                    x=trades_df["exit_date"],
                    y=trades_df["exit_z"],
                    mode="markers", name="Exit",
                    marker=dict(color="#FF7043", size=8, symbol="triangle-down")))
                fig_z.update_layout(**_bt_dark(f"Z-Score ({bt_z_win}gg) — {bt_ticker_bt}", h=280))
                st.plotly_chart(fig_z, use_container_width=True)

                # ── EQUITY CURVE + DISTRIBUZIONE P&L ─────────────────────────
                col_eq, col_dist = st.columns([3, 2])

                with col_eq:
                    equity_curve = trades_df["pnl_pct"].cumsum().values
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=list(range(1, len(equity_curve) + 1)),
                        y=equity_curve,
                        name="P&L Cumulativo %",
                        line=dict(color="#4CAF50" if equity_curve[-1] >= 0 else "#F44336",
                                  width=2),
                        fill="tozeroy",
                        fillcolor="rgba(76,175,80,0.1)" if equity_curve[-1] >= 0
                                  else "rgba(244,67,54,0.1)",
                    ))
                    fig_eq.add_hline(y=0, line_color="#9E9E9E", line_dash="dot")
                    fig_eq.update_layout(**_bt_dark("Equity Curve (P&L % cumulativo per trade)", h=300))
                    fig_eq.update_xaxes(title_text="Trade #", color="#9E9E9E")
                    fig_eq.update_yaxes(title_text="P&L % cumulativo", color="#9E9E9E")
                    st.plotly_chart(fig_eq, use_container_width=True)

                with col_dist:
                    colors_bar = ["#4CAF50" if v >= 0 else "#F44336"
                                  for v in trades_df["pnl_pct"]]
                    fig_dist = go.Figure(go.Bar(
                        x=list(range(1, len(trades_df) + 1)),
                        y=trades_df["pnl_pct"].values,
                        marker_color=colors_bar,
                        name="P&L per trade",
                    ))
                    fig_dist.add_hline(y=0, line_color="#9E9E9E", line_dash="dot")
                    fig_dist.update_layout(**_bt_dark("P&L % per trade", h=300))
                    fig_dist.update_xaxes(title_text="Trade #", color="#9E9E9E")
                    fig_dist.update_yaxes(title_text="P&L %", color="#9E9E9E")
                    st.plotly_chart(fig_dist, use_container_width=True)

                # ── TRADE LOG TABLE ───────────────────────────────────────────
                st.markdown("#### 📋 Log Operazioni Simulate")
                tlog = trades_df[[
                    "entry_date", "exit_date", "entry_price", "exit_price",
                    "entry_z", "exit_z", "days_held", "pnl_pct", "exit_reason"
                ]].copy()
                tlog.columns = [
                    "Entry", "Exit", "Prezzo Entry", "Prezzo Exit",
                    "Z Entry", "Z Exit", "Giorni", "P&L %", "Motivo Exit"
                ]
                tlog["P&L %"] = tlog["P&L %"].apply(lambda x: f"{x:+.2f}%")
                tlog["Prezzo Entry"] = tlog["Prezzo Entry"].apply(lambda x: f"${x:.2f}")
                tlog["Prezzo Exit"]  = tlog["Prezzo Exit"].apply(lambda x: f"${x:.2f}")

                st.dataframe(
                    tlog.style.apply(
                        lambda row: ["" if col != "P&L %" else
                                     ("color: #4CAF50" if "+" in str(row["P&L %"])
                                      else "color: #F44336")
                                     for col in row.index],
                        axis=1
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 35 + 35 * len(tlog)),
                )

                with st.expander("ℹ️ Note metodologiche — Backtest"):
                    st.markdown(f"""
                    **Logica di simulazione:**
                    - Entry: primo giorno in cui Z-Score attraversa verso il basso la soglia {bt_entry_z:.1f}σ
                      (crossover — evita re-entry multipli sullo stesso drawdown)
                    - Exit TP: primo giorno in cui Z-Score ≥ +{bt_exit_z:.1f}σ
                    - Exit Time-Stop: {bt_sma_stop} giorni consecutivi con prezzo < SMA200
                      (contatore si azzera se il prezzo risale sopra SMA200)
                    - Un solo trade aperto per volta per ticker

                    **Limitazioni:**
                    - Il backtest non simula lo stop fondamentale (F-Score < 5, FCF negativo)
                      perché non abbiamo serie storiche fondamentali trimestrali
                    - Selection bias: il ticker è stato scelto perché *oggi* supera i filtri FA.
                      Non sappiamo se li avrebbe superati in tutti i periodi storici testati.
                    - No slippage, no spread bid/ask, no commissioni nel P&L simulato.
                    - Z-Score calcolato su finestra {bt_z_win} giorni (modificabile con lo slider).
                    """)
