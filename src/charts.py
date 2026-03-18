"""
src/charts.py — Libreria grafici Plotly per il Fallen Angels Screener.

Tutti i grafici usano il dark theme Kriterion Quant e sono ottimizzati
per la visualizzazione in Streamlit (use_container_width=True).

Funzioni esportate:
    - build_price_sma_chart()       → Tab 2: prezzo + SMA50 + SMA200 + volume
    - build_zscore_chart()          → Tab 2: Z-Score rolling con threshold
    - build_volume_chart()          → Tab 2: volume vs ADV20 con anomalie
    - build_fscore_breakdown()      → Tab 2: breakdown 9 componenti F-Score
    - build_fcf_yield_trend()       → Tab 2: FCF Yield trimestrale
    - build_icr_trend()             → Tab 2: ICR trimestrale con soglia settoriale
    - build_fscore_history()        → Tab 3: trend F-Score nel tempo
    - build_fcf_history()           → Tab 3: trend FCF nel tempo
    - build_icr_history()           → Tab 3: trend ICR nel tempo
    - build_sma200_monitor()        → Tab 3: prezzo vs SMA200 + counter giorni
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PALETTE COLORI — DARK THEME KRITERION QUANT
# ============================================================
COLORS = {
    "primary":    "#2196F3",   # blu — linee principali, highlights
    "secondary":  "#FF9800",   # arancio — SMA50, linee secondarie
    "positive":   "#4CAF50",   # verde — valori positivi, OK
    "negative":   "#F44336",   # rosso — valori negativi, alert
    "neutral":    "#9E9E9E",   # grigio — riferimento, SMA200
    "background": "#1E1E2E",   # sfondo scuro
    "surface":    "#2A2A3E",   # pannelli / card
    "text":       "#E0E0E0",   # testo principale
    "accent":     "#AB47BC",   # viola — indicatori speciali
    "warning":    "#FFC107",   # giallo — warning
    "grid":       "#333355",   # griglia grafici
}


def _base_layout(
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 400,
) -> dict:
    """Layout Plotly condiviso — dark theme professionale."""
    return dict(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"]), x=0.01),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        height=height,
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, color=COLORS["text"],
            showspikes=True, spikecolor=COLORS["neutral"], spikethickness=1,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, color=COLORS["text"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"],
            font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=50, b=50),
    )


# ============================================================
# TAB 2 — TICKER DETAIL CHARTS
# ============================================================

def build_price_sma_chart(
    prices: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
) -> go.Figure:
    """
    Grafico prezzo adjusted con SMA50 (arancio), SMA200 (grigio) e volume.

    Args:
        prices:       DataFrame con colonne [date, close/adjusted_close, volume, sma50, sma200]
                      oppure raw prices (sma calcolate internamente)
        ticker:       Simbolo per il titolo del grafico
        show_volume:  Aggiunge pannello volume sotto (default True)

    Returns:
        go.Figure con 1 o 2 subplot (price + volume opzionale)
    """
    if prices.empty:
        return go.Figure()

    p = prices.sort_values("date").copy()
    price_col = "adjusted_close" if "adjusted_close" in p.columns else "close"

    # Calcola SMA se non già presenti nel DataFrame
    if "sma50" not in p.columns:
        p["sma50"] = p[price_col].rolling(50).mean()
    if "sma200" not in p.columns:
        p["sma200"] = p[price_col].rolling(200).mean()

    # Layout con 2 righe se vogliamo anche il volume
    rows = 2 if (show_volume and "volume" in p.columns) else 1
    row_heights = [0.72, 0.28] if rows == 2 else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=("", "Volume" if rows == 2 else ""),
    )

    # --- Prezzo ---
    fig.add_trace(go.Scatter(
        x=p["date"], y=p[price_col],
        name="Prezzo",
        line=dict(color=COLORS["primary"], width=1.8),
        hovertemplate="%{y:$.2f}<extra>Prezzo</extra>",
    ), row=1, col=1)

    # --- SMA50 ---
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["sma50"],
        name="SMA50",
        line=dict(color=COLORS["secondary"], width=1.2, dash="dot"),
        hovertemplate="%{y:$.2f}<extra>SMA50</extra>",
    ), row=1, col=1)

    # --- SMA200 ---
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["sma200"],
        name="SMA200",
        line=dict(color=COLORS["neutral"], width=1.2, dash="dash"),
        hovertemplate="%{y:$.2f}<extra>SMA200</extra>",
    ), row=1, col=1)

    # --- Volume ---
    if rows == 2 and "volume" in p.columns:
        price_vals  = p[price_col].values
        price_shift = p[price_col].shift(1).values
        vol_colors  = [
            COLORS["positive"] if c >= o else COLORS["negative"]
            for c, o in zip(price_vals, price_shift)
        ]
        fig.add_trace(go.Bar(
            x=p["date"], y=p["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.75,
            hovertemplate="%{y:,.0f}<extra>Volume</extra>",
        ), row=2, col=1)

        # Linea ADV20 sul volume
        if "volume" in p.columns:
            adv20 = p["volume"].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=p["date"], y=adv20,
                name="ADV20",
                line=dict(color=COLORS["warning"], width=1, dash="dot"),
                hovertemplate="%{y:,.0f}<extra>ADV20</extra>",
            ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            f"{ticker} — Prezzo | SMA50 | SMA200",
            height=500 if rows == 2 else 380,
        ),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Prezzo (USD)", row=1, col=1, color=COLORS["text"])
    if rows == 2:
        fig.update_yaxes(title_text="Volume", row=2, col=1, color=COLORS["text"])

    return fig


def build_zscore_chart(prices: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico Z-Score rolling (50 periodi) con linea trigger a -2.5.

    Evidenzia in rosso le zone dove Z <= -2.5 (trigger operativo attivo).

    Args:
        prices: DataFrame con colonne [date, z_score] o raw prices per calcolo
        ticker: Simbolo

    Returns:
        go.Figure
    """
    if prices.empty:
        return go.Figure()

    p = prices.sort_values("date").copy()

    if "z_score" not in p.columns:
        price_col = "adjusted_close" if "adjusted_close" in p.columns else "close"
        close = p[price_col].astype(float)
        sma50 = close.rolling(50).mean()
        std50 = close.rolling(50).std()
        p["z_score"] = (close - sma50) / std50

    p = p.dropna(subset=["z_score"])

    fig = go.Figure()

    # Area positiva (Z > 0) in blu chiaro
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["z_score"].clip(lower=0),
        name="Z > 0",
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Area negativa (Z < 0) in rosso chiaro
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["z_score"].clip(upper=0),
        name="Z < 0",
        fill="tozeroy",
        fillcolor="rgba(244,67,54,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Linea Z-Score
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["z_score"],
        name="Z-Score",
        line=dict(color=COLORS["primary"], width=1.5),
        hovertemplate="%{y:.3f}<extra>Z-Score</extra>",
    ))

    # Linea trigger -2.5
    fig.add_hline(
        y=-2.5,
        line_color=COLORS["negative"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=" Trigger -2.5σ",
        annotation_font_color=COLORS["negative"],
        annotation_position="bottom right",
    )

    # Linea zero
    fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=0.8, opacity=0.6)

    # Evidenzia i punti sotto il trigger
    trigger_points = p[p["z_score"] <= -2.5]
    if not trigger_points.empty:
        fig.add_trace(go.Scatter(
            x=trigger_points["date"],
            y=trigger_points["z_score"],
            name="Trigger attivo",
            mode="markers",
            marker=dict(color=COLORS["negative"], size=6, symbol="circle"),
            hovertemplate="%{y:.3f}<extra>Trigger</extra>",
        ))

    fig.update_layout(**_base_layout(
        f"{ticker} — Z-Score (SMA50 | 50gg)",
        y_title="Z-Score (σ)",
        height=350,
    ))
    return fig


def build_fscore_breakdown(fscore_data: dict, ticker: str) -> go.Figure:
    """
    Grafico orizzontale dei 9 componenti del Piotroski F-Score.

    Verde = componente attivo (1), Rosso = componente inattivo (0).
    Raggruppa visivamente per le 3 categorie (Profittabilità, Leva/Liquidità, Efficienza).

    Args:
        fscore_data: Dict con chiavi F_ROA, F_CFO, F_DROA, F_ACC, F_DLEV,
                     F_DLIQ, F_EQ, F_DMAR, F_DTURN, f_score
        ticker:      Simbolo

    Returns:
        go.Figure
    """
    if not fscore_data:
        return go.Figure()

    # Definizione componenti con etichette leggibili e categoria
    components = [
        # (chiave, etichetta, categoria)
        ("F_DTURN",  "Asset Turnover ↑",      "C — Efficienza"),
        ("F_DMAR",   "Gross Margin ↑",         "C — Efficienza"),
        ("F_EQ",     "No Diluizione",           "C — Efficienza"),
        ("F_DLIQ",   "Current Ratio ↑",         "B — Leva & Liquidità"),
        ("F_DLEV",   "Leva Finanziaria ↓",      "B — Leva & Liquidità"),
        ("F_ACC",    "Qualità Accruals",         "B — Leva & Liquidità"),
        ("F_DROA",   "ROA in Miglioramento",     "A — Profittabilità"),
        ("F_CFO",    "Cash Flow Op. > 0",        "A — Profittabilità"),
        ("F_ROA",    "ROA > 0",                  "A — Profittabilità"),
    ]

    labels  = [c[1] for c in components]
    values  = [fscore_data.get(c[0], 0) for c in components]
    groups  = [c[2] for c in components]
    colors  = [COLORS["positive"] if v == 1 else COLORS["negative"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[str(v) for v in values],
        textposition="inside",
        textfont=dict(color="white", size=13, weight="bold"),
        customdata=groups,
        hovertemplate="<b>%{y}</b><br>Categoria: %{customdata}<br>Punteggio: %{x}<extra></extra>",
    ))

    total_score = fscore_data.get("f_score", sum(values))

    # Colore del titolo in base al punteggio
    score_color = (
        COLORS["positive"] if total_score >= 7
        else COLORS["warning"] if total_score >= 5
        else COLORS["negative"]
    )

    fig.update_layout(
        **_base_layout(
            f"{ticker} — Piotroski F-Score: {total_score}/9",
            x_title="Punteggio (0 = inattivo, 1 = attivo)",
            height=380,
        ),
        showlegend=False,
    )
    # update_xaxes separato per evitare conflitto con la chiave 'xaxis' già in _base_layout
    fig.update_xaxes(range=[0, 1.3], tickvals=[0, 1], ticktext=["0", "1"],
                     showgrid=False, color=COLORS["text"])

    # Linea verticale a x=1 (massimo per componente)
    fig.add_vline(x=1, line_color=COLORS["grid"], line_dash="dot", line_width=1)

    return fig


def build_fcf_yield_trend(fcf_data: pd.DataFrame, ticker: str, sector: str = "") -> go.Figure:
    """
    Grafico a barre del FCF Yield per gli ultimi 8 trimestri.

    Args:
        fcf_data: DataFrame con colonne [date, fcf_yield] (valori 0-1 come decimale)
        ticker:   Simbolo
        sector:   Settore GICS (mostrato nel titolo)

    Returns:
        go.Figure
    """
    if fcf_data.empty or "fcf_yield" not in fcf_data.columns:
        return go.Figure()

    df = fcf_data.sort_values("date").tail(8).copy()
    df["fcf_yield_pct"] = df["fcf_yield"] * 100
    bar_colors = [
        COLORS["positive"] if v >= 5 else COLORS["warning"] if v >= 3 else COLORS["negative"]
        for v in df["fcf_yield_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=df["date"].dt.strftime("%Y-Q%q") if hasattr(df["date"].dt, "quarter") else df["date"].astype(str),
        y=df["fcf_yield_pct"],
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in df["fcf_yield_pct"]],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="%{x}<br>FCF Yield: %{y:.2f}%<extra></extra>",
    ))

    # Soglia 5%
    fig.add_hline(
        y=5, line_color=COLORS["positive"],
        line_dash="dash", line_width=1.5,
        annotation_text=" Soglia 5%",
        annotation_font_color=COLORS["positive"],
        annotation_position="top right",
    )

    sector_label = f" | {sector}" if sector else ""
    fig.update_layout(**_base_layout(
        f"{ticker} — FCF Yield TTM{sector_label}",
        x_title="Trimestre",
        y_title="FCF Yield (%)",
        height=320,
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


def build_icr_trend(icr_data: pd.DataFrame, ticker: str, sector: str = "", threshold: float = 5.0) -> go.Figure:
    """
    Grafico a barre dell'Interest Coverage Ratio per gli ultimi 8 trimestri.

    Args:
        icr_data:  DataFrame con colonne [date, icr]
        ticker:    Simbolo
        sector:    Settore GICS
        threshold: Soglia ICR settoriale

    Returns:
        go.Figure
    """
    if icr_data.empty or "icr" not in icr_data.columns:
        return go.Figure()

    df = icr_data.sort_values("date").tail(8).copy()

    # Gestisce ICR infinito (no debito)
    df["icr_display"] = df["icr"].apply(
        lambda x: min(x, threshold * 3) if x is not None and x != float("inf") else threshold * 3
    )
    bar_colors = [
        COLORS["positive"] if (v >= threshold) else COLORS["negative"]
        for v in df["icr"].fillna(0)
    ]

    x_labels = df["date"].dt.strftime("%Y-Q%q") if hasattr(df["date"].dt, "quarter") else df["date"].astype(str)

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=df["icr_display"],
        marker_color=bar_colors,
        text=[f"{v:.1f}x" if v != float("inf") else "∞" for v in df["icr"].fillna(0)],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate="%{x}<br>ICR: %{text}<extra></extra>",
    ))

    # Soglia settoriale
    fig.add_hline(
        y=threshold,
        line_color=COLORS["warning"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f" Soglia {sector}: {threshold}x",
        annotation_font_color=COLORS["warning"],
        annotation_position="top right",
    )

    fig.update_layout(**_base_layout(
        f"{ticker} — Interest Coverage Ratio (EBIT/Interest Exp.)",
        x_title="Trimestre",
        y_title="ICR (x)",
        height=320,
    ))
    return fig


# ============================================================
# TAB 3 — PHASE 2 MONITOR CHARTS
# ============================================================

def build_fscore_history(history: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico linea dell'F-Score nel tempo (ultimi 8 trimestri).
    Mostra il trend di deterioramento o miglioramento del profilo fondamentale.

    Args:
        history: DataFrame con colonne [date, f_score]
        ticker:  Simbolo

    Returns:
        go.Figure
    """
    if history.empty or "f_score" not in history.columns:
        return go.Figure()

    df = history.sort_values("date").copy()

    # Colore dei punti in base al punteggio
    point_colors = [
        COLORS["positive"] if v >= 7
        else COLORS["warning"] if v >= 5
        else COLORS["negative"]
        for v in df["f_score"]
    ]

    fig = go.Figure()

    # Area di sfondo per zona critica (F < 5)
    fig.add_hrect(
        y0=0, y1=4.9,
        fillcolor=COLORS["negative"],
        opacity=0.08,
        layer="below",
        line_width=0,
    )

    # Linea F-Score
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["f_score"],
        name="F-Score",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=point_colors, size=10, line=dict(color="white", width=1)),
        hovertemplate="%{x|%Y-%m}<br>F-Score: %{y}/9<extra></extra>",
    ))

    # Soglia operativa (>= 7)
    fig.add_hline(
        y=7, line_color=COLORS["positive"],
        line_dash="dash", line_width=1.5,
        annotation_text=" Soglia ingresso: 7",
        annotation_font_color=COLORS["positive"],
        annotation_position="top right",
    )

    # Soglia stop loss fondamentale (< 5)
    fig.add_hline(
        y=5, line_color=COLORS["negative"],
        line_dash="dash", line_width=1.5,
        annotation_text=" Stop Loss fondamentale: <5",
        annotation_font_color=COLORS["negative"],
        annotation_position="bottom right",
    )

    fig.update_layout(**_base_layout(
        f"{ticker} — F-Score TTM (trend trimestrale)",
        y_title="Piotroski F-Score",
        height=370,
    ))
    # update_yaxes separato per evitare conflitto con la chiave 'yaxis' già in _base_layout
    fig.update_yaxes(range=[0, 9.5], tickvals=list(range(10)),
                     color=COLORS["text"], showgrid=True, gridcolor=COLORS["grid"])
    return fig


def build_fcf_history(history: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico FCF Yield nel tempo con alert se negativo.

    Args:
        history: DataFrame con colonne [date, fcf_yield]
        ticker:  Simbolo

    Returns:
        go.Figure
    """
    if history.empty or "fcf_yield" not in history.columns:
        return go.Figure()

    df = history.sort_values("date").copy()
    df["fcf_pct"] = df["fcf_yield"] * 100

    bar_colors = [
        COLORS["positive"] if v >= 5
        else COLORS["warning"] if v >= 0
        else COLORS["negative"]
        for v in df["fcf_pct"]
    ]

    x_labels = df["date"].dt.strftime("%Y-%m") if hasattr(df["date"].dt, "month") else df["date"].astype(str)

    fig = go.Figure(go.Bar(
        x=x_labels, y=df["fcf_pct"],
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in df["fcf_pct"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="%{x}<br>FCF Yield: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["negative"], line_width=1.5,
                  annotation_text=" FCF Negativo = Alert",
                  annotation_font_color=COLORS["negative"])
    fig.add_hline(y=5, line_color=COLORS["positive"], line_dash="dash", line_width=1,
                  annotation_text=" 5% soglia",
                  annotation_font_color=COLORS["positive"],
                  annotation_position="top right")

    fig.update_layout(**_base_layout(
        f"{ticker} — FCF Yield Trend",
        y_title="FCF Yield (%)", height=320,
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


def build_icr_history(history: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico ICR nel tempo con soglia settoriale.

    Args:
        history: DataFrame con colonne [date, icr, icr_threshold]
        ticker:  Simbolo

    Returns:
        go.Figure
    """
    if history.empty or "icr" not in history.columns:
        return go.Figure()

    df = history.sort_values("date").copy()
    threshold = df["icr_threshold"].iloc[-1] if "icr_threshold" in df.columns else 5.0

    # Cap ICR a un valore leggibile per la visualizzazione
    df["icr_display"] = df["icr"].apply(
        lambda x: min(float(x), threshold * 3) if x is not None and x != float("inf") else threshold * 3
    )

    bar_colors = [
        COLORS["positive"] if (v is not None and v >= threshold) else COLORS["negative"]
        for v in df["icr"].fillna(0)
    ]
    x_labels = df["date"].dt.strftime("%Y-%m") if hasattr(df["date"].dt, "month") else df["date"].astype(str)

    fig = go.Figure(go.Bar(
        x=x_labels, y=df["icr_display"],
        marker_color=bar_colors,
        text=[f"{v:.1f}x" for v in df["icr_display"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text"]),
        hovertemplate="%{x}<br>ICR: %{text}<extra></extra>",
    ))

    fig.add_hline(
        y=threshold,
        line_color=COLORS["warning"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f" Soglia settoriale: {threshold}x",
        annotation_font_color=COLORS["warning"],
        annotation_position="top right",
    )

    fig.update_layout(**_base_layout(
        f"{ticker} — ICR Trend (EBIT / Interest Expense)",
        y_title="ICR (x)", height=320,
    ))
    return fig


def build_sma200_monitor(prices: pd.DataFrame, ticker: str, days_below: int = 0) -> go.Figure:
    """
    Grafico prezzo vs SMA200 con evidenziazione zona sotto la media.
    Mostra visivamente il time-stop della Fase 2.

    Args:
        prices:     DataFrame con [date, close/adjusted_close, sma200]
        ticker:     Simbolo
        days_below: Numero di giorni consecutivi attualmente sotto SMA200

    Returns:
        go.Figure
    """
    if prices.empty:
        return go.Figure()

    p = prices.sort_values("date").copy()
    price_col = "adjusted_close" if "adjusted_close" in p.columns else "close"

    if "sma200" not in p.columns:
        p["sma200"] = p[price_col].rolling(200).mean()

    p = p.dropna(subset=["sma200"])

    # Colore linea prezzo: rosso se sotto SMA200, verde se sopra
    current_price  = float(p[price_col].iloc[-1])
    current_sma200 = float(p["sma200"].iloc[-1])
    price_color    = COLORS["negative"] if current_price < current_sma200 else COLORS["positive"]

    fig = go.Figure()

    # Area sotto SMA200 in rosso tenue (zona danger)
    fig.add_trace(go.Scatter(
        x=pd.concat([p["date"], p["date"].iloc[::-1]]),
        y=pd.concat([p["sma200"], p[price_col].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(244,67,54,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Prezzo
    fig.add_trace(go.Scatter(
        x=p["date"], y=p[price_col],
        name="Prezzo",
        line=dict(color=price_color, width=1.8),
        hovertemplate="%{y:$.2f}<extra>Prezzo</extra>",
    ))

    # SMA200
    fig.add_trace(go.Scatter(
        x=p["date"], y=p["sma200"],
        name="SMA200",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
        hovertemplate="%{y:$.2f}<extra>SMA200</extra>",
    ))

    # Annotazione giorni sotto SMA200
    if days_below > 0:
        status_color = COLORS["negative"] if days_below >= 90 else COLORS["warning"]
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.97,
            text=f"📊 {days_below} gg consecutivi sotto SMA200"
                 + (" ⚠️ TIME-STOP ATTIVO" if days_below >= 90 else ""),
            showarrow=False,
            align="left",
            bgcolor="rgba(42,42,62,0.9)",
            bordercolor=status_color,
            font=dict(size=12, color=status_color),
        )

    fig.update_layout(**_base_layout(
        f"{ticker} — Prezzo vs SMA200 | Time-Stop Monitor",
        y_title="Prezzo (USD)",
        height=400,
    ))
    return fig
