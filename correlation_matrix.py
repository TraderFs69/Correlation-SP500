import os
import time
import random
import datetime as dt
from typing import Dict, List, Tuple, Optional

import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(
    page_title="Corrélation – Tickers manuels (Polygon)",
    layout="wide"
)
st.title("📊 Matrice de corrélation – Tickers manuels (Polygon)")

# ==============================
# Clé API Polygon
# ==============================
POLY = st.secrets.get("POLYGON_API_KEY", None)
if POLY is None:
    POLY = os.getenv("POLYGON_API_KEY")

if not POLY:
    st.error("⚠️ POLYGON_API_KEY manquant. Ajoute-le dans `.env` ou dans les Secrets Streamlit.")
    st.stop()

st.sidebar.caption(f"Polygon key loaded: {POLY[:4]}*** (len={len(POLY)})")

# ==============================
# Paramètres généraux
# ==============================
LIMIT_DAYS = 50000  # large

# ==============================
# Téléchargement Polygon – daily OHLCV
# ==============================
def polygon_aggs_daily(ticker: str, years: int) -> Optional[pd.DataFrame]:
    """
    Récupère 'years' années de chandelles daily via Polygon :
    /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
    Retourne un DataFrame indexé par date, colonnes : Open, High, Low, Close, Volume.
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(years * 365.25))

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": LIMIT_DAYS,
        "apiKey": POLY,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            # On essaie d'extraire un message d'erreur lisible
            try:
                js_err = r.json()
                msg = js_err.get("error", js_err.get("message", str(js_err)))
            except Exception:
                msg = r.text[:200]
            if st.session_state.get("debug_polygon", False):
                st.sidebar.error(f"Polygon KO pour {ticker}: HTTP {r.status_code} – {msg}")
            return None

        js = r.json()
        results = js.get("results", [])
        if not results:
            if st.session_state.get("debug_polygon", False):
                st.sidebar.warning(f"Polygon: aucun 'results' pour {ticker}.")
            return None

        df = pd.DataFrame(results)
        rename_map = {
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "t": "ts",
        }
        df = df.rename(columns=rename_map)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        df = df.set_index("ts").sort_index()

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not keep:
            return None

        out = df[keep].astype(float)
        return out

    except Exception as e:
        if st.session_state.get("debug_polygon", False):
            st.sidebar.error(f"Exception Polygon pour {ticker}: {e}")
        return None


@st.cache_data(show_spinner=False)
def download_bars_polygon_safe(
    tickers: Tuple[str, ...],
    years: int
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Télécharge les données daily pour chaque ticker via Polygon.
    Retourne (bars_dict, failed_tickers).
    """
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for t in tickers:
        dft = polygon_aggs_daily(t, years=years)
        if dft is not None and not dft.empty:
            out[t] = dft
        else:
            failed.append(t)
        # petite pause pour éviter de spam l'API
        time.sleep(0.25 + random.random() * 0.25)

    return out, failed

# ==============================
# Sidebar – paramètres utilisateur
# ==============================
st.sidebar.header("Paramètres")

years = st.sidebar.slider(
    "Nombre d'années d'historique",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
)

max_tickers = st.sidebar.number_input(
    "Nombre max de tickers à utiliser",
    min_value=2,
    max_value=50,
    value=10,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.header("Debug Polygon")
debug_polygon = st.sidebar.checkbox("Activer le debug Polygon", value=False)
st.session_state["debug_polygon"] = debug_polygon
test_symbol = st.sidebar.text_input("Ticker test (Polygon)", "AAPL")

if debug_polygon and st.sidebar.button("Tester ce ticker maintenant"):
    dft_test = polygon_aggs_daily(test_symbol.upper(), years=years)
    if dft_test is None or dft_test.empty:
        st.sidebar.error(f"❌ Polygon n'a renvoyé AUCUNE donnée pour {test_symbol.upper()}.")
    else:
        st.sidebar.success(f"✅ Polygon OK pour {test_symbol.upper()} – {len(dft_test)} barres daily.")
        st.sidebar.write(dft_test.tail())

# ==============================
# Saisie des tickers (manuelle)
# ==============================
st.subheader("🧮 Sélection des tickers")

default_text = "AMZN, AAPL, MSFT, GOOGL, IBM, NVDA"
tickers_str = st.text_area(
    "Liste de tickers (séparés par virgule, espace ou retour de ligne)",
    value=default_text,
    height=100,
)

# Parsing des tickers
raw_tokens = re.split(r"[,\s]+", tickers_str.upper())
tickers_list = sorted(set([t.strip() for t in raw_tokens if t.strip() != ""]))

if len(tickers_list) == 0:
    st.warning("Entre au moins 2 tickers pour calculer une matrice de corrélation.")
    st.stop()

if len(tickers_list) < 2:
    st.warning(f"Tu as entré un seul ticker ({tickers_list[0]}). Il en faut au moins 2.")
    st.stop()

# Appliquer la limite max_tickers
if len(tickers_list) > max_tickers:
    tickers_list = tickers_list[: max_tickers]
    st.info(f"Limitation à {max_tickers} tickers (tu peux changer la limite dans la sidebar).")

st.write(f"**Tickers utilisés ({len(tickers_list)}) :** {', '.join(tickers_list)}")

# ==============================
# Bouton d'exécution
# ==============================
go = st.button("▶️ Construire la matrice de corrélation")
if not go:
    st.stop()

# ==============================
# Téléchargement des données
# ==============================
tickers_tuple = tuple(tickers_list)

with st.spinner("Téléchargement des chandelles daily (Polygon)…"):
    bars, failed = download_bars_polygon_safe(tickers_tuple, years=years)

valid = sum(1 for t in tickers_tuple if t in bars and bars[t] is not None and not bars[t].empty)
st.caption(f"✅ Jeux de données valides : {valid}/{len(tickers_tuple)}")

if failed:
    st.warning(
        f"⚠️ Tickers échoués: {len(failed)} — ex.: {', '.join(failed[:8])}"
        + ("…" if len(failed) > 8 else "")
    )

if valid < 2:
    st.error("Pas assez de tickers valides pour calculer une matrice de corrélation.")
    st.stop()

# ==============================
# Construction des returns & corrélation
# ==============================
# DataFrame des Close alignés par date
close_prices = pd.DataFrame({
    t: bars[t]["Close"]
    for t in tickers_list
    if t in bars and "Close" in bars[t].columns and not bars[t].empty
})

# On enlève les dates avec des NaN
close_prices = close_prices.dropna(how="any")

if close_prices.shape[1] < 2:
    st.error("Moins de deux séries complètes après alignement des dates.")
    st.stop()

# Returns simples (pas log return)
returns = close_prices.pct_change().dropna(how="any")

if returns.empty:
    st.error("Impossible de calculer les returns (data vide après pct_change).")
    st.stop()

corr_mat = returns.corr()

# ==============================
# Affichage – Table + Heatmap
# ==============================
st.subheader("📊 Matrice de corrélation des returns (daily)")

st.dataframe(corr_mat, use_container_width=True)

fig = px.imshow(
    corr_mat,
    text_auto=False,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig.update_layout(
    width=None,
    height=700,
    xaxis_title="Ticker",
    yaxis_title="Ticker",
    coloraxis_colorbar=dict(title="Corr")
)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Export CSV
# ==============================
csv_corr = corr_mat.to_csv().encode("utf-8")
st.download_button(
    "💾 Télécharger la matrice (CSV)",
    data=csv_corr,
    file_name="correlation_matrix_manual_tickers.csv",
    mime="text/csv",
)

st.markdown(
    f"""
_Notes :_
- Corrélation basée sur les **returns daily simples** sur ~{years} an(s).
- Tu peux entrer n'importe quels tickers pris en charge par Polygon (ex.: `AMZN`, `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `SPY`, etc.).
- Limite actuelle dans la sidebar : **{max_tickers} tickers max**.
"""
)
