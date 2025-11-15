import os
import time
import random
import datetime as dt
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================
# Config Streamlit (à faire en premier)
# ==============================
st.set_page_config(
    page_title="Corrélation S&P 500 – Polygon",
    layout="wide"
)
st.title("📊 Matrice de corrélation – S&P 500 (Polygon)")

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
YEARS_DEFAULT = 3        # valeur par défaut (modifiable dans la sidebar)
LIMIT_DAYS    = 50000    # large
MAX_TICKERS_UI = 100     # MAX dans l'UI

# ==============================
# Lecture S&P 500 (Excel local)
# ==============================
@st.cache_data(show_spinner=False)
def get_sp500_constituents() -> Tuple[pd.DataFrame, List[str]]:
    """
    Lit le fichier Excel local sp500_constituents.xlsx.
    Doit contenir au moins une colonne 'Symbol'.
    'Company' et 'Sector' sont optionnelles (créées si absentes).
    """
    path = "sp500_constituents.xlsx"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier {path} introuvable. Ajoute-le dans ton repo (même dossier que correlation_matrix.py)."
        )

    df = pd.read_excel(path)

    if "Symbol" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne 'Symbol'.")

    if "Company" not in df.columns:
        df["Company"] = df["Symbol"]

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"] != ""]

    tickers = df["Symbol"].tolist()
    return df, tickers

# ==============================
# Téléchargement Polygon – daily OHLCV
# ==============================
def polygon_aggs_daily(ticker: str, years: int) -> Optional[pd.DataFrame]:
    """
    Récupère 'years' années de chandelles daily via Polygon.
    Utilise /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
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
            # On loggue dans la sidebar en mode debug
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
    Retourne (bars_dict, failed_tickers)
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
    value=YEARS_DEFAULT,
    step=1,
)

max_tickers = st.sidebar.number_input(
    "Nombre max de tickers pour la matrice",
    min_value=5,
    max_value=MAX_TICKERS_UI,
    value=30,
    step=5,
)

st.sidebar.markdown("---")
st.sidebar.header("Filtre S&P 500")

# Chargement S&P 500
with st.spinner("Chargement de la liste S&P 500…"):
    try:
        sp_df, all_tickers = get_sp500_constituents()
    except Exception as e:
        st.error(f"Erreur lors du chargement du S&P 500 : {e}")
        st.stop()

sectors = sorted(sp_df["Sector"].dropna().unique().tolist())
sector_sel = st.sidebar.multiselect("Secteurs", sectors, [])

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
# Filtrage des tickers
# ==============================
if sector_sel:
    base = sp_df[sp_df["Sector"].isin(sector_sel)].copy()
else:
    base = sp_df.copy()

base_list = base["Symbol"].tolist()
total_avail = len(base_list)

if total_avail == 0:
    st.warning("Aucun ticker disponible avec ces filtres (secteur).")
    st.stop()

# On applique la limite max_tickers
base_list = base_list[: int(max_tickers)]
st.caption(
    f"Tickers filtrés : {len(base_list)} (sur {total_avail} disponibles avec ce filtre)."
)

st.write("**Tickers utilisés pour la matrice :**")
st.write(", ".join(base_list))

# ==============================
# Bouton d'exécution
# ==============================
go = st.button("▶️ Construire la matrice de corrélation")
if not go:
    st.stop()

# ==============================
# Téléchargement des données
# ==============================
tickers_tuple = tuple(sorted(set(base_list)))

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
# On construit un DataFrame des Close alignés par date
close_prices = pd.DataFrame({
    t: bars[t]["Close"]
    for t in base_list
    if t in bars and "Close" in bars[t].columns and not bars[t].empty
})

# On enlève les dates où il manque des valeurs
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
    file_name="correlation_matrix_sp500.csv",
    mime="text/csv",
)

st.markdown(
    f"""
_Notes :_
- Corrélation basée sur les **returns daily simples** sur ~{years} an(s).
- Maximum de **{MAX_TICKERS_UI} tickers** sélectionnables dans l'interface (actuellement : {len(base_list)}).
- Tu peux changer les secteurs, le nombre d'années et le nombre max de tickers dans la sidebar.
"""
)
