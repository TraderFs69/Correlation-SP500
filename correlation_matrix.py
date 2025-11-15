# correlation_matrix.py
import os
import time
import random
import datetime as dt
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

# ==============================
# Config Streamlit (doit être le 1er st.*)
# ==============================
st.set_page_config(
    page_title="Corrélation sectorielle – S&P 500 (Polygon)",
    layout="wide"
)
st.title("📊 Matrice de corrélation par secteur – S&P 500 (Polygon)")

# ==============================
# Clé API Polygon
# ==============================
POLY = st.secrets.get("POLYGON_API_KEY", None)
if POLY is None:
    POLY = os.getenv("POLYGON_API_KEY")

if not POLY:
    st.error("⚠️ POLYGON_API_KEY manquant. Ajoute-le dans `.env` ou dans les Secrets Streamlit.")
    st.stop()

st.sidebar.caption(f"Polygon key: {POLY[:4]}*** (len={len(POLY)})")

# ==============================
# Lecture S&P 500 (Excel local)
# ==============================
@st.cache_data(show_spinner=False)
def get_sp500_constituents() -> Tuple[pd.DataFrame, List[str]]:
    """
    Lit le fichier Excel sp500_constituents.xlsx (même dossier que ce script).
    Doit contenir au minimum une colonne 'Symbol'.
    """
    path = "sp500_constituents.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable. Assure-toi qu'il est présent dans le repo, "
            "au même niveau que correlation_matrix.py."
        )

    df = pd.read_excel(path)

    if "Symbol" not in df.columns:
        raise ValueError("Le fichier Excel doit contenir une colonne 'Symbol'.")

    # Colonnes minimales
    if "Company" not in df.columns:
        df["Company"] = df["Symbol"]
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"] != ""]

    tickers = df["Symbol"].tolist()
    return df, tickers


# ==============================
# Polygon – téléchargement OHLC daily
# ==============================
def _polygon_daily_close(
    ticker: str,
    years: int
) -> Optional[pd.Series]:
    """
    Récupère {years} ans de daily via Polygon et renvoie une Series des 'Close'.
    Pas de Heikin, juste la clôture normale.
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(years * 365.25))

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLY,
    }

    retry_delays = [0.4, 0.8, 1.6, 3.2]
    last_error = None

    for delay in retry_delays:
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                try:
                    js_err = r.json()
                    msg = js_err.get("error", js_err.get("message", str(js_err)))
                except Exception:
                    msg = r.text[:200]
                last_error = f"HTTP {r.status_code} – {msg}"
                time.sleep(delay)
                continue

            js = r.json()
            results = js.get("results", [])
            if not results:
                last_error = "Empty results"
                time.sleep(delay)
                continue

            df = pd.DataFrame(results)
            if "c" not in df.columns or "t" not in df.columns:
                last_error = "Colonnes 'c' ou 't' absentes"
                time.sleep(delay)
                continue

            df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("ts").sort_index()
            close = df["c"].astype(float)
            close.name = ticker
            return close

        except Exception as e:
            last_error = f"Exception: {e}"
            time.sleep(delay)

    # Debug discret dans la sidebar (optionnel)
    st.sidebar.warning(f"Polygon KO pour {ticker}: {last_error}")
    return None


@st.cache_data(show_spinner=True)
def download_sector_prices(
    tickers: Tuple[str, ...],
    years: int
) -> pd.DataFrame:
    """
    Télécharge les clôtures daily pour une liste de tickers (tuple) sur 'years' années.
    Renvoie un DataFrame indexé par date, colonnes = tickers disponibles.
    """
    closes: Dict[str, pd.Series] = {}
    for t in tickers:
        serie = _polygon_daily_close(t, years)
        if serie is not None and len(serie) > 30:
            closes[t] = serie
        # petite pause anti-throttle
        time.sleep(0.3 + random.random() * 0.3)

    if not closes:
        return pd.DataFrame()

    # Alignement sur l'index commun (dates)
    df_prices = pd.DataFrame(closes)
    # On enlève les dates où tout est NaN
    df_prices = df_prices.dropna(how="all")
    return df_prices


# ==============================
# UI – Contrôles
# ==============================
with st.spinner("Chargement des constituants du S&P 500…"):
    sp_df, all_tickers = get_sp500_constituents()

if "Sector" not in sp_df.columns:
    st.error("La colonne 'Sector' est absente du fichier S&P 500. Impossible de filtrer par secteur.")
    st.stop()

sectors = sorted(sp_df["Sector"].dropna().unique().tolist())
st.sidebar.header("Paramètres de la matrice de corrélation")

sector_sel = st.sidebar.selectbox(
    "Secteur",
    sectors,
    index=sectors.index("Information Technology") if "Information Technology" in sectors else 0
)

max_tickers = st.sidebar.slider(
    "Nombre maximal de tickers dans le secteur",
    min_value=5,
    max_value=100,
    value=20,
    step=1
)

years = st.sidebar.slider(
    "Nombre d'années d'historique",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

min_common_days = st.sidebar.slider(
    "Nb minimum de jours communs (pour corrélation)",
    min_value=30,
    max_value=300,
    value=100,
    step=10
)

st.write(
    f"Secteur sélectionné : **{sector_sel}** — "
    f"{years} an(s) d'historique — max **{max_tickers}** tickers"
)

# Filtrage des tickers par secteur
sector_df = sp_df[sp_df["Sector"] == sector_sel].copy()
if sector_df.empty:
    st.error("Aucun ticker trouvé pour ce secteur.")
    st.stop()

sector_df = sector_df.sort_values("Symbol")
tickers_list = sector_df["Symbol"].head(max_tickers).tolist()

st.write(f"Tickers retenus ({len(tickers_list)}) : {', '.join(tickers_list)}")

go = st.button("▶️ Construire la matrice de corrélation")
if not go:
    st.stop()

# ==============================
# Téléchargement des prix & log returns
# ==============================
with st.spinner("Téléchargement des prix daily via Polygon…"):
    df_prices = download_sector_prices(tuple(tickers_list), years)

if df_prices.empty:
    st.error("Aucune série de prix utilisable n'a été récupérée. Vérifie la clé Polygon ou réessaie.")
    st.stop()

st.write(f"Dimensions des prix bruts: {df_prices.shape[0]} jours x {df_prices.shape[1]} tickers")

# On ne garde que les colonnes avec suffisamment de données
valid_cols = [
    c for c in df_prices.columns
    if df_prices[c].count() >= min_common_days
]
df_prices = df_prices[valid_cols]

if df_prices.shape[1] < 2:
    st.error("Pas assez de tickers avec suffisamment de données pour calculer une corrélation.")
    st.stop()

st.write(f"Tickers avec au moins {min_common_days} jours de données: {df_prices.shape[1]}")

# Log returns
df_logret = np.log(df_prices / df_prices.shift(1))
df_logret = df_logret.dropna(how="all")

if df_logret.empty:
    st.error("Impossible de calculer des log returns (beaucoup de NaN).")
    st.stop()
    
# === Paramètre utilisateur : nombre maximal de titres pour la matrice ===
max_corr = st.number_input(
    "Nombre maximal de titres pour la matrice de corrélation",
    min_value=5,
    max_value=100,
    value=30,
    step=5
)

# === Limitation de la liste utilisée pour la matrice ===
corr_list = base_list[: int(max_corr)]

st.info(f"Matrice basée sur {len(corr_list)} tickers : {', '.join(corr_list)}")

# Téléchargement des données seulement pour les tickers de corrélation
tickers_tuple = tuple(sorted(set(corr_list)))

with st.spinner("Téléchargement des données pour corrélation…"):
    bars, failed = download_bars_polygon_safe(tickers_tuple)

# Construction dataframe des close
close_prices = pd.DataFrame({
    t: bars[t]["Close"] for t in corr_list 
    if t in bars and bars[t] is not None and not bars[t].empty
})

# Nettoyage
close_prices = close_prices.dropna()

# Calcul returns
returns = close_prices.pct_change().dropna()

# Affichage matrice
corr_mat = returns.corr()

st.subheader("📊 Matrice de corrélation")
st.dataframe(corr_mat, use_container_width=True)

# Export CSV
csv_corr = corr_mat.to_csv().encode("utf-8")
st.download_button("💾 Télécharger la matrice (CSV)", data=csv_corr, file_name="correlation_matrix.csv")

# ==============================
# Matrice de corrélation
# ==============================
corr = df_logret.corr()

st.subheader("🔗 Matrice de corrélation (log returns)")
st.caption("Corrélation de Pearson sur les log returns journaliers.")

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title=f"Corrélation – {sector_sel}",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(corr, use_container_width=True)

# ==============================
# Paires les plus corrélées / décorrélées
# ==============================
pairs = []
cols = corr.columns.tolist()
n = len(cols)

for i in range(n):
    for j in range(i + 1, n):
        pairs.append({
            "Ticker 1": cols[i],
            "Ticker 2": cols[j],
            "Corr": corr.iloc[i, j]
        })

pairs_df = pd.DataFrame(pairs)
pairs_df["|Corr|"] = pairs_df["Corr"].abs()

top_n = min(25, len(pairs_df))

st.subheader(f"🔥 Top {top_n} paires les plus corrélées (en valeur absolue)")
st.dataframe(
    pairs_df.sort_values("|Corr|", ascending=False).head(top_n),
    use_container_width=True
)

st.subheader(f"❄️ Top {top_n} paires les plus DÉcorrélées (proches de 0)")
st.dataframe(
    pairs_df.sort_values("|Corr|", ascending=True).head(top_n),
    use_container_width=True
)

# Export CSV
csv_corr = corr.to_csv().encode("utf-8")
st.download_button(
    "💾 Télécharger la matrice de corrélation (CSV)",
    data=csv_corr,
    file_name=f"correlation_matrix_{sector_sel.replace(' ', '_')}.csv",
    mime="text/csv"
)
