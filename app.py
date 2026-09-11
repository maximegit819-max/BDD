import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import datetime

# 1. Configuration
st.set_page_config(page_title="Asset 360", layout="wide")

@st.cache_resource
def init_connection():
    try:
        db_url = st.secrets["SUPABASE_DB_URL"]
    except:
        load_dotenv()
        db_url = os.getenv("SUPABASE_DB_URL")
        
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

engine = init_connection()

# 2. Récupération des données
@st.cache_data(ttl=3600*24)
def load_assets():
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM asset", conn)

@st.cache_data(ttl=3600*24)
def load_all_prices():
    # On charge tous les prix pour calculer les corrélations ensuite
    with engine.connect() as conn:
        df = pd.read_sql("SELECT asset_id, date, close FROM historical_price", conn)
        df['date'] = pd.to_datetime(df['date'])
        # Nettoyage des doublons éventuels
        # Création du pivot : dates en index, asset_id en colonnes
        pivot = df.pivot(index='date', columns='asset_id', values='close')
        pivot = pivot.sort_index().ffill()
        return pivot

with st.spinner("Initialisation de la base de données..."):
    assets_df = load_assets()
    prices_pivot = load_all_prices()

# --- Interface Principale ---
st.title("Vue Détaillée Actif")

# Menu de sélection intelligent (Recherche par nom ou ticker)
assets_df['display_name'] = assets_df['name'] + " (" + assets_df['ticker_bloomberg'].fillna('') + ")"
asset_options = dict(zip(assets_df['display_name'], assets_df['asset_id']))

selected_asset_name = st.selectbox(
    "Recherchez un actif par nom ou ticker :",
    options=sorted(asset_options.keys())
)

if not selected_asset_name:
    st.stop()

selected_asset_id = asset_options[selected_asset_name]
asset_info = assets_df[assets_df['asset_id'] == selected_asset_id].iloc[0]

st.divider()

# --- 1. Fiche d'Identité ---
st.subheader("Fiche d'Identité")

is_index = asset_info.get('asset_type') == 'INDEX' or pd.notna(asset_info.get('issuer'))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nom", str(asset_info['name'])[:40])
    st.metric("Ticker Bloomberg", str(asset_info.get('ticker_bloomberg', 'N/A')))
with col2:
    st.metric("Type", str(asset_info.get('asset_type', 'N/A')))
    st.metric("ISIN", str(asset_info.get('isin', 'N/A')))
with col3:
    st.metric("Thème / Secteur", str(asset_info.get('sector', 'N/A')))
    st.metric("Sous Type", str(asset_info.get('asset_subtype', 'N/A')))
with col4:
    st.metric("Pays", str(asset_info.get('country', 'N/A')))
    st.metric("Devise", str(asset_info.get('currency', 'N/A')))

if is_index and pd.notna(asset_info.get('issuer')):
    st.markdown("---")
    st.markdown("**Caractéristiques de l'Indice (Run Hebdo)**")
    idx_col1, idx_col2, idx_col3, idx_col4 = st.columns(4)
    with idx_col1:
        st.metric("Émetteur", str(asset_info.get('issuer', 'N/A')))
        st.metric("Sous Secteur", str(asset_info.get('sub_sector', 'N/A')))
    with idx_col2:
        div_val = asset_info.get('dividend_yield')
        div_str = f"{float(div_val)*100:.2f} %" if pd.notna(div_val) and div_val is not None else "N/A"
        st.metric("Dividende distribué en 2025 avec effet de réinvestissement", div_str)
        
        comp_count = asset_info.get('components_count')
        comp_str = str(int(comp_count)) if pd.notna(comp_count) and comp_count is not None else "N/A"
        st.metric("Composants", comp_str)
    with idx_col3:
        st.markdown("**Construction**")
        st.write(str(asset_info.get('construction', 'N/A')))
    with idx_col4:
        st.markdown("**Spécificités**")
        st.write(str(asset_info.get('specificities', 'N/A')))


st.divider()

# --- Préparation des séries de prix pour l'actif ---
if selected_asset_id not in prices_pivot.columns:
    st.warning("Aucun historique de prix disponible pour cet actif (base de données vide pour ce ticker).")
    st.stop()

asset_prices = prices_pivot[selected_asset_id].dropna()
asset_prices = asset_prices[asset_prices > 0] # Ignorer les prix à 0 pour éviter les bugs mathématiques
if len(asset_prices) == 0:
    st.warning("Aucun historique de prix valide (non nul) disponible pour cet actif.")
    st.stop()

daily_returns = asset_prices.pct_change(fill_method=None).dropna()
current_date = asset_prices.index.max()
current_price = asset_prices.iloc[-1]
start_of_year = pd.Timestamp(year=current_date.year, month=1, day=1)

# Fonction utilitaire pour récupérer le prix au plus proche d'une date
def get_price_at(date_target):
    available_dates = asset_prices[asset_prices.index <= date_target]
    if len(available_dates) == 0:
        return np.nan
    return available_dates.iloc[-1]

def get_perf(days=None, ytd=False):
    if ytd:
        # On cherche le dernier prix de l'année précédente pour le calcul YTD
        end_of_prev_year = pd.Timestamp(year=current_date.year - 1, month=12, day=31)
        old_price = get_price_at(end_of_prev_year)
    else:
        old_price = get_price_at(current_date - pd.Timedelta(days=days))
    
    if pd.isna(old_price): return np.nan
    return ((current_price / old_price) - 1) * 100

def get_vol(days=None, ytd=False):
    if ytd:
        end_of_prev_year = pd.Timestamp(year=current_date.year - 1, month=12, day=31)
        sub_returns = daily_returns[daily_returns.index > end_of_prev_year]
    else:
        sub_returns = daily_returns[daily_returns.index >= current_date - pd.Timedelta(days=days)]
    
    if len(sub_returns) < 2: return np.nan
    return sub_returns.std() * np.sqrt(252) * 100

# --- 2. Tableaux Perf / Vol ---
st.subheader(f"Performances et Volatilités (Dernier cours : {current_price:.2f})")

perf_data = {
    "5 Jours": get_perf(days=5),
    "1 Mois": get_perf(days=30), # 30 jours calendaires ~ 21 jours ouvrés
    "3 Mois": get_perf(days=90),
    "YTD": get_perf(ytd=True),
}

vol_data = {
    "5 Jours": get_vol(days=5),
    "1 Mois": get_vol(days=30),
    "3 Mois": get_vol(days=90),
    "YTD": get_vol(ytd=True),
    "1 An": get_vol(days=365),
    "5 Ans": get_vol(days=365*5),
}

col_perf, col_vol = st.columns(2)

with col_perf:
    st.markdown("**Performances**")
    perf_df = pd.DataFrame([perf_data]).T
    perf_df.columns = ["Performance"]
    perf_df.style.format("{:.2f} %").map(lambda x: 'color: green' if pd.notna(x) and x > 0 else 'color: red' if pd.notna(x) and x < 0 else ''),

with col_vol:
    st.markdown("**Volatilité Annualisée**")
    vol_df = pd.DataFrame([vol_data]).T
    vol_df.columns = ["Volatilité"]
    st.dataframe(
        vol_df.style.format("{:.2f} %"),
        use_container_width=True
    )

st.divider()

# --- 3. Graphique Technique ---
st.subheader("Cours")

# Mapping Benchmark
benchmark_mapping = {
    'FRANCE': 'CAC Index', 'France': 'CAC Index',
    'GERMANY': 'DAX Index', 'ALLEMAGNE': 'DAX Index', 'Allemagne': 'DAX Index',
    'US': 'SPX Index', 'USA': 'SPX Index', 'UNITED STATES': 'SPX Index', 'Amérique du Nord': 'SPX Index',
    'BRITAIN': 'UKX Index',
    'SWITZERLAND': 'SMI Index',
    'JAPAN': 'NKY Index', 'Japon': 'NKY Index', 'Japon ': 'NKY Index',
    'CHINE': 'SHSZ300 INDEX', 'Chine': 'SHSZ300 INDEX', 'CHINA': 'SHSZ300 INDEX',
    'HONG KONG': 'HSI Index', 'MACAU': 'HSI Index',
    # Europe élargie -> Euro Stoxx 50
    'EUROPE': 'SX5E Index', 'EURO ZONE': 'SX5E Index', 'ZONE EURO': 'SX5E Index', 'Europe': 'SX5E Index',
    'ITALY': 'SX5E Index', 'SPAIN': 'SX5E Index', 'PORTUGAL': 'SX5E Index', 'MALTA': 'SX5E Index',
    'BELGIQUE': 'SX5E Index', 'BELGIUM': 'SX5E Index', 'NETHERLANDS': 'SX5E Index', 'LUXEMBOURG': 'SX5E Index',
    'SWEDEN': 'SX5E Index', 'DENMARK': 'SX5E Index', 'NORWAY': 'SX5E Index', 'FINLAND': 'SX5E Index',
    'FAROE ISLANDS': 'SX5E Index', 'AUSTRIA': 'SX5E Index', 'CZECH': 'SX5E Index', 'POLAND': 'SX5E Index',
    'HUNGARY': 'SX5E Index', 'IRELAND': 'SX5E Index',
    
    # --- NOUVEAUX BENCHMARKS MSCI ---
    
    # Monde, Transatlantique & Zones Mixtes -> MSCI World
    'Monde': 'MSCI WORLD Index', 
    'WORLD': 'MSCI WORLD Index', 
    'Monde - Marchés émergents': 'MSCI WORLD Index',
    'BERMUDA': 'MSCI WORLD Index', 
    'TRANSATLANTIC': 'MSCI WORLD Index', 
    'Transatlantique': 'MSCI WORLD Index',
    'Europe - US - Japon': 'MSCI WORLD Index',
    'Asie - US': 'MSCI WORLD Index',
    'EurAsie': 'MSCI WORLD Index',
    'Euro-asie': 'MSCI WORLD Index',
    'Europe - Asie': 'MSCI WORLD Index',
    'Chine - Europe': 'MSCI WORLD Index',
    
    # Asie Pacifique -> MSCI AC Asia Pacific
    'Asie': 'MSCI AC ASIA PACIFIC Index', 
    'AUSTRALIA': 'MSCI AC ASIA PACIFIC Index',
    
    # Émergents -> MSCI Emerging Markets
    'Marchés émergents': 'MSCI EM Index',
    'BRAZIL': 'MSCI EM Index', 
    'CHILE': 'MSCI EM Index', 
    'JORDAN': 'MSCI EM Index', 
    'KAZAKHSTAN': 'MSCI EM Index', 
    'RUSSIA': 'MSCI EM Index'
}

asset_country = str(asset_info.get('country', '')).strip()
benchmark_ticker = benchmark_mapping.get(asset_country)
benchmark_asset_id = None

if benchmark_ticker:
    bench_match = assets_df[assets_df['ticker_bloomberg'] == benchmark_ticker]
    if not bench_match.empty:
        benchmark_asset_id = bench_match.iloc[0]['asset_id']

show_benchmark = False
if benchmark_asset_id and benchmark_asset_id in prices_pivot.columns:
    show_benchmark = st.checkbox(f"Afficher la comparaison avec le Benchmark {benchmark_ticker} (Rebasé sur le prix de l'actif)", value=True)

fig = go.Figure()

if show_benchmark:
    bench_prices = prices_pivot[benchmark_asset_id].dropna()
    bench_prices = bench_prices[bench_prices > 0] # Ignorer les prix à 0
    start_date = asset_prices.index[0]
    
    # On garde le vrai prix de l'actif
    asset_norm = asset_prices
    
    bench_sub = bench_prices[bench_prices.index >= start_date]
    if not bench_sub.empty:
        # On rebase le benchmark pour qu'il démarre exactement au même prix que l'actif
        bench_norm = (bench_sub / bench_sub.iloc[0]) * asset_prices.iloc[0]
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm, mode='lines', 
            name=f'Benchmark ({benchmark_ticker})', 
            line=dict(color='gray', width=1.5, dash='dot')
        ))
    
    sma50 = asset_norm.rolling(window=50, min_periods=1).mean()
    sma200 = asset_norm.rolling(window=200, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(x=asset_norm.index, y=asset_norm, mode='lines', name='Prix', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50, mode='lines', name='SMA 50', line=dict(color='#00d2ff', width=1.5)))
    fig.add_trace(go.Scatter(x=sma200.index, y=sma200, mode='lines', name='SMA 200', line=dict(color='#ff512f', width=1.5)))
else:
    sma50 = asset_prices.rolling(window=50, min_periods=1).mean()
    sma200 = asset_prices.rolling(window=200, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(x=asset_prices.index, y=asset_prices, mode='lines', name='Prix', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=sma50.index, y=sma50, mode='lines', name='SMA 50', line=dict(color='#00d2ff', width=1.5)))
    fig.add_trace(go.Scatter(x=sma200.index, y=sma200, mode='lines', name='SMA 200', line=dict(color='#ff512f', width=1.5)))

fig.update_layout(hovermode="x unified", height=500, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 4. Analyse des Corrélations (1 an glissant) ---
st.subheader("Corrélations sur 1 An Glissant")

# Extraction des valeurs uniques pour les filtres
subtypes = sorted([str(x) for x in assets_df['asset_subtype'].dropna().unique()])
sectors = sorted([str(x) for x in assets_df['sector'].dropna().unique()])
countries = sorted([str(x) for x in assets_df['country'].dropna().unique()])

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    filter_subtype = st.multiselect("Filtre Sous Type", options=subtypes)
with col_f2:
    filter_sector = st.multiselect("Filtre Secteur", options=sectors)
with col_f3:
    filter_country = st.multiselect("Filtre Pays", options=countries)

with st.spinner("Calcul des corrélations en cours..."):
    # 1 an glissant
    one_year_ago = current_date - pd.Timedelta(days=365)
    prices_1y = prices_pivot[prices_pivot.index >= one_year_ago]
    
    # Filtrage de l'univers
    mask = pd.Series(True, index=assets_df.index)
    if len(filter_subtype) > 0:
        mask = mask & (assets_df['asset_subtype'].isin(filter_subtype))
    if len(filter_sector) > 0:
        mask = mask & (assets_df['sector'].isin(filter_sector))
    if len(filter_country) > 0:
        mask = mask & (assets_df['country'].isin(filter_country))
        
    valid_ids = assets_df[mask]['asset_id'].tolist()
    
    # On garde toujours l'actif sélectionné dans le calcul, même s'il n'est pas du type filtré
    if selected_asset_id not in valid_ids:
        valid_ids.append(selected_asset_id)
    # Intersection avec les colonnes existantes
    valid_ids = [vid for vid in valid_ids if vid in prices_1y.columns]
    prices_1y = prices_1y[valid_ids]
    
    # Calcul des corrélations
    returns_1y = prices_1y.pct_change(fill_method=None).dropna(how='all')
    if selected_asset_id in returns_1y.columns:
        corr_series = returns_1y.corrwith(returns_1y[selected_asset_id]).dropna()
        
        # On enlève la corrélation de l'actif avec lui-même (qui est toujours 1.0)
        corr_series = corr_series.drop(index=selected_asset_id, errors='ignore')
        
        if len(corr_series) > 0:
            # Récupérer les noms
            id_to_name = dict(zip(assets_df['asset_id'], assets_df['display_name']))
            corr_df = corr_series.reset_index()
            corr_df.columns = ['asset_id', 'Correlation']
            corr_df['Nom de l\'actif'] = corr_df['asset_id'].map(id_to_name)
            
            top_10 = corr_df.nlargest(10, 'Correlation')[['Nom de l\'actif', 'Correlation']]
            bottom_10 = corr_df.nsmallest(10, 'Correlation')[['Nom de l\'actif', 'Correlation']]
            
            col_top, col_bottom = st.columns(2)
            with col_top:
                st.success("Les plus corrélés")
                st.dataframe(top_10.style.format({'Correlation': "{:.4f}"}), use_container_width=True, hide_index=True)
            with col_bottom:
                st.error("Les moins corrélés")
                st.dataframe(bottom_10.style.format({'Correlation': "{:.4f}"}), use_container_width=True, hide_index=True)
        else:
            st.info("Pas assez de données pour calculer les corrélations sur cet univers.")
    else:
        st.warning("L'actif sélectionné n'a pas de données sur la dernière année.")