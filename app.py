import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import io
from dotenv import load_dotenv
import plotly.graph_objects as go
import datetime

# 1. Configuration de la page
st.set_page_config(page_title="Asset 360", layout="wide")

@st.cache_resource
def init_connection():
    try:
        db_url = st.secrets["SUPABASE_DB_URL"]
    except Exception:
        load_dotenv()
        db_url = os.getenv("SUPABASE_DB_URL")
        
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    return create_engine(db_url)

engine = init_connection()

# 2. Récupération et mise en cache des données
@st.cache_data(ttl=3600*24)
def load_assets():
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM asset", conn)

@st.cache_data(ttl=3600*24)
def load_all_prices():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT asset_id, date, close FROM historical_price", conn)
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot(index='date', columns='asset_id', values='close')
        pivot = pivot.sort_index().ffill()
        return pivot

@st.cache_data(ttl=3600*24)
def compute_screener_metrics(assets_df, prices_pivot):
    if prices_pivot.empty:
        return assets_df
    
    date_now = prices_pivot.index[-1]
    last_prices = prices_pivot.iloc[-1]
    
    # Performances (1M, YTD, 1A)
    dates_1m = prices_pivot.index[prices_pivot.index <= date_now - pd.Timedelta(days=30)]
    p_1m = prices_pivot.loc[dates_1m[-1]] if len(dates_1m) > 0 else pd.Series(np.nan, index=prices_pivot.columns)
    perf_1m = ((last_prices / p_1m) - 1) * 100

    dates_ytd = prices_pivot.index[prices_pivot.index <= pd.Timestamp(year=date_now.year - 1, month=12, day=31)]
    p_ytd = prices_pivot.loc[dates_ytd[-1]] if len(dates_ytd) > 0 else pd.Series(np.nan, index=prices_pivot.columns)
    perf_ytd = ((last_prices / p_ytd) - 1) * 100

    dates_1y = prices_pivot.index[prices_pivot.index <= date_now - pd.Timedelta(days=365)]
    p_1y = prices_pivot.loc[dates_1y[-1]] if len(dates_1y) > 0 else pd.Series(np.nan, index=prices_pivot.columns)
    perf_1y = ((last_prices / p_1y) - 1) * 100

    # Volatilités annualisées (3 Mois et 1 An)
    daily_returns = prices_pivot.pct_change(fill_method=None)
    returns_3m = daily_returns.loc[daily_returns.index >= date_now - pd.Timedelta(days=90)]
    vol_3m = returns_3m.std() * np.sqrt(252) * 100

    returns_1y = daily_returns.loc[daily_returns.index >= date_now - pd.Timedelta(days=365)]
    vol_1y = returns_1y.std() * np.sqrt(252) * 100

    # Max Drawdown 1 An
    prices_1y = prices_pivot.loc[prices_pivot.index >= date_now - pd.Timedelta(days=365)]
    cummax_1y = prices_1y.cummax()
    drawdown_1y = (prices_1y - cummax_1y) / cummax_1y
    max_dd_1y = drawdown_1y.min() * 100

    # Écartement aux Moyennes Mobiles (SMA 50 et SMA 200)
    sma50 = prices_pivot.rolling(50, min_periods=5).mean().iloc[-1]
    sma200 = prices_pivot.rolling(200, min_periods=20).mean().iloc[-1]
    dist_sma50 = ((last_prices / sma50) - 1) * 100
    dist_sma200 = ((last_prices / sma200) - 1) * 100

    metrics_df = pd.DataFrame({
        'asset_id': prices_pivot.columns,
        'last_price': last_prices.values,
        'perf_1m': perf_1m.values,
        'perf_ytd': perf_ytd.values,
        'perf_1y': perf_1y.values,
        'vol_3m': vol_3m.values,
        'vol_1y': vol_1y.values,
        'max_dd_1y': max_dd_1y.values,
        'dist_sma50': dist_sma50.values,
        'dist_sma200': dist_sma200.values
    })

    merged = pd.merge(assets_df, metrics_df, on='asset_id', how='left')
    return merged

with st.spinner("Chargement des données en cours..."):
    assets_df = load_assets()
    prices_pivot = load_all_prices()
    
    # Ajout du display_name pour la recherche (avant le merge des métriques !)
    assets_df['display_name'] = assets_df['name'].fillna('Inconnu') + " (" + assets_df['ticker_bloomberg'].fillna('') + ")"
    
    screener_raw_df = compute_screener_metrics(assets_df, prices_pivot)

asset_options = dict(zip(assets_df['display_name'], assets_df['asset_id']))

# 3. Onglets principaux de navigation
tab_screener, tab_detail, tab_compare = st.tabs(["Screener Univers", "Fiche Détaillée Actif", "Comparaison de Cours"])

# ==========================================
# ONGLET 1 : SCREENER UNIVERS
# ==========================================
with tab_screener:
    st.header("Screener Univers")

    # Barre de recherche rapide
    search_query = st.text_input("Recherche rapide (Nom, Ticker Bloomberg, ISIN) :", "").strip().lower()

    # Filtres catégoriels
    types_list = sorted([str(x) for x in screener_raw_df['asset_type'].dropna().unique() if str(x).strip()])
    subtypes_list = sorted([str(x) for x in screener_raw_df['asset_subtype'].dropna().unique() if str(x).strip()])
    sectors_list = sorted([str(x) for x in screener_raw_df['sector'].dropna().unique() if str(x).strip()])
    countries_list = sorted([str(x) for x in screener_raw_df['country'].dropna().unique() if str(x).strip()])
    currencies_list = sorted([str(x) for x in screener_raw_df['currency'].dropna().unique() if str(x).strip()])

    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    with col_f1:
        sel_types = st.multiselect("Type d'actif", options=types_list)
    with col_f2:
        sel_subtypes = st.multiselect("Sous-Type", options=subtypes_list)
    with col_f3:
        sel_sectors = st.multiselect("Secteur", options=sectors_list)
    with col_f4:
        sel_countries = st.multiselect("Pays", options=countries_list)
    with col_f5:
        sel_currencies = st.multiselect("Devise", options=currencies_list)

    filter_only_priced = st.checkbox("Masquer les actifs sans historique de cours", value=True)

    # Application des filtres
    df_filtered = screener_raw_df.copy()

    if search_query:
        mask_search = (
            df_filtered['name'].astype(str).str.lower().str.contains(search_query, na=False) |
            df_filtered['ticker_bloomberg'].astype(str).str.lower().str.contains(search_query, na=False) |
            df_filtered['isin'].astype(str).str.lower().str.contains(search_query, na=False)
        )
        df_filtered = df_filtered[mask_search]

    if sel_types:
        df_filtered = df_filtered[df_filtered['asset_type'].isin(sel_types)]
    if sel_subtypes:
        df_filtered = df_filtered[df_filtered['asset_subtype'].isin(sel_subtypes)]
    if sel_sectors:
        df_filtered = df_filtered[df_filtered['sector'].isin(sel_sectors)]
    if sel_countries:
        df_filtered = df_filtered[df_filtered['country'].isin(sel_countries)]
    if sel_currencies:
        df_filtered = df_filtered[df_filtered['currency'].isin(sel_currencies)]

    if filter_only_priced:
        df_filtered = df_filtered[df_filtered['last_price'].notna() & (df_filtered['last_price'] > 0)]

    # Sélection et renommage des colonnes épurées
    columns_mapping = {
        'name': 'Nom',
        'ticker_bloomberg': 'Ticker Bloomberg',
        'isin': 'ISIN',
        'sector': 'Secteur',
        'currency': 'Devise',
        'last_price': 'Dernier Cours',
        'perf_1m': 'Perf 1M (%)',
        'perf_ytd': 'Perf YTD (%)',
        'perf_1y': 'Perf 1A (%)',
        'vol_3m': 'Vol 3M (%)',
        'vol_1y': 'Vol 1A (%)',
        'max_dd_1y': 'Max DD 1A (%)',
        'dist_sma50': 'Écart SMA 50 (%)',
        'dist_sma200': 'Écart SMA 200 (%)'
    }

    cols_to_keep = [c for c in columns_mapping.keys() if c in df_filtered.columns]
    table_display = df_filtered[cols_to_keep].rename(columns=columns_mapping)

    # Entête d'information et export
    col_info, col_exp1, col_exp2 = st.columns([3, 1, 1])
    with col_info:
        st.write(f"**Actifs affichés :** {len(table_display)} sur {len(screener_raw_df)}")

    with col_exp1:
        csv_data = table_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger CSV",
            data=csv_data,
            file_name=f"screener_univers_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                table_display.to_excel(writer, index=False, sheet_name='Screener')
            st.download_button(
                label="Télécharger Excel",
                data=excel_buffer.getvalue(),
                file_name=f"screener_univers_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception:
            st.button("Excel indisponible (installer openpyxl)", disabled=True, use_container_width=True)

    # Configuration des colonnes pour le tableau interactif
    column_config = {
        "Dernier Cours": st.column_config.NumberColumn(format="%.2f"),
        "Perf 1M (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Perf YTD (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Perf 1A (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Vol 3M (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Vol 1A (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Max DD 1A (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Écart SMA 50 (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Écart SMA 200 (%)": st.column_config.NumberColumn(format="%.2f %%")
    }

    event = st.dataframe(
        table_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=600,
        selection_mode="single-row",
        on_select="rerun"
    )

    # Si l'utilisateur clique sur une ligne, on pré-charge l'actif pour l'onglet 2
    if event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        selected_disp_name = df_filtered.iloc[selected_row_idx]['display_name']
        st.session_state["detail_asset_select"] = selected_disp_name


# ==========================================
# ONGLET 3 : COMPARAISON DE COURS
# ==========================================
with tab_compare:
    st.header("Comparaison de Cours (Base 100)")
    
    selected_compare_names = st.multiselect(
        "Ajoutez des actifs à comparer :",
        options=sorted(asset_options.keys()),
        default=[],
        key="compare_assets_multiselect"
    )
    
    if len(selected_compare_names) > 0:
        time_filter = st.radio(
            "Période",
            ["1 Mois", "3 Mois", "YTD", "1 An", "3 Ans", "5 Ans", "Max"],
            index=6,
            horizontal=True
        )
        
        selected_ids = [asset_options[name] for name in selected_compare_names]
        valid_ids = [aid for aid in selected_ids if aid in prices_pivot.columns]
        
        if len(valid_ids) == 0:
            st.warning("Aucun historique de prix n'est disponible pour les actifs sélectionnés.")
        else:
            compare_prices = prices_pivot[valid_ids].dropna(how='all')
            
            # Filtre temporel
            current_date = compare_prices.index.max()
            if time_filter == "1 Mois":
                cutoff = current_date - pd.Timedelta(days=30)
            elif time_filter == "3 Mois":
                cutoff = current_date - pd.Timedelta(days=90)
            elif time_filter == "YTD":
                cutoff = pd.Timestamp(year=current_date.year - 1, month=12, day=31)
            elif time_filter == "1 An":
                cutoff = current_date - pd.Timedelta(days=365)
            elif time_filter == "3 Ans":
                cutoff = current_date - pd.Timedelta(days=365*3)
            elif time_filter == "5 Ans":
                cutoff = current_date - pd.Timedelta(days=365*5)
            else: # Max
                cutoff = compare_prices.index.min()
                
            compare_prices = compare_prices[compare_prices.index >= cutoff]
            
            # Trouver la date de départ commune (max des dates de début individuelles)
            start_dates = compare_prices.apply(lambda x: x.first_valid_index())
            common_start_date = start_dates.max()
            
            if pd.isna(common_start_date):
                st.warning("Aucune période commune trouvée entre ces actifs pour la période sélectionnée.")
            else:
                # Filtrer depuis la date de départ commune
                compare_prices_common = compare_prices[compare_prices.index >= common_start_date]
                
                # Rebaser en base 100
                first_prices = compare_prices_common.iloc[0]
                base_100_prices = (compare_prices_common / first_prices) * 100
                
                fig = go.Figure()
                for aid in valid_ids:
                    name_disp = [k for k, v in asset_options.items() if v == aid][0]
                    fig.add_trace(go.Scatter(
                        x=base_100_prices.index,
                        y=base_100_prices[aid],
                        mode='lines',
                        name=name_disp,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    hovermode="x unified",
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title="Base 100",
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Petit tableau récapitulatif des performances
                st.markdown(f"**Performance depuis le point commun ({common_start_date.strftime('%d/%m/%Y')})**")
                perf_series = (base_100_prices.iloc[-1] - 100)
                perf_df = perf_series.reset_index()
                perf_df.columns = ['asset_id', 'Performance']
                id_to_name = {v: k for k, v in asset_options.items()}
                perf_df['Actif'] = perf_df['asset_id'].map(id_to_name)
                perf_df = perf_df[['Actif', 'Performance']].sort_values(by='Performance', ascending=False)
                
                st.dataframe(
                    perf_df.style.format({'Performance': "{:.2f} %"}),
                    use_container_width=True,
                    hide_index=True
                )
# ==========================================
# ONGLET 2 : FICHE DETAILLEE ACTIF
# ==========================================
with tab_detail:
    st.header("Vue Détaillée Actif")

    selected_asset_name = st.selectbox(
        "Recherchez un actif par nom ou ticker :",
        options=sorted(asset_options.keys()),
        key="detail_asset_select"
    )

    if not selected_asset_name:
        st.stop()

    selected_asset_id = asset_options[selected_asset_name]
    asset_info = assets_df[assets_df['asset_id'] == selected_asset_id].iloc[0]

    st.divider()

    # --- 1. Fiche d'Identité ---
    st.subheader("Fiche d'Identité")

    is_index = asset_info.get('asset_type') == 'Indice' or pd.notna(asset_info.get('issuer'))

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
    asset_prices = asset_prices[asset_prices > 0]
    if len(asset_prices) == 0:
        st.warning("Aucun historique de prix valide (non nul) disponible pour cet actif.")
        st.stop()

    daily_returns = asset_prices.pct_change(fill_method=None).dropna()
    current_date = asset_prices.index.max()
    current_price = asset_prices.iloc[-1]
    start_of_year = pd.Timestamp(year=current_date.year, month=1, day=1)

    def get_price_at(date_target):
        available_dates = asset_prices[asset_prices.index <= date_target]
        if len(available_dates) == 0:
            return np.nan
        return available_dates.iloc[-1]

    def get_perf(days=None, ytd=False):
        if ytd:
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

    # --- Résolution du Benchmark (pour le Bêta et le Graphique) ---
    benchmark_mapping = {
        'FRANCE': 'CAC Index',
        'GERMANY': 'DAX Index', 'ALLEMAGNE': 'DAX Index',
        'US': 'SPX Index', 'USA': 'SPX Index', 'UNITED STATES': 'SPX Index', 'AMÉRIQUE DU NORD': 'SPX Index',
        'ÉTATS-UNIS': 'SPX Index', 'ÉTATS UNIS': 'SPX Index', 'ETATS-UNIS': 'SPX Index', 'ETATS UNIS': 'SPX Index',
        'BRITAIN': 'UKX Index', 'ROYAUME-UNI': 'UKX Index',
        'SWITZERLAND': 'SMI Index', 'SUISSE': 'SMI Index',
        'JAPAN': 'NKY Index', 'JAPON': 'NKY Index',
        'CHINE': 'SHSZ300 INDEX', 'CHINA': 'SHSZ300 INDEX',
        'HONG KONG': 'HSI Index', 'MACAU': 'HSI Index',
        'EUROPE': 'SX5E Index', 'EURO ZONE': 'SX5E Index', 'ZONE EURO': 'SX5E Index',
        'ITALY': 'SX5E Index', 'ITALIE': 'SX5E Index',
        'SPAIN': 'SX5E Index', 'ESPAGNE': 'SX5E Index',
        'PORTUGAL': 'SX5E Index', 'MALTA': 'SX5E Index',
        'BELGIQUE': 'SX5E Index', 'BELGIUM': 'SX5E Index',
        'NETHERLANDS': 'SX5E Index', 'PAYS-BAS': 'SX5E Index',
        'LUXEMBOURG': 'SX5E Index',
        'SWEDEN': 'SX5E Index', 'SUÈDE': 'SX5E Index', 'SUEDE': 'SX5E Index',
        'DENMARK': 'SX5E Index', 'DANEMARK': 'SX5E Index',
        'NORWAY': 'SX5E Index', 'NORVÈGE': 'SX5E Index', 'NORVEGE': 'SX5E Index',
        'FINLAND': 'SX5E Index', 'FINLANDE': 'SX5E Index',
        'FAROE ISLANDS': 'SX5E Index', 'ÎLES FÉROÉ': 'SX5E Index', 'ILES FEROE': 'SX5E Index',
        'AUSTRIA': 'SX5E Index', 'AUTRICHE': 'SX5E Index',
        'CZECH': 'SX5E Index', 'RÉPUBLIQUE TCHÈQUE': 'SX5E Index', 'REPUBLIQUE TCHEQUE': 'SX5E Index',
        'POLAND': 'SX5E Index', 'POLOGNE': 'SX5E Index',
        'HUNGARY': 'SX5E Index', 'HONGRIE': 'SX5E Index',
        'IRELAND': 'SX5E Index', 'IRLANDE': 'SX5E Index',
        'MONDE': 'MSCI WORLD Index', 
        'WORLD': 'MSCI WORLD Index', 
        'MONDE - MARCHÉS ÉMERGENTS': 'MSCI WORLD Index',
        'BERMUDA': 'MSCI WORLD Index', 
        'TRANSATLANTIC': 'MSCI WORLD Index', 
        'TRANSATLANTIQUE': 'MSCI WORLD Index',
        'EUROPE - US - JAPON': 'MSCI WORLD Index', 'EUROPE - ÉTATS-UNIS - JAPON': 'MSCI WORLD Index', 'EUROPE - ETATS-UNIS - JAPON': 'MSCI WORLD Index',
        'ASIE - US': 'MSCI WORLD Index', 'ASIE - ÉTATS-UNIS': 'MSCI WORLD Index', 'ASIE - ETATS-UNIS': 'MSCI WORLD Index',
        'EURASIE': 'MSCI WORLD Index',
        'EURO-ASIE': 'MSCI WORLD Index',
        'EUROPE - ASIE': 'MSCI WORLD Index',
        'CHINE - EUROPE': 'MSCI WORLD Index',
        'ASIE': 'MSCI AC ASIA PACIFIC Index', 
        'AUSTRALIA': 'MSCI AC ASIA PACIFIC Index',
        'MARCHÉS ÉMERGENTS': 'MSCI EM Index', 'MARCHES EMERGENTS': 'MSCI EM Index',
        'BRAZIL': 'MSCI EM Index', 'BRÉSIL': 'MSCI EM Index', 'BRESIL': 'MSCI EM Index',
        'CHILE': 'MSCI EM Index', 'CHILI': 'MSCI EM Index',
        'JORDAN': 'MSCI EM Index', 
        'KAZAKHSTAN': 'MSCI EM Index', 
        'RUSSIA': 'MSCI EM Index', 'RUSSIE': 'MSCI EM Index'
    }

    asset_country = str(asset_info.get('country', '')).strip().upper()
    benchmark_ticker = benchmark_mapping.get(asset_country)
    benchmark_asset_id = None

    if benchmark_ticker:
        bench_match = assets_df[assets_df['ticker_bloomberg'] == benchmark_ticker]
        if not bench_match.empty:
            benchmark_asset_id = bench_match.iloc[0]['asset_id']

    def get_beta(days=365):
        if not benchmark_asset_id or benchmark_asset_id not in prices_pivot.columns:
            return np.nan
        bench_prices = prices_pivot[benchmark_asset_id].dropna()
        bench_prices = bench_prices[bench_prices > 0]
        if len(bench_prices) == 0: return np.nan
        bench_returns = bench_prices.pct_change(fill_method=None).dropna()
        
        # Aligner les dates
        cutoff_date = current_date - pd.Timedelta(days=days)
        sub_asset = daily_returns[daily_returns.index >= cutoff_date]
        sub_bench = bench_returns[bench_returns.index >= cutoff_date]
        
        common_dates = sub_asset.index.intersection(sub_bench.index)
        if len(common_dates) < 20: # Il faut un minimum de jours
            return np.nan
            
        aligned_asset = sub_asset.loc[common_dates]
        aligned_bench = sub_bench.loc[common_dates]
        
        cov = aligned_asset.cov(aligned_bench)
        var = aligned_bench.var()
        if var == 0: return np.nan
        return cov / var

    # --- 2. Tableaux Perf / Vol / Bêta ---
    st.subheader(f"Performances et Risques (Dernier cours : {current_price:.2f})")

    perf_data = {
        "5 Jours": get_perf(days=5),
        "1 Mois": get_perf(days=30),
        "3 Mois": get_perf(days=90),
        "YTD": get_perf(ytd=True),
    }

    vol_data = {
        "1 Mois": get_vol(days=30),
        "3 Mois": get_vol(days=90),
        "YTD": get_vol(ytd=True),
        "1 An": get_vol(days=365),
    }
    
    beta_data = {
        "1 An": get_beta(days=365),
        "3 Ans": get_beta(days=365*3),
    }

    col_perf, col_vol, col_beta = st.columns(3)

    with col_perf:
        st.markdown("**Performances**")
        perf_df = pd.DataFrame([perf_data]).T
        perf_df.columns = ["Performance"]
        st.dataframe(perf_df.style.format("{:.2f} %"), use_container_width=True)

    with col_vol:
        st.markdown("**Volatilité Annualisée**")
        vol_df = pd.DataFrame([vol_data]).T
        vol_df.columns = ["Volatilité"]
        st.dataframe(vol_df.style.format("{:.2f} %"), use_container_width=True)
        
    with col_beta:
        bench_disp = benchmark_ticker if benchmark_ticker else "N/A"
        st.markdown(f"**Bêta (vs {bench_disp})**")
        beta_df = pd.DataFrame([beta_data]).T
        beta_df.columns = ["Coefficient"]
        # Streamlit style for handling NaN gracefully
        st.dataframe(beta_df.style.format(na_rep="N/A", formatter="{:.2f}"), use_container_width=True)

    st.divider()

    # --- 3. Graphique Technique ---
    st.subheader("Cours")

    show_benchmark = False
    if benchmark_asset_id and benchmark_asset_id in prices_pivot.columns:
        show_benchmark = st.checkbox(f"Afficher la comparaison avec le Benchmark {benchmark_ticker} (Rebasé sur le prix de l'actif)", value=True)

    fig = go.Figure()

    if show_benchmark:
        bench_prices = prices_pivot[benchmark_asset_id].dropna()
        bench_prices = bench_prices[bench_prices > 0]
        
        asset_norm = asset_prices
        common_dates = asset_prices.index.intersection(bench_prices.index)
        if len(common_dates) > 0:
            first_common = common_dates[0]
            bench_sub = bench_prices[bench_prices.index >= first_common]
            bench_norm = (bench_sub / bench_prices.loc[first_common]) * asset_prices.loc[first_common]
            
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

    subtypes = sorted([str(x) for x in assets_df['asset_subtype'].dropna().unique()])
    sectors = sorted([str(x) for x in assets_df['sector'].dropna().unique()])
    countries = sorted([str(x) for x in assets_df['country'].dropna().unique()])

    col_f1_c, col_f2_c, col_f3_c = st.columns(3)
    with col_f1_c:
        filter_subtype = st.multiselect("Filtre Sous Type", options=subtypes, key="corr_subtype")
    with col_f2_c:
        filter_sector = st.multiselect("Filtre Secteur", options=sectors, key="corr_sector")
    with col_f3_c:
        filter_country = st.multiselect("Filtre Pays", options=countries, key="corr_country")

    with st.spinner("Calcul des corrélations en cours..."):
        one_year_ago = current_date - pd.Timedelta(days=365)
        prices_1y_c = prices_pivot[prices_pivot.index >= one_year_ago]
        
        mask = pd.Series(True, index=assets_df.index)
        if len(filter_subtype) > 0:
            mask = mask & (assets_df['asset_subtype'].isin(filter_subtype))
        if len(filter_sector) > 0:
            mask = mask & (assets_df['sector'].isin(filter_sector))
        if len(filter_country) > 0:
            mask = mask & (assets_df['country'].isin(filter_country))
            
        valid_ids = assets_df[mask]['asset_id'].tolist()
        
        if selected_asset_id not in valid_ids:
            valid_ids.append(selected_asset_id)
        valid_ids = [vid for vid in valid_ids if vid in prices_1y_c.columns]
        prices_1y_c = prices_1y_c[valid_ids]
        
        returns_1y_c = prices_1y_c.pct_change(fill_method=None).dropna(how='all')
        if selected_asset_id in returns_1y_c.columns:
            corr_series = returns_1y_c.corrwith(returns_1y_c[selected_asset_id]).dropna()
            corr_series = corr_series.drop(index=selected_asset_id, errors='ignore')
            
            if len(corr_series) > 0:
                id_to_name = dict(zip(assets_df['asset_id'], assets_df['display_name']))
                corr_df = corr_series.reset_index()
                corr_df.columns = ['asset_id', 'Correlation']
                corr_df['Nom de l\'actif'] = corr_df['asset_id'].map(id_to_name)
                
                top_10 = corr_df.nlargest(10, 'Correlation')[['Nom de l\'actif', 'Correlation']]
                bottom_10 = corr_df.nsmallest(10, 'Correlation')[['Nom de l\'actif', 'Correlation']]
                
                col_top, col_bottom = st.columns(2)
                with col_top:
                    st.markdown("**Les plus corrélés**")
                    st.dataframe(top_10.style.format({'Correlation': "{:.4f}"}), use_container_width=True, hide_index=True)
                with col_bottom:
                    st.markdown("**Les moins corrélés**")
                    st.dataframe(bottom_10.style.format({'Correlation': "{:.4f}"}), use_container_width=True, hide_index=True)
            else:
                st.info("Pas assez de données pour calculer les corrélations sur cet univers.")
        else:
            st.warning("L'actif sélectionné n'a pas de données sur la dernière année.")

