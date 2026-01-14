import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Expansion Intel | JV.DATA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to align with a "Dark/Professional" aesthetic
st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    h1, h2, h3 {font-family: 'Helvetica Neue', sans-serif; font-weight: 600;}
    .highlight {color: #4CAF50; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Load with latin1 to handle Spanish accents correctly
    try:
        df = pd.read_csv("tasa_autorizacion_riesgo_predicho.csv", encoding='latin1')
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'tasa_autorizacion_riesgo_predicho.csv' is in the directory.")
        return pd.DataFrame()

    # Clean string columns to ensure proper UTF-8 display if mixed
    text_cols = ['nom_ent', 'nom_mun', 'nom_loc', 'nombre']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x.encode('latin1').decode('utf-8', 'ignore') if isinstance(x, str) else x)

    # Rename columns for English UI (Professional standardization)
    df.rename(columns={
        'nom_ent': 'State',
        'nom_mun': 'Municipality',
        'nom_loc': 'Locality',
        'nombre': 'Neighborhood', # Colonia
        'prob_ALTO RIESGO': 'Prob_High_Risk',
        'prob_BAJO RIESGO': 'Prob_Low_Risk',
        'prob_MEDIO RIESGO': 'Prob_Medium_Risk',
        'prob_ALTA': 'Prob_High_Auth',
        'prob_BAJA': 'Prob_Low_Auth',
        'riesgo_clasifacion_predicha': 'Risk_Class',
        'tasa_clasificacion_predicha': 'Auth_Rate_Class'
    }, inplace=True)

    # Feature Engineering: Opportunity Score
    # Score = (Low Risk Prob * 0.6) + (High Auth Prob * 0.4)
    # This helps rank neighborhoods by "Best Business Opportunity"
    df['Opportunity_Score'] = (df['Prob_Low_Risk'] * 0.6) + (df['Prob_High_Auth'] * 0.4)
    
    return df

df = load_and_clean_data()

if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    # Cascading Filters
    selected_state = st.selectbox("Select State", options=sorted(df['State'].unique()))
    
    munis_in_state = sorted(df[df['State'] == selected_state]['Municipality'].unique())
    selected_muni = st.selectbox("Select Municipality", options=["All Municipalities"] + munis_in_state)
    
    st.markdown("---")
    st.markdown("### Threshold Settings")
    min_low_risk = st.slider("Min. Low Risk Probability", 0.0, 1.0, 0.50, help="Filter neighborhoods with at least this probability of being Low Risk.")
    min_high_auth = st.slider("Min. High Authorization Prob", 0.0, 1.0, 0.40, help="Filter neighborhoods with at least this probability of High Authorization Rates.")
    
    st.info("Built by **JV.DATA**\nUsing Dual-Model Classification")

# -----------------------------------------------------------------------------
# 4. FILTERING LOGIC
# -----------------------------------------------------------------------------
filtered_df = df[df['State'] == selected_state].copy()
if selected_muni != "All Municipalities":
    filtered_df = filtered_df[filtered_df['Municipality'] == selected_muni]

# Apply Thresholds for the "Opportunity" view
opportunity_df = filtered_df[
    (filtered_df['Prob_Low_Risk'] >= min_low_risk) & 
    (filtered_df['Prob_High_Auth'] >= min_high_auth)
]

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------

# Header
st.title("Credit Portfolio Expansion Intelligence")
st.markdown(f"**Strategic Analysis for:** {selected_state} {'/ ' + selected_muni if selected_muni != 'All Municipalities' else ''}")

# Top Level KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Neighborhoods", f"{len(filtered_df):,}")
with col2:
    avg_risk_prob = filtered_df['Prob_Low_Risk'].mean()
    st.metric("Avg. Safety Score", f"{avg_risk_prob:.1%}", delta_color="normal", help="Average Probability of Low Risk")
with col3:
    prime_locations = len(opportunity_df)
    st.metric("Prime Targets", f"{prime_locations}", delta=f"{prime_locations/len(filtered_df):.1%} of Total")
with col4:
    avg_auth = filtered_df['Prob_High_Auth'].mean()
    st.metric("Mkt. Penetration Potential", f"{avg_auth:.1%}", help="Average Probability of High Authorization Rate")

# TABS FOR DEPTH
tab1, tab2, tab3 = st.tabs(["Strategy Matrix", "Granular Explorer", "Model Logic"])

# --- TAB 1: STRATEGY MATRIX (Quadrants) ---
with tab1:
    st.markdown("### Risk vs. Growth Matrix")
    st.caption("Identify 'Sweet Spots': Neighborhoods with **Low Default Risk** (X-Axis) and **High Authorization Potential** (Y-Axis).")
    
    # Scatter Plot
    fig_scatter = px.scatter(
        filtered_df,
        x="Prob_Low_Risk",
        y="Prob_High_Auth",
        color="Risk_Class",
        hover_data=["Neighborhood", "Municipality", "Prob_High_Risk"],
        color_discrete_map={"BAJO RIESGO": "#00CC96", "MEDIO RIESGO": "#FFA15A", "ALTO RIESGO": "#EF553B"},
        title="Portfolio Optimization Quadrant",
        labels={"Prob_Low_Risk": "Probability of Low Risk (Safety)", "Prob_High_Auth": "Probability of High Auth Rate (Growth)"}
    )
    
    # Add Quadrant Lines
    fig_scatter.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="white")
    fig_scatter.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="white")
    
    # Annotations for Business Context
    fig_scatter.add_annotation(x=0.85, y=0.9, text="EXPANSION ZONE", showarrow=False, font=dict(color="#00CC96", size=14))
    fig_scatter.add_annotation(x=0.15, y=0.1, text="AVOID ZONE", showarrow=False, font=dict(color="#EF553B", size=14))
    
    fig_scatter.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Top Opportunities List
    st.markdown("### Top 5 Strategic Neighborhoods")
    top_opps = opportunity_df.sort_values(by="Opportunity_Score", ascending=False).head(5)
    
    if not top_opps.empty:
        # Display as columns/cards
        cols = st.columns(5)
        for i, (index, row) in enumerate(top_opps.iterrows()):
            with cols[i]:
                st.info(f"**{row['Neighborhood']}**")
                st.markdown(f"**Safety:** {row['Prob_Low_Risk']:.0%}")
                st.markdown(f"**Auth:** {row['Prob_High_Auth']:.0%}")
                st.progress(int(row['Opportunity_Score']*100))
    else:
        st.warning("No neighborhoods meet the current threshold settings. Try lowering the filters in the sidebar.")

# --- TAB 2: GRANULAR EXPLORER ---
with tab2:
    st.markdown("### Geographic Drill-Down")
    
    col_table, col_dist = st.columns([2, 1])
    
    with col_dist:
        st.markdown("**Risk Distribution**")
        fig_pie = px.pie(filtered_df, names='Risk_Class', hole=0.4, color='Risk_Class',
                         color_discrete_map={"BAJO RIESGO": "#00CC96", "MEDIO RIESGO": "#FFA15A", "ALTO RIESGO": "#EF553B"})
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("**Authorization Potential**")
        # Preparamos los datos correctamente para las nuevas versiones de Pandas
        counts_df = filtered_df['Auth_Rate_Class'].value_counts().reset_index()
        
        # En Pandas 2.0+, las columnas son ['Auth_Rate_Class', 'count']
        # En versiones viejas eran ['index', 'Auth_Rate_Class']
        # Para que sea compatible con ambos, renombramos explícitamente:
        counts_df.columns = ['Class', 'Count']
        
        fig_bar = px.bar(
            counts_df, 
            x='Class', 
            y='Count', 
            labels={'Class': 'Category', 'Count': 'Frequency'},
            color='Class', 
            title=""
        )
        fig_bar.update_layout(showlegend=False, height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)


    with col_table:
        st.markdown("**Detailed Neighborhood Data**")
        
        # Format for display
        display_df = filtered_df[['Neighborhood', 'Municipality', 'Risk_Class', 'Prob_Low_Risk', 'Auth_Rate_Class', 'Prob_High_Auth']].copy()
        
        # Apply color styling to dataframe
        def highlight_risk(val):
            color = '#90EE90' if val == 'BAJO RIESGO' else '#FFB6C1' if val == 'ALTO RIESGO' else '#FFE4B5'
            return f'background-color: {color}; color: black'

        st.dataframe(
            display_df.style.applymap(highlight_risk, subset=['Risk_Class'])
                            .format({'Prob_Low_Risk': '{:.1%}', 'Prob_High_Auth': '{:.1%}'}),
            height=600,
            use_container_width=True
        )

# --- TAB 3: MODEL LOGIC ---
with tab3:
    st.markdown("""
    ### Behind the Intelligence
    This tool leverages two independent Machine Learning models to provide a holistic view of the market.
    
    #### 1. Credit Risk Model (Classification)
    *   **Goal:** Predict likelihood of default based on socio-demographic granular indicators.
    *   **Output:** `Prob_Low_Risk`, `Prob_Medium_Risk`, `Prob_High_Risk`.
    *   **Business Impact:** Used to set credit limits and interest rates dynamically.
    
    #### 2. Authorization Rate Model (Regression/Binning)
    *   **Goal:** Estimate the market penetration and approval likelihood for specific products.
    *   **Output:** `Prob_High_Auth` (Likelihood that >70% of apps are approved).
    *   **Business Impact:** Identifying zones where marketing spend yields high conversion.
    
    #### Data Engineering Note
    Data is processed via ETL pipelines extracting granular metrics at the 'Colonia' level across Mexico, ensuring specific targeting rather than broad municipal generalizations.
    """)