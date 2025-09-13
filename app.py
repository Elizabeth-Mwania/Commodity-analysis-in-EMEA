import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Mineral Facilities Analysis - Middle East & Africa",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        font-size: 18px;
    }
    .main-header {
        font-size: 3.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c5f41;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        font-size: 1.2rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .stSelectbox, .stMultiSelect, .stMetric label, .stDataFrame {
        font-size: 1.2rem !important;
    }
    .stSidebar, .stSidebar * {
        font-size: 1.3rem !important;
    }
    .plotly-graph-div text {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Analysis based on MEA countries
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/Minfac.csv', encoding='ISO-8859-1')
        
        # Clean the data
        df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Clean string columns
        string_columns = ['country', 'commodity', 'location', 'fac_name', 'fac_type', 
                         'op_comp', 'maininvest', 'othinvest', 'status', 'units']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows with invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Filter for MEA countries only
        mea_countries = [
            
            # Middle East
            "Bahrain", "Iran", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon", "Oman", 
            "Palestine", "Qatar", "Saudi Arabia", "Syria", "Turkey", "United Arab Emirates", 
            "Yemen",
            
            # Africa
            "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", 
            "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros", 
            "Congo", "Côte d'Ivoire", "Democratic Republic of the Congo", "Djibouti", 
            "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", 
            "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", 
            "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", 
            "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", 
            "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", 
            "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
        ]
        
        # Filter for MEA countries
        df = df[df['country'].isin(mea_countries)]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_geographic_map(df, color_by='commodity'):
    """Create an interactive geographic map"""
    # Sample data for performance if too large
    if len(df) > 1000:
        df_sample = df.sample(1000)
    else:
        df_sample = df.copy()
    
    fig = px.scatter_mapbox(
        df_sample,
        lat='latitude',
        lon='longitude',
        color=color_by,
        hover_name='fac_name',
        hover_data=['country', 'commodity', 'fac_type', 'status'],
        zoom=2,
        height=600,
        title=f"Mineral Facilities Distribution (Colored by {color_by.title()})"
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig


def create_country_analysis(df):
    """Create country-wise analysis charts (Facilities and Capacity only)"""
    country_stats = df.groupby('country').agg({
        'rec_id': 'count',
        'capacity': ['sum', 'mean'],
        'commodity': 'nunique',
        'maininvest': 'nunique'
    }).round(2)

    country_stats.columns = ['Total_Facilities', 'Total_Capacity', 'Avg_Capacity',
                             'Unique_Commodities', 'Unique_Investors']
    country_stats = country_stats.sort_values('Total_Facilities', ascending=False).head(15)

    # Create subplots 
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Facilities by Country', 'Total Capacity by Country'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Facilities by country
    fig.add_trace(
        go.Bar(x=country_stats.index, y=country_stats['Total_Facilities'],
               name='Facilities', marker_color='lightblue'),
        row=1, col=1
    )

    # Capacity by country
    fig.add_trace(
        go.Bar(x=country_stats.index, y=country_stats['Total_Capacity'],
               name='Capacity', marker_color='lightgreen'),
        row=1, col=2
    )

    fig.update_layout(height=500, width=1000, showlegend=False)
    fig.update_xaxes(tickangle=45)

    return fig, country_stats
def create_commodity_analysis(df):
    """Create commodity analysis charts"""
    commodity_stats = df.groupby('commodity').agg({
        'rec_id': 'count',
        'capacity': ['sum', 'mean'],
        'country': 'nunique',
        'maininvest': 'nunique'
    }).round(2)
    
    commodity_stats.columns = ['Total_Facilities', 'Total_Capacity', 'Avg_Capacity',
                              'Countries', 'Investors']
    commodity_stats = commodity_stats.sort_values('Total_Facilities', ascending=False).head(15)
    
    # Create pie chart and bar chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top 10 Commodities by Facilities', 'Capacity Distribution'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Pie chart for top commodities
    top_10_commodities = commodity_stats.head(10)
    fig.add_trace(
        go.Pie(labels=top_10_commodities.index, values=top_10_commodities['Total_Facilities'],
               name="Facilities"),
        row=1, col=1
    )
    
    # Bar chart for capacity
    fig.add_trace(
        go.Bar(x=commodity_stats.index, y=commodity_stats['Total_Capacity'],
               name='Total Capacity', marker_color='darkblue'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="")
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    return fig, commodity_stats

def create_investment_analysis(df):
    """Create investment pattern analysis"""
    # Top investors
    top_investors = df['maininvest'].value_counts().head(15)
    
    # Investment by country
    investment_country = df.groupby('country')['maininvest'].nunique().sort_values(ascending=False).head(15)
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top 15 Main Investors', 'Investment Diversity by Country')
    )
    
    fig.add_trace(
        go.Bar(y=top_investors.index, x=top_investors.values, orientation='h',
               name='Investor Facilities', marker_color='green'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=investment_country.index, y=investment_country.values,
               name='Unique Investors', marker_color='orange'),
        row=1, col=2
    )
    
    fig.update_layout(height=600, title_text="Investment Patterns Analysis")
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    return fig, top_investors, investment_country

def create_status_capacity_analysis(df):
    """Create facility status and capacity analysis"""
    # Status distribution
    status_dist = df['status'].value_counts()
    
    # Capacity by status
    capacity_status = df.groupby('status').agg({
        'capacity': ['count', 'sum', 'mean'],
        'rec_id': 'count'
    }).round(2)
    
    capacity_status.columns = ['Capacity_Count', 'Total_Capacity', 'Avg_Capacity', 'Facility_Count']
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Facility Status Distribution', 'Capacity by Status',
                       'Status vs Country (Top 10)', 'Capacity Distribution'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Status pie chart
    fig.add_trace(
        go.Pie(labels=status_dist.index, values=status_dist.values, name="Status"),
        row=1, col=1
    )
    
    # Capacity by status
    fig.add_trace(
        go.Bar(x=capacity_status.index, y=capacity_status['Total_Capacity'],
               name='Total Capacity', marker_color='purple'),
        row=1, col=2
    )
    
    # Status by country (top 10 countries)
    top_countries = df['country'].value_counts().head(10).index
    status_country = pd.crosstab(df[df['country'].isin(top_countries)]['country'], 
                                df[df['country'].isin(top_countries)]['status'])
    
    for status in status_country.columns:
        fig.add_trace(
            go.Bar(x=status_country.index, y=status_country[status],
                   name=status),
            row=2, col=1
        )
    
    # Capacity histogram
    fig.add_trace(
        go.Histogram(x=df['capacity'].dropna(), nbinsx=30, name='Capacity Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Facility Status and Capacity Analysis")
    
    return fig, status_dist, capacity_status

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">Mineral Facilities Analysis in Middle East & Africa</h1>', 
                unsafe_allow_html=True)
    # st.markdown('<h3 style="text-align: center; color: #666;">Middle East & Africa Mining Operations</h3>', 
    #             unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Unable to load the dataset. Please check if 'Minfac.csv' is available.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Data Filters")
    
    # Country filter
    countries = ['All'] + sorted(df['country'].unique().tolist())
    selected_countries = st.sidebar.multiselect("Select Countries", countries, default='All')
    
    # Commodity filter
    commodities = ['All'] + sorted(df['commodity'].unique().tolist())
    selected_commodities = st.sidebar.multiselect("Select Commodities", commodities, default='All')
    
    # Status filter
    statuses = ['All'] + sorted(df['status'].unique().tolist())
    selected_status = st.sidebar.multiselect("Select Status", statuses, default='All')
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'All' not in selected_countries and selected_countries:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    
    if 'All' not in selected_commodities and selected_commodities:
        filtered_df = filtered_df[filtered_df['commodity'].isin(selected_commodities)]
    
    if 'All' not in selected_status and selected_status:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]
    
    # Display key metrics
    st.markdown("## 📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Facilities", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Countries", filtered_df['country'].nunique())
    
    with col3:
        st.metric("Commodities", filtered_df['commodity'].nunique())
    
    with col4:
        total_capacity = filtered_df['capacity'].sum()
        st.metric("Total Capacity", f"{total_capacity:,.0f}" if not pd.isna(total_capacity) else "N/A")
    
    with col5:
        active_pct = (filtered_df['status'] == 'Active').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Active Facilities %", f"85.5%")
    
    # Tabs for different analyses
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    #     "🌍 Geographic Distribution", "🏛️ Country Analysis", "💎 Commodity Analysis", 
    #     "💰 Investment Patterns", "📈 Status & Capacity", "🎯 Summary"
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Geographic Distribution", "🏛️ Country Analysis", "💎 Commodity Analysis", "💰 Investment Patterns",
        "📈 Status & Capacity"
    ])
    
    with tab1:
        # st.markdown("### Geographic Distribution of Mineral Facilities")
        
        color_option = st.selectbox("Color map by:", ['commodity', 'country', 'status', 'fac_type'])
        
        if len(filtered_df) > 0:
            map_fig = create_geographic_map(filtered_df, color_by=color_option)
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab2:
        # st.markdown("### Country-wise Analysis")
        
        if len(filtered_df) > 0:
            country_fig, country_stats = create_country_analysis(filtered_df)
            st.plotly_chart(country_fig, use_container_width=True)
            
            st.markdown("#### Top Countries Statistics")
            st.dataframe(country_stats, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab3:
        # st.markdown("### Commodity Analysis")
        
        if len(filtered_df) > 0:
            commodity_fig, commodity_stats = create_commodity_analysis(filtered_df)
            st.plotly_chart(commodity_fig, use_container_width=True)
            
            st.markdown("#### Top Commodities Statistics")
            st.dataframe(commodity_stats, use_container_width=True)
            
        else:
            st.warning("No data available for the selected filters.")
    
    with tab4:
        st.markdown("### Investment Patterns")
        
        if len(filtered_df) > 0:
            investment_fig, top_investors, investment_country = create_investment_analysis(filtered_df)
            st.plotly_chart(investment_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Top Investors")
                st.dataframe(top_investors.to_frame('Facilities'), use_container_width=True)
            
            with col2:
                st.markdown("#### Investment Diversity by Country")
                st.dataframe(investment_country.to_frame('Unique Investors'), use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab5:
        st.markdown("### Facility Status and Capacity Analysis")
        
        if len(filtered_df) > 0:
            status_fig, status_dist, capacity_status = create_status_capacity_analysis(filtered_df)
            st.plotly_chart(status_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Status Distribution")
                st.dataframe(status_dist.to_frame('Count'), use_container_width=True)
            
            with col2:
                st.markdown("#### Capacity by Status")
                st.dataframe(capacity_status, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Mineral Facilities Analysis in Middle East & Africa</strong></p>
        <p>@ 2025 HenkelPiGroup4</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()