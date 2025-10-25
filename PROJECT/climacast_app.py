import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import io
import base64
from climate_data_fetcher import ClimateDataFetcher
from ai_advisor_new import ClimateAIAdvisor
from ai_chatbot import ClimateAIChatbot
import os

# Set page config
st.set_page_config(
    page_title="ClimaCast - Climate Impact Predictor",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .danger-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .export-btn {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    .export-btn:hover {
        background: linear-gradient(135deg, #218838, #1c9c6e);
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_metrics' not in st.session_state:
    st.session_state.current_metrics = {
        'temperature': {'value': 22.5, 'change': '+2.1%', 'unit': '°C'},
        'rainfall': {'value': 845, 'change': '-12%', 'unit': 'mm'},
        'co2': {'value': 421, 'change': '+3.2%', 'unit': 'ppm'},
        'risk_score': {'value': 7.2, 'level': 'Moderate risk', 'max': 10}
    }
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = {}
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = None
if 'real_data_loaded' not in st.session_state:
    st.session_state.real_data_loaded = False

# Initialize climate data fetcher with real API keys
@st.cache_resource
def initialize_data_fetcher():
    """Initialize the climate data fetcher with API keys"""
    try:
        # Prefer environment variables for secrets; fall back to project defaults if not set
        noaa_token = os.getenv("NOAA_API_TOKEN", "ABwdnxQHhZxjyoBcKfslphYNwuKrEian")
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "de16680967cf7ad68b0e88ab1f6229c1")

        fetcher = ClimateDataFetcher(noaa_token, openweather_api_key)
        return fetcher
    except Exception as e:
        st.error(f"Failed to initialize data fetcher: {e}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🌦️ ClimaCast</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Climate Impact Predictor - Powered by NOAA & OpenWeatherMap</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌍 Configuration")

# Initialize data fetcher
if st.session_state.data_fetcher is None:
    st.session_state.data_fetcher = initialize_data_fetcher()

if st.session_state.data_fetcher is None:
    st.sidebar.error("❌ Failed to initialize climate data services")
    st.stop()

# Region selection with real data support
regions = {
    "New York, USA": {
        "lat": 40.7128, "lon": -74.0060,
        "base_temp": 22.5, "base_rain": 845, "base_co2": 421,
        "temp_trend": 0.5, "rain_trend": -15, "co2_trend": 1.2
    },
    "London, UK": {
        "lat": 51.5074, "lon": -0.1278,
        "base_temp": 18.2, "base_rain": 1200, "base_co2": 415,
        "temp_trend": 0.8, "rain_trend": -8, "co2_trend": 1.5
    },
    "Tokyo, Japan": {
        "lat": 35.6762, "lon": 139.6503,
        "base_temp": 25.8, "base_rain": 680, "base_co2": 425,
        "temp_trend": 1.2, "rain_trend": -5, "co2_trend": 2.1
    },
    "Sydney, Australia": {
        "lat": -33.8688, "lon": 151.2093,
        "base_temp": 28.4, "base_rain": 560, "base_co2": 418,
        "temp_trend": 1.5, "rain_trend": -12, "co2_trend": 1.8
    },
    "Mumbai, India": {
        "lat": 19.0760, "lon": 72.8777,
        "base_temp": 32.1, "base_rain": 920, "base_co2": 432,
        "temp_trend": 1.8, "rain_trend": -18, "co2_trend": 2.5
    },
    "São Paulo, Brazil": {
        "lat": -23.5505, "lon": -46.6333,
        "base_temp": 24.7, "base_rain": 1450, "base_co2": 428,
        "temp_trend": 1.1, "rain_trend": -6, "co2_trend": 1.9
    }
}

selected_region = st.sidebar.selectbox("Select Region", list(regions.keys()))

# API Status Display
if st.session_state.data_fetcher:
    api_status = st.session_state.data_fetcher.get_api_status()
    
    with st.sidebar.expander("📊 API Status", expanded=False):
        # NOAA Status
        noaa_status = api_status["noaa"]
        noaa_icon = "✅" if noaa_status["status"] == "healthy" else "⚠️"
        st.write(f"{noaa_icon} **NOAA API**: {noaa_status['status'].title()}")
        if noaa_status["last_success"]:
            st.write(f"Last success: {noaa_status['last_success'].strftime('%H:%M:%S')}")
        if noaa_status["failures"] > 0:
            st.write(f"Consecutive failures: {noaa_status['failures']}")
        
        # OpenWeatherMap Status
        owm_status = api_status["openweather"]
        owm_icon = "✅" if owm_status["status"] == "healthy" else "⚠️"
        st.write(f"{owm_icon} **OpenWeatherMap**: {owm_status['status'].title()}")
        if owm_status["last_success"]:
            st.write(f"Last success: {owm_status['last_success'].strftime('%H:%M:%S')}")
        if owm_status["failures"] > 0:
            st.write(f"Consecutive failures: {owm_status['failures']}")

# Load real data button
if st.sidebar.button("🔄 Load Real-Time Climate Data"):
    with st.spinner("🌍 Fetching real-time climate data from NOAA and OpenWeatherMap..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Connecting to climate data APIs...")
            progress_bar.progress(20)
            
            # Fetch comprehensive climate data summary
            status_text.text("📈 Analyzing historical climate patterns...")
            progress_bar.progress(50)
            
            climate_summary = st.session_state.data_fetcher.get_climate_data_summary(selected_region)
            
            status_text.text("🧠 Processing climate insights...")
            progress_bar.progress(80)
            
            # Update metrics based on real data
            if "historical_stats" in climate_summary:
                hist_stats = climate_summary["historical_stats"]
                
                # Extract real values
                temp_data = hist_stats.get("temperature", {})
                rain_data = hist_stats.get("rainfall", {})
                co2_data = hist_stats.get("co2", {})
                
                # Calculate percentage changes with proper sign handling
                temp_annual_change = temp_data.get('trend', {}).get('annual_change', 0)
                rain_annual_change = rain_data.get('trend', {}).get('annual_change', 0)
                co2_annual_change = co2_data.get('trend', {}).get('annual_change', 0)
                
                temp_change = f"{temp_annual_change:+.1f}%"
                rain_change = f"{rain_annual_change:+.0f}%"
                co2_change = f"{co2_annual_change:+.1f}%"
                
                # Update risk score
                risk_assessment = climate_summary.get("risk_assessment", {})
                risk_score = risk_assessment.get("overall_risk", 5.0)
                risk_level = risk_assessment.get("risk_level", "Moderate risk")
                
                st.session_state.current_metrics = {
                    'temperature': {
                        'value': temp_data.get('mean', 22.5), 
                        'change': temp_change, 
                        'unit': '°C',
                        'source': 'real-time'
                    },
                    'rainfall': {
                        'value': rain_data.get('mean', 845), 
                        'change': rain_change, 
                        'unit': 'mm',
                        'source': 'real-time'
                    },
                    'co2': {
                        'value': co2_data.get('mean', 421), 
                        'change': co2_change, 
                        'unit': 'ppm',
                        'source': 'real-time'
                    },
                    'risk_score': {
                        'value': risk_score, 
                        'level': risk_level, 
                        'max': 10,
                        'source': 'real-time'
                    }
                }
                
                # Store real data for analysis
                st.session_state.climate_summary = climate_summary
                st.session_state.real_data_loaded = True
                
                status_text.text("✅ Real-time climate data loaded successfully!")
                progress_bar.progress(100)
                
                # Show current weather info if available
                if "current" in climate_summary:
                    current = climate_summary["current"]
                    st.sidebar.success(f"✅ Live data loaded! Current: {current.get('temperature', 'N/A')}°C, {current.get('description', 'N/A')}")
                else:
                    st.sidebar.success("✅ Real-time climate data loaded successfully!")
                
            else:
                st.sidebar.warning("⚠️ Partial data loaded - some APIs may be unavailable")
                
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load real-time data: {e}")
            
        finally:
            # Clean up progress indicators
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

# Display current metrics
def display_metrics():
    metrics = st.session_state.current_metrics
    
    # Show data source indicator
    if any(metric.get('source') == 'real-time' for metric in metrics.values()):
        st.info("🔴 **Live Data**: Metrics below are updated with real-time climate data")
    else:
        st.info("🟡 **Sample Data**: Click 'Load Real-Time Climate Data' in sidebar for live updates")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_metric = metrics['temperature']
        indicator = "🔴" if temp_metric.get('source') == 'real-time' else "🟡"
        st.metric(
            label=f"{indicator} Temperature", 
            value=f"{temp_metric['value']:.1f} {temp_metric['unit']}", 
            delta=temp_metric['change']
        )
    
    with col2:
        rain_metric = metrics['rainfall']
        indicator = "🔴" if rain_metric.get('source') == 'real-time' else "🟡"
        st.metric(
            label=f"{indicator} Rainfall", 
            value=f"{rain_metric['value']:.0f} {rain_metric['unit']}", 
            delta=rain_metric['change']
        )
    
    with col3:
        co2_metric = metrics['co2']
        indicator = "🔴" if co2_metric.get('source') == 'real-time' else "🟡"
        st.metric(
            label=f"{indicator} CO₂ Levels", 
            value=f"{co2_metric['value']:.0f} {co2_metric['unit']}", 
            delta=co2_metric['change']
        )
    
    with col4:
        risk_metric = metrics['risk_score']
        indicator = "🔴" if risk_metric.get('source') == 'real-time' else "🟡"
        st.metric(
            label=f"{indicator} Risk Score", 
            value=f"{risk_metric['value']:.1f}/{risk_metric['max']}", 
            delta=risk_metric['level']
        )

# Display metrics at the top
display_metrics()

# Real data generation functions
def generate_real_forecast_data(variable, start_date, end_date, region_name):
    """Generate forecast data using historical analysis and real APIs"""
    
    # Ensure dates are properly converted to datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    if st.session_state.data_fetcher:
        try:
            # First, get historical analysis for the same date range
            historical_data, hist_unit, hist_color = generate_date_range_historical_data(
                variable, start_date, end_date, region_name
            )
            
            # Try to fetch real forecast data
            forecast_data = st.session_state.data_fetcher.fetch_openweather_forecast_data(region_name, 30)
            
            # Generate date range for forecast
            forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create enhanced forecast based on historical patterns
            if not historical_data.empty and variable in historical_data.columns:
                # Calculate historical statistics for the same date range
                hist_mean = historical_data[variable].mean()
                hist_std = historical_data[variable].std()
                
                # Calculate seasonal trend from historical data
                if len(historical_data) > 1:
                    # Ensure date column is datetime
                    if 'date' in historical_data.columns:
                        try:
                            historical_data['date'] = pd.to_datetime(historical_data['date'])
                            # Group by day of year to find seasonal patterns
                            historical_data['day_of_year'] = historical_data['date'].dt.dayofyear
                            seasonal_pattern = historical_data.groupby('day_of_year')[variable].mean()
                        except Exception as e:
                            print(f"Error processing historical data dates: {e}")
                            # If no date column, create a simple pattern
                            seasonal_pattern = pd.Series([hist_mean] * 365, index=range(1, 366))
                    else:
                        # If no date column, create a simple pattern
                        seasonal_pattern = pd.Series([hist_mean] * 365, index=range(1, 366))
                    
                    # Generate forecast based on historical patterns with trend
                    forecast_values = []
                    for date in forecast_dates:
                        day_of_year = date.timetuple().tm_yday
                        
                        if day_of_year in seasonal_pattern.index:
                            # Use historical seasonal pattern
                            base_value = seasonal_pattern[day_of_year]
                        else:
                            # Use overall historical mean
                            base_value = hist_mean
                        
                        # Add some trend and variation
                        trend_factor = 1.02 if variable == 'temperature' or variable == 'co2' else 0.98  # Slight warming/CO2 increase, rainfall decrease
                        forecast_value = base_value * trend_factor + np.random.normal(0, hist_std * 0.1)
                        forecast_values.append(forecast_value)
                    
                    enhanced_forecast = pd.DataFrame({
                        'date': forecast_dates,
                        variable: forecast_values
                    })
                    
                    # Add confidence intervals based on historical variability
                    confidence_margin = hist_std * 0.2
                    enhanced_forecast[f'{variable}_confidence_lower'] = enhanced_forecast[variable] - confidence_margin
                    enhanced_forecast[f'{variable}_confidence_upper'] = enhanced_forecast[variable] + confidence_margin
                    
                    # Store historical analysis for later use
                    st.session_state[f'historical_analysis_{variable}'] = {
                        'data': historical_data,
                        'mean': hist_mean,
                        'std': hist_std,
                        'seasonal_pattern': seasonal_pattern
                    }
                    
                    unit_map = {"temperature": "°C", "rainfall": "mm", "co2": "ppm"}
                    color_map = {"temperature": "red", "rainfall": "blue", "co2": "gray"}
                    
                    return enhanced_forecast, unit_map[variable], color_map[variable]
            
            # Fallback to basic forecast if historical data is not available
            if not forecast_data.empty:
                # Filter by date range
                forecast_data = forecast_data[
                    (forecast_data['date'] >= pd.to_datetime(start_date)) & 
                    (forecast_data['date'] <= pd.to_datetime(end_date))
                ]
                
                if not forecast_data.empty:
                    # Add confidence intervals (estimated)
                    forecast_data[f'{variable}_confidence_lower'] = forecast_data[variable] * 0.95
                    forecast_data[f'{variable}_confidence_upper'] = forecast_data[variable] * 1.05
                    
                    unit_map = {"temperature": "°C", "rainfall": "mm", "co2": "ppm"}
                    color_map = {"temperature": "red", "rainfall": "blue", "co2": "gray"}
                    
                    return forecast_data, unit_map[variable], color_map[variable]
        
        except Exception as e:
            st.warning(f"Could not fetch real forecast data: {e}")
    
    # Final fallback: generate basic synthetic forecast data
    try:
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate basic forecast data based on variable type
        if variable == "temperature":
            base_temp = 20  # Base temperature
            values = [base_temp + np.random.normal(0, 5) for _ in range(len(forecast_dates))]
        elif variable == "rainfall":
            base_rain = 5  # Base rainfall
            values = [max(0, base_rain + np.random.normal(0, 3)) for _ in range(len(forecast_dates))]
        elif variable == "co2":
            base_co2 = 400  # Base CO2
            values = [base_co2 + np.random.normal(0, 10) for _ in range(len(forecast_dates))]
        else:
            values = [0 for _ in range(len(forecast_dates))]
        
        forecast_data = pd.DataFrame({
            'date': forecast_dates,
            variable: values
        })
        
        unit_map = {"temperature": "°C", "rainfall": "mm", "co2": "ppm"}
        color_map = {"temperature": "red", "rainfall": "blue", "co2": "gray"}
        
        return forecast_data, unit_map[variable], color_map[variable]
        
    except Exception as e:
        print(f"Error generating fallback forecast: {e}")
        return pd.DataFrame(), "", ""

def generate_real_historical_data(variable, start_year, end_year, region_name):
    """Generate historical data using real APIs"""
    
    if st.session_state.data_fetcher:
        try:
            # Fetch real historical data from NOAA
            historical_data = st.session_state.data_fetcher.fetch_noaa_historical_data(region_name, start_year, end_year)
            
            if not historical_data.empty:
                unit_map = {"temperature": "°C", "rainfall": "mm", "co2": "ppm"}
                color_map = {"temperature": "red", "rainfall": "blue", "co2": "gray"}
                
                return historical_data, unit_map[variable], color_map[variable]
            else:
                print(f"No historical data found for {region_name}")
                return pd.DataFrame(), "", ""
        
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame(), "", ""
    
    return pd.DataFrame(), "", ""

def generate_date_range_historical_data(variable, start_date, end_date, region_name):
    """Generate historical data for specific date ranges across past 4 years"""
    
    if st.session_state.data_fetcher:
        try:
            # Convert string dates to datetime objects if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Ensure we have proper datetime objects
            if not hasattr(start_date, 'month') or not hasattr(end_date, 'month'):
                print(f"Warning: Invalid date objects - start_date: {type(start_date)}, end_date: {type(end_date)}")
                return pd.DataFrame(), "", ""
            
            # Extract month and day from the date range
            start_month = start_date.month
            start_day = start_date.day
            end_month = end_date.month
            end_day = end_date.day
            
            # Fetch historical data for the same date range across past 4 years
            historical_data = st.session_state.data_fetcher.fetch_historical_data_for_date_range(
                region_name, start_month, start_day, end_month, end_day, years_back=4
            )
            
            if not historical_data.empty:
                unit_map = {"temperature": "°C", "rainfall": "mm", "co2": "ppm"}
                color_map = {"temperature": "red", "rainfall": "blue", "co2": "gray"}
                
                return historical_data, unit_map[variable], color_map[variable]
        
        except Exception as e:
            st.warning(f"Could not fetch date range historical data: {e}")
    
    return pd.DataFrame(), "", ""

def create_forecast_chart(df, variable, unit, color):
    """Create forecast chart"""
    fig = go.Figure()
    
    # Add main forecast line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[variable],
        mode='lines',
        name=f'Forecast ({variable})',
        line=dict(color=color, width=3),
        hovertemplate=f'%{{y:.1f}} {unit}<extra></extra>'
    ))
    
    # Add confidence intervals
    if f'{variable}_confidence_lower' in df.columns and f'{variable}_confidence_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[f'{variable}_confidence_upper'],
            mode='lines',
            name='Upper Confidence',
            line=dict(color=color, width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[f'{variable}_confidence_lower'],
            mode='lines',
            name='Lower Confidence',
            line=dict(color=color, width=1, dash='dash'),
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'{variable.title()} Forecast for {selected_region}',
        xaxis_title='Date',
        yaxis_title=f'{variable.title()} ({unit})',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_historical_chart(df, variable, unit, color):
    """Create historical chart"""
    fig = go.Figure()
    
    rgb_color = "255, 0, 0" if color == "red" else "0, 0, 255" if color == "blue" else "128, 128, 128"

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[variable],
        mode='lines',
        name=f'Historical {variable}',
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({rgb_color}, 0.3)',
        hovertemplate=f'%{{y:.1f}} {unit}<extra></extra>'
    ))
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(range(len(df)), df[variable], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=p(range(len(df))),
            mode='lines',
            name='Trend Line',
            line=dict(color='orange', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f'Historical {variable.title()} Analysis for {selected_region}',
        xaxis_title='Date',
        yaxis_title=f'{variable.title()} ({unit})',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_enhanced_historical_chart(df, variable, unit, color):
    """Create enhanced historical chart with year-over-year comparison"""
    fig = go.Figure()
    
    if 'year' in df.columns:
        # Group by year and create separate traces
        years = sorted(df['year'].unique())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Different colors for different years
        
        for i, year in enumerate(years):
            year_data = df[df['year'] == year].copy()
            year_data = year_data.sort_values('date')
            
            color_idx = i % len(colors)
            
            fig.add_trace(go.Scatter(
                x=year_data['date'],
                y=year_data[variable],
                mode='lines+markers',
                name=f'{year}',
                line=dict(color=colors[color_idx], width=2),
                marker=dict(size=4),
                hovertemplate=f'Year: {year}<br>Date: %{{x}}<br>Value: %{{y:.1f}} {unit}<extra></extra>'
            ))
        
        # Add overall trend line
        if len(df) > 1:
            # Calculate trend using all data points
            df_sorted = df.sort_values('date')
            z = np.polyfit(range(len(df_sorted)), df_sorted[variable], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df_sorted['date'],
                y=p(range(len(df_sorted))),
                mode='lines',
                name='Overall Trend',
                line=dict(color='black', width=3, dash='dash'),
                hoverinfo='skip'
            ))
        
        # Add average line
        avg_value = df[variable].mean()
        fig.add_hline(
            y=avg_value,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"4-Year Average: {avg_value:.1f} {unit}",
            annotation_position="top right"
        )
        
        title = f'Historical {variable.title()} Analysis (Past 4 Years) - {selected_region}'
    else:
        # Fallback to regular chart if no year column
        rgb_color = "255, 0, 0" if color == "red" else "0, 0, 255" if color == "blue" else "128, 128, 128"
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[variable],
            mode='lines',
            name=f'Historical {variable}',
            line=dict(color=color, width=3),
            fill='tozeroy',
            fillcolor=f'rgba({rgb_color}, 0.3)',
            hovertemplate=f'%{{y:.1f}} {unit}<extra></extra>'
        ))
        
        title = f'Historical {variable.title()} Analysis for {selected_region}'
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'{variable.title()} ({unit})',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def calculate_dynamic_insights(df, variable, region_name):
    """Calculate insights based on actual data and region"""
    
    if st.session_state.real_data_loaded and 'climate_summary' in st.session_state:
        climate_summary = st.session_state.climate_summary
        hist_stats = climate_summary.get("historical_stats", {})
        trends = climate_summary.get("trends", {})
        risk_assessment = climate_summary.get("risk_assessment", {})
        
        insights = []
        
        if variable == "temperature":
            temp_data = hist_stats.get("temperature", {})
            temp_trend = trends.get("temperature", {})
            
            # Main analysis
            avg_temp = temp_data.get("mean", df[variable].mean())
            trend_change = temp_trend.get("annual_change", 0)
            
            insights.append({
                "type": "info",
                "title": "Temperature Analysis",
                "message": f"Average temperature: {avg_temp:.1f}°C. Annual trend: {trend_change:+.2f}°C per year."
            })
            
            # Risk assessment based on real data
            temp_risk = risk_assessment.get("temperature_risk", 0)
            if temp_risk >= 3:
                insights.append({
                    "type": "danger",
                    "title": "High Temperature Risk",
                    "message": f"High temperature risk detected. Extreme weather events likely to increase by {abs(trend_change * 2):.0f}%."
                })
            elif temp_risk >= 2:
                insights.append({
                    "type": "warning",
                    "title": "Rising Temperatures",
                    "message": f"Temperature rising at {trend_change:.2f}°C per year. Heat mitigation strategies recommended."
                })
            
            # Recommendation
            if trend_change > 0.3:
                insights.append({
                    "type": "info",
                    "title": "Recommendations",
                    "message": "Implement heat mitigation strategies, increase green spaces, and prepare for more frequent heatwaves."
                })
        
        elif variable == "rainfall":
            rain_data = hist_stats.get("rainfall", {})
            rain_trend = trends.get("rainfall", {})
            
            avg_rain = rain_data.get("mean", df[variable].mean())
            trend_change = rain_trend.get("annual_change", 0)
            
            insights.append({
                "type": "info",
                "title": "Rainfall Analysis",
                "message": f"Average rainfall: {avg_rain:.0f}mm. Annual trend: {trend_change:+.0f}mm per year."
            })
            
            # Risk assessment
            rain_risk = risk_assessment.get("rainfall_risk", 0)
            if rain_risk >= 3:
                insights.append({
                    "type": "danger",
                    "title": "Severe Drought Risk",
                    "message": f"Severe drought risk detected. Water conservation measures urgently needed."
                })
            elif rain_risk >= 2:
                insights.append({
                    "type": "warning",
                    "title": "Rainfall Decline",
                    "message": f"Rainfall decreasing at {trend_change:.0f}mm per year. Monitor water resources closely."
                })
            
            # Recommendation
            if trend_change < -3:
                insights.append({
                    "type": "info",
                    "title": "Recommendations",
                    "message": "Implement water conservation measures, develop drought-resistant crops, and secure alternative water sources."
                })
        
        else:  # CO2
            co2_data = hist_stats.get("co2", {})
            co2_trend = trends.get("co2", {})
            
            avg_co2 = co2_data.get("mean", df[variable].mean())
            trend_change = co2_trend.get("annual_change", 0)
            
            insights.append({
                "type": "info",
                "title": "CO₂ Analysis",
                "message": f"Average CO₂: {avg_co2:.0f}ppm. Annual trend: {trend_change:+.2f}ppm per year."
            })
            
            # Risk assessment
            co2_risk = risk_assessment.get("co2_risk", 0)
            if co2_risk >= 3:
                insights.append({
                    "type": "danger",
                    "title": "Critical CO₂ Levels",
                    "message": f"Critical CO₂ levels detected. Immediate climate action required."
                })
            elif co2_risk >= 2:
                insights.append({
                    "type": "warning",
                    "title": "Rising CO₂ Levels",
                    "message": f"CO₂ levels increasing at {trend_change:.2f}ppm per year. Enhance reduction efforts."
                })
            
            # Recommendation
            if trend_change > 0.3:
                insights.append({
                    "type": "info",
                    "title": "Recommendations",
                    "message": "Accelerate carbon reduction initiatives, invest in renewable energy, and strengthen climate policies."
                })
        
        return insights
    
    return []

def get_chart_data(_fig, format="png"):
    """Returns the chart data in the specified format"""
    buffer = io.BytesIO() if format != "html" else io.StringIO()
    if format == "html":
        _fig.write_html(buffer)
    else:
        try:
            # Try to import kaleido first
            import kaleido
            _fig.write_image(buffer, format=format)
        except ImportError:
            st.error("Please install the 'kaleido' package to export to static image formats (`pip install kaleido`).")
            return None
        except Exception as e:
            if "kaleido" in str(e).lower():
                st.error(f"Kaleido error: {e}. Try reinstalling with `pip install --force-reinstall kaleido`")
                return None
            else:
                st.error(f"Export error: {e}")
                return None
    return buffer.getvalue()

# Helper: aggregate multi-variable forecasts for advisor
def build_ai_prompt_payload():
    fd = st.session_state.get('forecast_data', {})
    if not fd:
        return None, None, None

    location = selected_region
    start_dates, end_dates, variables_present = [], [], []
    for var, info in fd.items():
        if info.get('start_date') and info.get('end_date'):
            start_dates.append(info['start_date'])
            end_dates.append(info['end_date'])
            variables_present.append(var)

    if not variables_present:
        return None, None, None

    start_date = min(start_dates) if start_dates else None
    end_date = max(end_dates) if end_dates else None

    means, mins, maxs, trends = {}, {}, {}, {}
    for var in variables_present:
        df = fd[var]['df']
        if var in df.columns and len(df) > 1:
            series = df[var]
            means[var] = float(series.mean())
            mins[var] = float(series.min())
            maxs[var] = float(series.max())
            days = (df['date'].iloc[-1] - df['date'].iloc[0]).days or 1
            trend_val = (series.iloc[-1] - series.iloc[0]) / days
            if trend_val > 0.01:
                trend_label = "rising"
            elif trend_val < -0.01:
                trend_label = "decreasing"
            else:
                trend_label = "stable"
            trends[var] = f"{trend_label} ({trend_val:+.3f}/day)"

    all_vars = {"temperature", "rainfall", "co2"}
    missing_vars = sorted(list(all_vars - set(variables_present)))

    forecast_data = {
        "location": location,
        "variables": variables_present,
        "missing_variables": missing_vars,
        "start_date": start_date,
        "end_date": end_date,
        "mean": means,
        "min": mins,
        "max": maxs,
        "trend": trends,
    }

    hist_ctx = {}
    if 'climate_summary' in st.session_state:
        cs = st.session_state['climate_summary']
        hist_ctx = {
            "historical_avg": {k: (cs.get("historical_stats", {}).get(k, {}) or {}).get("mean") for k in variables_present},
            "anomaly": {k: (cs.get("trends", {}).get(k, {}) or {}).get("annual_change") for k in variables_present},
            "extreme_events": (cs.get("risk_assessment", {}) or {}).get("extreme_events", 0)
        }

    audiences = ["farmers", "policymakers", "public_health"]
    return forecast_data, hist_ctx, audiences

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Analysis", "🤖 AI Advisor", "💬 AI Chatbot"])

# Forecast Tab
with tab1:
    st.header("📈 Climate Forecast")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        variable = st.selectbox("Select Variable", ["temperature", "rainfall", "co2"])
    
    with col2:
        start_date = st.date_input("Start Date", value=datetime.now().date())
    
    with col3:
        end_date = st.date_input("End Date", value=datetime.now().date() + timedelta(days=365))
    
    if st.button("Generate Forecast", type="primary"):
        if start_date >= end_date:
            st.error("End date must be after start date!")
        else:
            with st.spinner("Generating forecast..."):
                forecast_df, unit, color = generate_real_forecast_data(variable, start_date, end_date, selected_region)
                
                if not forecast_df.empty:
                    st.session_state.forecast_data[variable] = {
                        'df': forecast_df,
                        'unit': unit,
                        'color': color,
                        'region': selected_region,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                    st.session_state.current_forecast_variable = variable
                    st.session_state.forecast_fig = create_forecast_chart(forecast_df, variable, unit, color)
                    
                    # Clear any existing AI response to force regeneration
                    st.session_state.ai_response = None
                    st.success("✅ Forecast generated! Visit the AI Advisor tab to generate insights.")

    if 'forecast_fig' in st.session_state and 'current_forecast_variable' in st.session_state:
        st.plotly_chart(st.session_state.forecast_fig, use_container_width=True)
        
        # Display basic graph explanation (non-AI)
        current_var = st.session_state.current_forecast_variable
        fdat = st.session_state.forecast_data.get(current_var)
        if fdat:
            dfv = fdat['df']
            unit = fdat['unit']
            avg_value = dfv[current_var].mean()
            min_value = dfv[current_var].min()
            max_value = dfv[current_var].max()
            days = (dfv['date'].iloc[-1] - dfv['date'].iloc[0]).days or 1
            trend = (dfv[current_var].iloc[-1] - dfv[current_var].iloc[0]) / days
            
            st.subheader("Graph Explanation")
            graph_explanation = f"This graph shows the {current_var} forecast over time. The average value is {avg_value:.1f} {unit}, ranging from {min_value:.1f} to {max_value:.1f} {unit}. The overall trend is {trend:+.3f} {unit} per day."
            st.markdown(f'<div class="insight-card">{graph_explanation}</div>', unsafe_allow_html=True)

        # Overall Graph Details
        current_var = st.session_state.current_forecast_variable
        fdat = st.session_state.forecast_data.get(current_var)
        if fdat:
            dfv = fdat['df']
            unit = fdat['unit']
            avg_value = dfv[current_var].mean()
            min_value = dfv[current_var].min()
            max_value = dfv[current_var].max()
            days = (dfv['date'].iloc[-1] - dfv['date'].iloc[0]).days or 1
            trend = (dfv[current_var].iloc[-1] - dfv[current_var].iloc[0]) / days

            st.subheader("📋 Overall Graph Details")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Average", f"{avg_value:.1f} {unit}")
            with c2: st.metric("Minimum", f"{min_value:.1f} {unit}")
            with c3: st.metric("Maximum", f"{max_value:.1f} {unit}")
            with c4: st.metric("Trend per day", f"{trend:+.3f} {unit}")
        
        # Export options
        col1, col2 = st.columns([1, 4])
        with col1:
            export_format = st.selectbox("Export Format", ["png", "jpg", "pdf", "html"], key=f"forecast_export_{variable}")
        with col2:
            chart_data = get_chart_data(st.session_state.forecast_fig, export_format)
            if chart_data:
                st.download_button(
                    label="📥 Export Chart",
                    data=chart_data,
                    file_name=f"forecast_{variable}_{selected_region}.{export_format}",
                    mime=f"image/{export_format}" if export_format != 'html' else 'text/html',
                )

# Analysis Tab
with tab2:
    st.header("📊 Historical Analysis")
    
    has_forecast = 'current_forecast_variable' in st.session_state and st.session_state.current_forecast_variable in st.session_state.forecast_data
    
    if has_forecast:
        forecast_info = st.session_state.forecast_data[st.session_state.current_forecast_variable]
        st.info(f"🔍 Analyzing historical data for the same date range as your forecast: {forecast_info['start_date']} to {forecast_info['end_date']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_variable = st.selectbox("Select Variable", ["temperature", "rainfall", "co2"], key="analysis_var")
    
    analysis_mode = "Custom Date Range"
    if has_forecast:
        with col2:
            st.write("**Analysis Mode:**")
            analysis_mode = st.radio("Choose analysis type", ["Based on Forecast Dates", "Custom Date Range"], key="analysis_mode")
        
        with col3:
            if analysis_mode == "Based on Forecast Dates":
                st.write("**Date Range (from forecast):**")
                st.write(f"{forecast_info['start_date']} to {forecast_info['end_date']}")
                st.write("*Analyzing past 4 years for same dates*")
            else:
                start_year = st.selectbox("Start Year", list(range(2000, 2024)), key="start_year")
                end_year = st.selectbox("End Year", list(range(2000, 2025)), index=4, key="end_year")
    else:
        with col2:
            start_year = st.selectbox("Start Year", list(range(2000, 2024)), key="start_year")
        with col3:
            end_year = st.selectbox("End Year", list(range(2000, 2025)), index=4, key="end_year")

    if st.button("Analyze Historical Data", type="primary"):
        if has_forecast and analysis_mode == "Based on Forecast Dates":
            with st.spinner("Analyzing historical data for forecast date range..."):
                historical_df, unit, color = generate_date_range_historical_data(analysis_variable, forecast_info['start_date'], forecast_info['end_date'], selected_region)
                if not historical_df.empty:
                    st.session_state.analysis_data[analysis_variable] = {'df': historical_df, 'unit': unit, 'color': color}
                    st.session_state.analysis_fig = create_enhanced_historical_chart(historical_df, analysis_variable, unit, color)
        elif 'start_year' in locals() and 'end_year' in locals():
            if start_year >= end_year:
                st.error("End year must be after start year!")
            else:
                with st.spinner("Analyzing historical data..."):
                    historical_df, unit, color = generate_real_historical_data(analysis_variable, start_year, end_year, selected_region)
                    if not historical_df.empty:
                        st.session_state.analysis_data[analysis_variable] = {'df': historical_df, 'unit': unit, 'color': color}
                        st.session_state.analysis_fig = create_historical_chart(historical_df, analysis_variable, unit, color)
        else:
            st.error("Please select valid date range parameters.")

    if 'analysis_fig' in st.session_state:
        st.plotly_chart(st.session_state.analysis_fig, use_container_width=True)
        
        analysis_df = st.session_state.analysis_data[analysis_variable]['df']
        unit = st.session_state.analysis_data[analysis_variable]['unit']
        avg_value = analysis_df[analysis_variable].mean()
        min_value = analysis_df[analysis_variable].min()
        max_value = analysis_df[analysis_variable].max()
        trend = (analysis_df[analysis_variable].iloc[-1] - analysis_df[analysis_variable].iloc[0]) / len(analysis_df) * 365

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Average", f"{avg_value:.1f} {unit}")
        with c2: st.metric("Minimum", f"{min_value:.1f} {unit}")
        with c3: st.metric("Maximum", f"{max_value:.1f} {unit}")
        with c4: st.metric("Annual Trend", f"{trend:+.2f} {unit}/year", delta_color="off")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            export_format = st.selectbox("Export Format", ["png", "jpg", "pdf", "html"], key=f"analysis_export_{analysis_variable}")
        with col2:
            chart_data = get_chart_data(st.session_state.analysis_fig, export_format)
            if chart_data:
                st.download_button(
                    label="📥 Export Chart",
                    data=chart_data,
                    file_name=f"analysis_{analysis_variable}_{selected_region}.{export_format}",
                    mime=f"image/{export_format}" if export_format != 'html' else 'text/html',
                )

# AI Advisor Tab
with tab3:
    st.header("🤖 AI Climate Advisor")

    # Check if we have forecast data
    if not st.session_state.real_data_loaded or not st.session_state.get('forecast_data'):
        st.info("Please load real-time data and generate at least one forecast to activate the AI Advisor.")
    else:
        # Add manual refresh button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("### Generate AI Insights")
        with col2:
            if st.button("🔄 Refresh AI Insights", type="primary"):
                # Clear existing response and regenerate
                st.session_state.ai_response = None
                st.rerun()
        with col3:
            if st.button("⏱️ Generate with Timeout", help="Generate insights with extended timeout"):
                st.session_state.ai_response = None
                st.rerun()

        # Generate AI insights if not available or if refresh was clicked
        ai_response = st.session_state.get('ai_response')
        
        if not ai_response:
            with st.spinner("🧠 Generating AI insights... This may take a moment."):
                try:
                    forecast_data, historical_context, audiences = build_ai_prompt_payload()
                    if forecast_data:
                        advisor = ClimateAIAdvisor()
                        # Add timeout handling with extended timeout for better reliability
                        import time
                        start_time = time.time()
                        ai_resp = advisor.get_insights(forecast_data, historical_context, audiences=audiences, timeout=60)
                        generation_time = time.time() - start_time
                        
                        st.session_state.ai_response = ai_resp
                        
                        # Show success message
                        st.success("✅ AI insights generated successfully!")
                    else:
                        st.session_state.ai_response = {
                            "success": False,
                            "error": "No forecast data available for analysis"
                        }
                except Exception as e:
                    st.session_state.ai_response = {
                        "success": False,
                        "error": f"Failed to generate AI insights: {str(e)}"
                    }
                    st.error(f"❌ Error generating AI insights: {str(e)}")
        
        # Display the AI response
        ai_response = st.session_state.get('ai_response')
        
        if not ai_response or not ai_response.get('success'):
            error_msg = ai_response.get('error') if ai_response else 'No response generated'
            st.warning(f"⚠️ AI insights are not available. Error: {error_msg}")
            
            # Show comprehensive fallback insights based on forecast data
            st.subheader("📊 Climate Analysis (Based on Forecast Data)")
            forecast_data, _, _ = build_ai_prompt_payload()
            if forecast_data:
                variables = forecast_data.get('variables', [])
                means = forecast_data.get('mean', {})
                trends = forecast_data.get('trend', {})
                location = forecast_data.get('location', 'the region')
                
                # Generate comprehensive insights
                insights_parts = []
                insights_parts.append(f"**Climate Analysis for {location}:**")
                insights_parts.append("")
                
                for var in variables:
                    mean_val = means.get(var, 0)
                    trend_info = trends.get(var, "stable")
                    
                    if var == "temperature":
                        if "rising" in trend_info.lower():
                            insights_parts.append(f"🌡️ **Temperature Trend**: Rising temperatures (avg {mean_val:.1f}°C) indicate increasing heat stress. This suggests higher energy demand for cooling, increased heat-related health risks, and potential impacts on agriculture.")
                        elif "decreasing" in trend_info.lower():
                            insights_parts.append(f"🌡️ **Temperature Trend**: Decreasing temperatures (avg {mean_val:.1f}°C) may indicate cooling trends. This could affect heating demand, agricultural growing seasons, and energy consumption patterns.")
                        else:
                            insights_parts.append(f"🌡️ **Temperature Trend**: Stable temperatures (avg {mean_val:.1f}°C) suggest consistent climate patterns. Monitor for any sudden changes that could indicate climate shifts.")
                    
                    elif var == "rainfall":
                        if "decreasing" in trend_info.lower():
                            insights_parts.append(f"🌧️ **Rainfall Trend**: Decreasing rainfall (avg {mean_val:.1f}mm) indicates drought risk. This suggests water scarcity concerns, agricultural irrigation needs, and potential impacts on water resources.")
                        elif "increasing" in trend_info.lower():
                            insights_parts.append(f"🌧️ **Rainfall Trend**: Increasing rainfall (avg {mean_val:.1f}mm) suggests wetter conditions. This could lead to flooding risks, soil erosion, and changes in agricultural practices.")
                        else:
                            insights_parts.append(f"🌧️ **Rainfall Trend**: Stable rainfall patterns (avg {mean_val:.1f}mm) indicate consistent precipitation. Continue monitoring for seasonal variations and extreme weather events.")
                    
                    elif var == "co2":
                        if "rising" in trend_info.lower():
                            insights_parts.append(f"🌫️ **CO₂ Trend**: Rising CO₂ levels (avg {mean_val:.1f}ppm) indicate increasing atmospheric carbon. This contributes to global warming, affects air quality, and requires immediate carbon reduction strategies.")
                        elif "decreasing" in trend_info.lower():
                            insights_parts.append(f"🌫️ **CO₂ Trend**: Decreasing CO₂ levels (avg {mean_val:.1f}ppm) suggest positive environmental progress. This indicates effective carbon reduction efforts and improved air quality.")
                        else:
                            insights_parts.append(f"🌫️ **CO₂ Trend**: Stable CO₂ levels (avg {mean_val:.1f}ppm) suggest consistent atmospheric conditions. Continue monitoring and maintain carbon reduction efforts.")
                
                insights_parts.append("")
                insights_parts.append("**Risk Assessment**: Based on the forecast data, monitor for extreme weather events, temperature anomalies, and precipitation changes that could impact local communities.")
                
                # Display insights
                insights_text = "\n".join(insights_parts)
                st.markdown(f'<div class="insight-card" style="background-color: #e9f5ff; border-left-color: #17a2b8;">{insights_text}</div>', unsafe_allow_html=True)
                
                # Generate detailed recommendations
                st.subheader("🎯 Recommendations by Audience")
                basic_recs = {
                    "farmers": [
                        "Monitor soil moisture levels and implement water conservation techniques",
                        "Adapt crop varieties to changing temperature and rainfall patterns",
                        "Plan irrigation schedules based on forecast precipitation",
                        "Prepare for extreme weather events with protective measures",
                        "Consider climate-smart agricultural practices"
                    ],
                    "policymakers": [
                        "Develop climate adaptation strategies based on forecast trends",
                        "Implement water management policies for changing precipitation patterns",
                        "Create emergency response plans for extreme weather events",
                        "Support renewable energy initiatives to reduce CO₂ emissions",
                        "Invest in climate monitoring and early warning systems"
                    ],
                    "public_health": [
                        "Monitor heat-related health risks during temperature extremes",
                        "Prepare for air quality impacts from changing CO₂ levels",
                        "Develop health advisories for weather-related health risks",
                        "Plan for increased demand during extreme weather events",
                        "Create public awareness campaigns about climate health impacts"
                    ]
                }
                
                for audience, recs in basic_recs.items():
                    with st.expander(f"For {audience.replace('_', ' ').title()}", expanded=True):
                        for rec in recs:
                            st.markdown(f"• {rec}")
        else:
            # Display missing variables warning
            payload, _, _ = build_ai_prompt_payload()
            if payload and payload.get("missing_variables"):
                missing = ", ".join(payload["missing_variables"])
                st.warning(f"**Missing Forecasts:** The AI analysis is based only on the variables you have forecasted so far ({', '.join(payload['variables'])}). For a complete picture, generate forecasts for {missing}.")

            # Display AI Insights
            st.subheader("💡 AI Insights Summary")
            insights = ai_response.get('ai_insights', "No insights generated.")
            
            # Ensure insights is a string, not a dict
            if isinstance(insights, dict):
                insights = str(insights)
            elif not isinstance(insights, str):
                insights = "No insights generated."
                
            st.markdown(f'<div class="insight-card" style="background-color: #e9f5ff; border-left-color: #17a2b8;">{insights}</div>', unsafe_allow_html=True)

            # Display Audience-Specific Recommendations
            st.subheader("🎯 Audience-Specific Recommendations")
            aud_recs = ai_response.get('audience_recommendations')
            if aud_recs and isinstance(aud_recs, dict):
                for audience, recs in aud_recs.items():
                    with st.expander(f"For {audience.replace('_', ' ').title()}", expanded=True):
                        if isinstance(recs, list):
                            for rec in recs:
                                # Ensure each recommendation is a string
                                if isinstance(rec, dict):
                                    st.markdown(f"- {str(rec)}")
                                else:
                                    st.markdown(f"- {rec}")
                        else:
                            st.markdown(str(recs))
            else:
                st.info("No specific recommendations were generated for different audiences.")
        
        

# AI Chatbot Tab
with tab4:
    st.header("💬 AI Climate Chatbot")
    st.markdown("Ask questions about specific dates and get personalized climate recommendations!")
    
    # Initialize chatbot if not exists
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ClimateAIChatbot(model="llama3.2:3b")
    
    # Date selection and region
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_date = st.date_input(
            "📅 Select Future Date",
            value=datetime.now().date() + timedelta(days=30),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=365*2),
            help="Choose a future date to ask about"
        )
    
    with col2:
        region_options = ["New York, USA", "London, UK", "Mumbai, India", "Sydney, Australia", "Tokyo, Japan"]
        selected_region = st.selectbox(
            "🌍 Select Region",
            options=region_options,
            index=0,
            help="Choose the region for your climate query"
        )
    
    # Get forecast data for the selected date
    forecast_data = None
    
    # For far future dates (beyond 2 years), use synthetic data
    current_year = datetime.now().year
    selected_year = selected_date.year
    
    if selected_year > current_year + 2:
        # Generate synthetic forecast data for far future dates
        st.info(f"ℹ️ Generating synthetic forecast data for {selected_date} (far future date)")
        
        # Create synthetic data based on seasonal patterns
        month = selected_date.month
        day = selected_date.day
        
        # Generate realistic seasonal data
        if month in [12, 1, 2]:  # Winter
            base_temp = 15
            base_rainfall = 5
        elif month in [3, 4, 5]:  # Spring
            base_temp = 20
            base_rainfall = 8
        elif month in [6, 7, 8]:  # Summer
            base_temp = 28
            base_rainfall = 3
        else:  # Fall
            base_temp = 22
            base_rainfall = 6
        
        # Add some variation based on day
        temp_variation = (day % 10) - 5
        rainfall_variation = (day % 7) - 3
        
        forecast_data = pd.DataFrame({
            'date': [pd.to_datetime(selected_date)],
            'temperature': [base_temp + temp_variation + np.random.normal(0, 2)],
            'rainfall': [max(0, base_rainfall + rainfall_variation + np.random.normal(0, 1))],
            'co2': [420 + np.random.normal(0, 5)]
        })
        
        st.success(f"✅ Synthetic forecast data generated for {selected_date}")
        
    elif st.session_state.get('real_data_loaded') and st.session_state.get('forecast_data'):
        try:
            # Generate forecast data for the selected date range
            start_date = selected_date.strftime("%Y-%m-%d")
            end_date = (selected_date + timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Try to get forecast data for multiple variables
            forecast_data = pd.DataFrame()
            
            for variable in ["temperature", "rainfall", "co2"]:
                try:
                    var_data, _, _ = generate_real_forecast_data(variable, start_date, end_date, selected_region)
                    if not var_data.empty:
                        if forecast_data.empty:
                            forecast_data = var_data.copy()
                        else:
                            # Merge with existing data
                            if variable in var_data.columns:
                                forecast_data[variable] = var_data[variable]
                except Exception as e:
                    st.warning(f"Could not fetch {variable} data: {str(e)}")
                    continue
            
            if not forecast_data.empty:
                st.success(f"✅ Forecast data loaded for {selected_date}")
            else:
                st.warning("⚠️ No forecast data available for the selected date")
        except Exception as e:
            st.error(f"❌ Error loading forecast data: {str(e)}")
    else:
        st.info("ℹ️ Please load real-time data first to get accurate forecasts")
    
    # Chat interface
    st.subheader("💬 Ask Your Question")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📝 Conversation History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {chat['user_question'][:50]}... ({chat['selected_date']})", expanded=False):
                st.markdown(f"**Date:** {chat['selected_date']}")
                st.markdown(f"**Region:** {chat['region']}")
                st.markdown(f"**Question:** {chat['user_question']}")
                st.markdown(f"**AI Response:** {chat['ai_response']}")
                st.markdown(f"**Time:** {chat['timestamp']}")
    
    # Question input
    user_question = st.text_area(
        "Ask about climate conditions, farming advice, health recommendations, or policy suggestions:",
        placeholder="e.g., Is this a good month for crop planting? What should I expect for outdoor activities? How will this affect air quality?",
        height=100,
        help="Ask specific questions about the selected date and region"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        ask_button = st.button("🤖 Ask AI", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    with col3:
        example_button = st.button("💡 Example Questions", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.chatbot.clear_history()
        st.rerun()
    
    if example_button:
        example_questions = [
            "Is this a good month for crop planting?",
            "What should I expect for outdoor activities?",
            "How will this affect air quality?",
            "Should I prepare for extreme weather?",
            "Is this suitable for construction work?",
            "What health precautions should I take?",
            "How will this impact water resources?",
            "Should I adjust my farming schedule?"
        ]
        st.info("💡 **Example Questions:** " + " | ".join(example_questions))
    
    # Process question
    if ask_button and user_question.strip():
        if forecast_data is None or forecast_data.empty:
            # Create basic forecast data for the selected date as fallback
            st.warning("⚠️ Using basic forecast data - for more accurate responses, please load real-time data first")
            
            # Generate more realistic fallback data based on season
            month = selected_date.month
            if month in [12, 1, 2]:  # Winter
                temp, rain = 15, 5
            elif month in [3, 4, 5]:  # Spring
                temp, rain = 20, 8
            elif month in [6, 7, 8]:  # Summer
                temp, rain = 28, 3
            else:  # Fall
                temp, rain = 22, 6
                
            forecast_data = pd.DataFrame({
                'date': [pd.to_datetime(selected_date)],
                'temperature': [temp],
                'rainfall': [rain],
                'co2': [420.0]
            })
        
        with st.spinner("🤖 AI is thinking..."):
            try:
                response = st.session_state.chatbot.get_chat_response(
                    user_question=user_question.strip(),
                    forecast_data=forecast_data,
                    selected_date=selected_date.strftime("%Y-%m-%d"),
                    region=selected_region
                )
                
                if response['success']:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user_question": user_question.strip(),
                        "selected_date": selected_date.strftime("%Y-%m-%d"),
                        "region": selected_region,
                        "ai_response": response['response'],
                        "timestamp": response['timestamp']
                    })
                    
                    st.success("✅ AI Response Generated!")
                    st.markdown("---")
                    st.markdown("### 🤖 AI Response:")
                    st.markdown(response['response'])
                    st.rerun()
                else:
                    st.error(f"❌ Error: {response['error']}")
                    
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
    
    # Display current forecast data for reference
    if forecast_data is not None and not forecast_data.empty:
        st.subheader("📊 Current Forecast Data")
        st.dataframe(forecast_data.head(10), use_container_width=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p><strong>ClimaCast - Climate Impact Predictor</strong></p>
    <p>Built for sustainability and climate awareness | Hackathon Project</p>
    <p>Powered by NOAA Climate Data & OpenWeatherMap API | AI-Powered Analysis</p>
</div>
""", unsafe_allow_html=True)