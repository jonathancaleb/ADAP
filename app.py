import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data.data_loader import DataLoader
from src.visualization.plots import (
    create_correlation_plot,
    create_time_series_plot,
    create_distribution_plot,
    create_scatter_plot
)
from src.models.predictor import CoffeeGrowthPredictor

class CoffeeGrowthAnalysisPlatform:
    def __init__(self):
        st.set_page_config(
            page_title="Coffee Growth Analysis Platform",
            page_icon="☕",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize components
        self.data_loader = DataLoader()
        self.predictor = CoffeeGrowthPredictor()
        
        # Session state initialization
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model' not in st.session_state:
            st.session_state.model = None

    def run(self):
        # Sidebar navigation
        self.create_sidebar()
        
        # Main content
        page = st.session_state.get('current_page')
        if page == 'Home':
            self.show_home_page()
        elif page == 'Data Upload & Analysis':
            self.show_data_analysis_page()
        elif page == 'Weather Insights':
            self.show_weather_insights_page()
        elif page == 'Soil Analysis':
            self.show_soil_analysis_page()
        elif page == 'Growth Predictions':
            self.show_predictions_page()
        elif page == 'Dashboard':
            self.show_dashboard_page()

    def create_sidebar(self):
        with st.sidebar:
            st.title("Navigation")
            
            pages = {
                'Home': '🏠',
                'Data Upload & Analysis': '📊',
                'Weather Insights': '🌤️',
                'Soil Analysis': '🌱',
                'Growth Predictions': '📈',
                'Dashboard': '📱'
            }
            
            st.session_state.current_page = st.radio(
                "Go to",
                pages.keys(),
                format_func=lambda x: f"{pages[x]} {x}"
            )
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Quick Stats")
            if st.session_state.data is not None:
                st.sidebar.metric("Total Records", len(st.session_state.data))
                st.sidebar.metric("Data Range", "Jan 2023 - Present")
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### About")
            st.sidebar.info(
                """
                This platform analyzes coffee growth patterns 
                in Uganda using environmental data and 
                machine learning techniques.
                """
            )

    def show_home_page(self):
        st.title("☕ Coffee Growth Analysis Platform")
        
        # Interactive Map of Uganda with placeholder data
        st.markdown("### Uganda Coffee Growing Regions and Climate Insights")
        locations = pd.DataFrame({
            'Region': ['Region A', 'Region B', 'Region C'],
            'Latitude': [1.0, 1.5, -0.5],
            'Longitude': [32.0, 32.5, 31.5],
            'Avg Temp (°C)': [24, 22, 26],
            'Soil Quality': ['Good', 'Average', 'Excellent']
        })
        
        fig = px.scatter_mapbox(
            locations,
            lat="Latitude",
            lon="Longitude",
            text="Region",
            hover_data={'Avg Temp (°C)': True, 'Soil Quality': True},
            zoom=6,
            height=500
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            title="Map of Uganda with Coffee Growing Regions",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Overview of insights
        st.markdown("---")
        st.markdown("### Regional Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Coffee Growing Region", "Region A")
        with col2:
            st.metric("Optimal Temperature", "24°C")
        with col3:
            st.metric("Highest Soil Quality", "Excellent")
        
        st.info("Add real data to visualize soil quality, temperature ranges, and crop growth regions.")

    def show_data_analysis_page(self):
        st.title("📊 Data Upload & Analysis")
        
        # Data upload section
        st.markdown("### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file containing your farm data",
            type=['csv'],
            help="Upload a CSV file with columns for temperature, rainfall, soil pH, etc."
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
                
                # Data preview and summary
                with st.expander("📝 Data Preview"):
                    st.dataframe(st.session_state.data.head())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Data Summary")
                        st.write(st.session_state.data.describe())
                    with col2:
                        st.markdown("#### Missing Values")
                        st.write(st.session_state.data.isnull().sum())
                
                # Visualization Options
                st.markdown("### 📈 Data Visualization")
                viz_type = st.selectbox(
                    "Select Visualization Type",
                    ["Correlation Analysis", "Time Series", "Distribution", "Scatter Plot"]
                )
                
                if viz_type == "Correlation Analysis":
                    fig = create_correlation_plot(st.session_state.data)
                    st.plotly_chart(fig, use_container_width=True)
                elif viz_type == "Time Series":
                    x_col = st.selectbox("Select Time Column", st.session_state.data.columns)
                    y_col = st.selectbox("Select Value Column", st.session_state.data.columns)
                    fig = create_time_series_plot(st.session_state.data, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")

    def show_weather_insights_page(self):
        st.title("🌤️ Weather Insights")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload & Analysis section.")
            return
        
        # Temperature and Rainfall Analysis
        tab1, tab2 = st.tabs(["Temperature Analysis", "Rainfall Patterns"])
        
        with tab1:
            st.markdown("### Temperature Trends")
            temp_data = st.session_state.data['temperature'] if 'temperature' in st.session_state.data else np.random.randn(50)
            st.line_chart(temp_data)
        
        with tab2:
            st.markdown("### Rainfall Patterns")
            rainfall_data = st.session_state.data['rainfall'] if 'rainfall' in st.session_state.data else np.random.randn(50)
            st.bar_chart(rainfall_data)

    def show_soil_analysis_page(self):
        st.title("🌱 Soil Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload & Analysis section.")
            return
        
        # Soil pH and nutrient analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Soil pH Levels")
            ph_data = st.session_state.data['soil_ph'] if 'soil_ph' in st.session_state.data else np.random.randn(20)
            st.bar_chart(ph_data)
            
        with col2:
            st.markdown("### Nutrient Levels")
            nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
            values = np.random.rand(3) * 100
            fig = go.Figure(data=[go.Bar(name='Nutrients', x=nutrients, y=values)])
            st.plotly_chart(fig)
        
    def show_predictions_page(self):
        st.title("📈 Growth Predictions")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload & Analysis section.")
            return
        
        st.markdown("### Model Configuration")
        model_type = st.selectbox("Select Model Type", ["Random Forest", "Linear Regression"])
        target_variable = st.selectbox("Select Target Variable", st.session_state.data.columns)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Mock training and placeholder model training function
                st.session_state.model = self.predictor.train(st.session_state.data, target_variable)
                st.success("Model trained successfully!")
        
        # Prediction Form
        st.markdown("### Make Predictions")
        with st.form("prediction_form"):
            temp = st.number_input("Temperature (°C)", value=25.0)
            rainfall = st.number_input("Rainfall (mm)", value=100.0)
            soil_ph = st.number_input("Soil pH", value=6.5)
            submitted = st.form_submit_button("Get Prediction")
            
            if submitted:
                prediction = self.predictor.predict(temp, rainfall, soil_ph)
                st.success(f"Predicted Growth Rate: {prediction}%")

    def show_dashboard_page(self):
        st.title("📱 Dashboard")
        
        # Create a dashboard layout with key metrics and charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Average Growth Rate",
                value="2.3 cm/week",
                delta="0.3 cm"
            )
        
        with col2:
            st.metric(
                label="Soil Health Index",
                value="85%",
                delta="5%"
            )
        
        with col3:
            st.metric(
                label="Weather Condition",
                value="Optimal",
                delta="Stable"
            )
        
        # Add interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Growth Trends")
            # Placeholder for growth trend chart
            st.line_chart(np.random.randn(20, 1))
        
        with col2:
            st.markdown("### Environmental Conditions")
            # Placeholder for environmental conditions chart
            st.area_chart(np.random.randn(20, 3))
        
        # Add a data table with recent measurements
        st.markdown("### Recent Measurements")
        df = pd.DataFrame(
            np.random.randn(5, 4),
            columns=['Temperature', 'Humidity', 'Soil pH', 'Growth Rate']
        )
        st.dataframe(df)

if __name__ == "__main__":
    app = CoffeeGrowthAnalysisPlatform()
    app.run()