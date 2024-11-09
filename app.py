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
        if st.session_state.get('current_page') == 'Home':
            self.show_home_page()
        elif st.session_state.get('current_page') == 'Data Upload & Analysis':
            self.show_data_analysis_page()
        elif st.session_state.get('current_page') == 'Weather Insights':
            self.show_weather_insights_page()
        elif st.session_state.get('current_page') == 'Soil Analysis':
            self.show_soil_analysis_page()
        elif st.session_state.get('current_page') == 'Growth Predictions':
            self.show_predictions_page()
        elif st.session_state.get('current_page') == 'Dashboard':
            self.show_dashboard_page()

    def create_sidebar(self):
        with st.sidebar:
            st.image("https://via.placeholder.com/150", caption="ADAP Logo")
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
        
        # Introduction section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ## Welcome to ADAP
            The Agricultural Data Analysis Platform (ADAP) helps coffee farmers make 
            data-driven decisions by analyzing environmental factors affecting coffee growth.
            
            ### Key Features:
            - 📊 Comprehensive data analysis
            - 🌤️ Weather pattern insights
            - 🌱 Soil condition monitoring
            - 📈 Growth predictions
            - 📱 Real-time monitoring
            """)
        
        with col2:
            # Placeholder for a feature image or stats
            st.image("https://via.placeholder.com/300x200", caption="Coffee Farm Analytics")
        
        # Quick start guide
        st.markdown("---")
        st.markdown("## 🚀 Quick Start Guide")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 1. Upload Data
            Upload your farm's environmental data to get started with analysis.
            """)
        with col2:
            st.markdown("""
            ### 2. Analyze Patterns
            Explore weather and soil patterns affecting growth.
            """)
        with col3:
            st.markdown("""
            ### 3. Get Predictions
            Receive AI-powered growth predictions and recommendations.
            """)

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
                
                # Data preview tab
                with st.expander("📝 Data Preview"):
                    st.dataframe(st.session_state.data.head())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Data Summary")
                        st.write(st.session_state.data.describe())
                    with col2:
                        st.markdown("#### Missing Values")
                        st.write(st.session_state.data.isnull().sum())
                
                # Data visualization section
                st.markdown("### 📈 Data Visualization")
                viz_type = st.selectbox(
                    "Select Visualization Type",
                    ["Correlation Analysis", "Time Series", "Distribution", "Scatter Plot"]
                )
                
                if viz_type == "Correlation Analysis":
                    fig = create_correlation_plot(st.session_state.data)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Time Series":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select Time Column", st.session_state.data.columns)
                    with col2:
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
        
        # Weather analysis tabs
        tab1, tab2, tab3 = st.tabs(["Temperature Analysis", "Rainfall Patterns", "Humidity Trends"])
        
        with tab1:
            st.markdown("### Temperature Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Temperature", "23.5°C", "1.2°C")
            with col2:
                st.metric("Temperature Range", "18°C - 29°C", "-2°C")
            
            # Placeholder for temperature trend chart
            st.markdown("#### Temperature Trends")
            st.line_chart(np.random.randn(20, 2))
        
        with tab2:
            st.markdown("### Rainfall Patterns")
            # Add rainfall analysis visualizations here
            st.info("Rainfall analysis visualizations will be added here")
        
        with tab3:
            st.markdown("### Humidity Trends")
            # Add humidity analysis visualizations here
            st.info("Humidity analysis visualizations will be added here")

    def show_soil_analysis_page(self):
        st.title("🌱 Soil Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload & Analysis section.")
            return
        
        # Soil analysis sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Soil pH Levels")
            # Placeholder for pH level chart
            st.bar_chart(np.random.randn(10, 1))
            
        with col2:
            st.markdown("### Nutrient Levels")
            nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
            values = np.random.rand(3) * 100
            
            # Create a simple bar chart for nutrients
            fig = go.Figure(data=[
                go.Bar(name='Current Level', x=nutrients, y=values)
            ])
            st.plotly_chart(fig)
        
        # Soil composition analysis
        st.markdown("### Soil Composition Analysis")
        composition_data = {
            'Clay': 30,
            'Silt': 40,
            'Sand': 30
        }
        
        fig = go.Figure(data=[go.Pie(labels=list(composition_data.keys()),
                                    values=list(composition_data.values()))])
        st.plotly_chart(fig)

    def show_predictions_page(self):
        st.title("📈 Growth Predictions")
        
        if st.session_state.data is None:
            st.warning("Please upload data first in the Data Upload & Analysis section.")
            return
        
        # Model selection and training
        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Linear Regression"]
            )
            
        with col2:
            target_variable = st.selectbox(
                "Select Target Variable",
                st.session_state.data.columns
            )
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Placeholder for model training
                st.success("Model trained successfully!")
        
        # Predictions section
        st.markdown("### Make Predictions")
        with st.form("prediction_form"):
            # Add input fields for prediction
            col1, col2, col3 = st.columns(3)
            with col1:
                temp = st.number_input("Temperature (°C)", value=25.0)
            with col2:
                rainfall = st.number_input("Rainfall (mm)", value=100.0)
            with col3:
                soil_ph = st.number_input("Soil pH", value=6.5)
                
            submitted = st.form_submit_button("Get Prediction")
            if submitted:
                # Placeholder for prediction result
                st.success("Predicted Growth Rate: 75%")
                
                # Show feature importance
                st.markdown("### Feature Importance")
                importance_data = {
                    'Temperature': 0.3,
                    'Rainfall': 0.4,
                    'Soil pH': 0.3
                }
                fig = go.Figure(data=[
                    go.Bar(x=list(importance_data.keys()),
                          y=list(importance_data.values()))
                ])
                st.plotly_chart(fig)

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