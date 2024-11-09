import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_correlation_plot(data: pd.DataFrame, numeric_columns: list = None):
    """
    Create a correlation heatmap using plotly
    
    Args:
        data (pd.DataFrame): Input dataframe
        numeric_columns (list, optional): List of numeric columns to include in correlation
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap figure
    """
    # If no numeric columns specified, use all numeric columns
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        height=600,
        width=800
    )
    
    return fig

def create_time_series_plot(data: pd.DataFrame, 
                          date_column: str, 
                          value_column: str,
                          title: str = None):
    """
    Create a time series plot using plotly
    
    Args:
        data (pd.DataFrame): Input dataframe
        date_column (str): Name of the date column
        value_column (str): Name of the value column to plot
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Time series figure
    """
    fig = px.line(data, 
                  x=date_column, 
                  y=value_column,
                  title=title or f'{value_column} over time')
    
    fig.update_layout(
        xaxis_title=date_column,
        yaxis_title=value_column,
        height=400
    )
    
    return fig

def create_scatter_plot(data: pd.DataFrame,
                       x_column: str,
                       y_column: str,
                       color_column: str = None,
                       title: str = None):
    """
    Create a scatter plot using plotly
    
    Args:
        data (pd.DataFrame): Input dataframe
        x_column (str): Name of the x-axis column
        y_column (str): Name of the y-axis column
        color_column (str, optional): Name of the column to use for color coding
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot figure
    """
    fig = px.scatter(data,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    title=title or f'{y_column} vs {x_column}')
    
    fig.update_layout(
        height=400,
        showlegend=bool(color_column)
    )
    
    return fig

def create_distribution_plot(data: pd.DataFrame, column: str):
    """
    Create a distribution plot using plotly
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Name of the column to plot
        
    Returns:
        plotly.graph_objects.Figure: Distribution plot figure
    """
    fig = px.histogram(data,
                      x=column,
                      title=f'Distribution of {column}',
                      marginal='box')
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    return fig

def create_box_plot(data: pd.DataFrame,
                   category_column: str,
                   value_column: str,
                   title: str = None):
    """
    Create a box plot using plotly
    
    Args:
        data (pd.DataFrame): Input dataframe
        category_column (str): Name of the category column
        value_column (str): Name of the value column
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Box plot figure
    """
    fig = px.box(data,
                 x=category_column,
                 y=value_column,
                 title=title or f'{value_column} by {category_column}')
    
    fig.update_layout(
        height=400
    )
    
    return fig