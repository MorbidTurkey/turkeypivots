"""
Chart generation utilities using Plotly with time series sorting
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, List
import re
import datetime

class ChartGenerator:
    def __init__(self):
        # Define a nice color palette for charts
        self.color_palette = px.colors.qualitative.Plotly

    def _create_error_chart(self, error_message):
        """Create a chart showing an error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#FF0000")
        )
        fig.update_layout(
            title="Chart Error",
            template="plotly_white",
            height=400
        )
        return fig

    def _detect_date_column(self, df, column_name):
        """
        Detect if a column is likely a date or time column and return 
        converted values if possible
        """
        # Already a datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column_name]):
            return True, df[column_name]
            
        # Check sample values for date patterns
        if df[column_name].dtype == object:
            # Common date formats to check
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY or MM-DD-YYYY
                r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
                r'\d{1,2} \w+ \d{4}'   # DD Month YYYY
            ]
            
            sample = df[column_name].dropna().head(10)
            for value in sample:
                for pattern in date_patterns:
                    if re.match(pattern, str(value)):
                        # Try to convert the column to datetime
                        try:
                            converted = pd.to_datetime(df[column_name], errors='coerce')
                            if converted.isna().sum() / len(df) < 0.2:  # Less than 20% NaN values
                                return True, converted
                        except:
                            pass
                        
        # Check if name suggests date
        date_keywords = ['date', 'time', 'year', 'month', 'day', 'created', 'modified', 'timestamp']
        if any(keyword in column_name.lower() for keyword in date_keywords):
            try:
                converted = pd.to_datetime(df[column_name], errors='coerce')
                if converted.isna().sum() / len(df) < 0.3:  # Less than 30% NaN values
                    return True, converted
            except:
                pass
                
        return False, None

    def _sort_time_series(self, df, x_col):
        """
        Sort dataframe by date column if x_col is a date column.
        Returns the sorted dataframe and a flag indicating whether sorting was performed.
        """
        is_date, date_series = self._detect_date_column(df, x_col)
        if is_date:
            # Create a copy of the dataframe with the converted date column
            sorted_df = df.copy()
            sorted_df[x_col] = date_series
            
            # Sort by date
            sorted_df = sorted_df.sort_values(by=x_col)
            return True, sorted_df
        
        return False, df

    def generate_insight(self, df, chart_config):
        """
        Generate a simple insight/summary for the chart.
        """
        chart_type = chart_config.get('type', 'bar')
        x_col = chart_config.get('x')
        y_col = chart_config.get('y')
        filters = chart_config.get('filters', {})
        title = chart_config.get('title', '')

        # Bar chart with count aggregation
        if chart_type == 'bar' and x_col and (y_col == 'count' or filters.get('aggregation') == 'count'):
            # Check if data is already pre-aggregated (has 'count' column)
            if 'count' in df.columns:
                # Use the pre-aggregated data
                most_common_row = df.loc[df['count'].idxmax()]
                least_common_row = df.loc[df['count'].idxmin()]
                
                most_common = most_common_row[x_col]
                least_common = least_common_row[x_col]
                most_common_count = most_common_row['count']
                least_common_count = least_common_row['count']
                total_items = df['count'].sum()
            else:
                # Count occurrences
                counts = df[x_col].value_counts()
                most_common = counts.idxmax()
                least_common = counts.idxmin()
                most_common_count = counts[most_common]
                least_common_count = counts[least_common]
                total_items = counts.sum()
            
            insight = f"The most common {x_col} is '{most_common}' with {most_common_count} occurrences "
            insight += f"({(most_common_count/total_items*100):.1f}% of total). "
            insight += f"The least common is '{least_common}' with {least_common_count} occurrences."
            
            return insight
        
        # Line chart - detect trends
        elif chart_type == 'line' and x_col and y_col:
            is_date, _ = self._detect_date_column(df, x_col)
            
            if is_date:
                # Sort by date
                _, sorted_df = self._sort_time_series(df, x_col)
                
                # Calculate trend
                try:
                    y_values = sorted_df[y_col].dropna()
                    if len(y_values) > 1:
                        first = y_values.iloc[0]
                        last = y_values.iloc[-1]
                        change = last - first
                        percent_change = (change / first * 100) if first != 0 else float('inf')
                        
                        # Determine direction
                        if percent_change > 0:
                            direction = "increased"
                        elif percent_change < 0:
                            direction = "decreased"
                        else:
                            direction = "remained stable"
                            
                        insight = f"Over this period, {y_col} {direction} "
                        if not np.isinf(percent_change):
                            insight += f"by {abs(percent_change):.1f}% "
                        insight += f"from {first:.2f} to {last:.2f}."
                        
                        return insight
                except:
                    pass
            
            return f"This line chart shows the relationship between {x_col} and {y_col}."
        
        # Pie chart - show distribution
        elif chart_type == 'pie' and (chart_config.get('names') or x_col):
            names_col = chart_config.get('names') or x_col
            values_col = chart_config.get('values') or y_col
            
            if values_col and values_col in df.columns:
                # Get distribution by values
                total = df[values_col].sum()
                top_items = df.groupby(names_col)[values_col].sum().sort_values(ascending=False).head(3)
                
                insight = f"Top categories by {values_col}: "
                for i, (cat, val) in enumerate(top_items.items()):
                    if i > 0:
                        insight += ", "
                    insight += f"{cat} ({(val/total*100):.1f}%)"
                
                return insight
            else:
                # Count-based
                counts = df[names_col].value_counts().head(3)
                total = counts.sum()
                
                insight = f"Top categories: "
                for i, (cat, val) in enumerate(counts.items()):
                    if i > 0:
                        insight += ", "
                    insight += f"{cat} ({(val/total*100):.1f}%)"
                
                return insight
                
        # Generic insight for other chart types
        return f"This {chart_type} chart visualizes {y_col or ''} {'by ' + x_col if x_col else ''}."

    def create_chart(self, df, chart_config):
        """
        Create a chart based on configuration
        """
        if df is None or df.empty:
            return self._create_error_chart("No data available")
            
        chart_type = chart_config.get('type', 'bar')
        x_col = chart_config.get('x')
        y_col = chart_config.get('y')
        color_col = chart_config.get('color')
        title = chart_config.get('title', 'Chart')
        
        # Special handling for different chart types
        
        # For line charts, sort by date if x column is a date
        if chart_type == 'line' and x_col:
            is_time_series, sorted_df = self._sort_time_series(df, x_col)
            if is_time_series:
                df = sorted_df
            
        # Bar chart
        if chart_type == 'bar':
            if x_col is None:
                return self._create_error_chart("X-axis column not specified")
                
            # Handle count aggregation for bar charts
            if y_col == 'count' or chart_config.get('aggregation') == 'count':
                # Check if the data is already pre-aggregated (has 'count' column)
                if 'count' in df.columns:
                    # Use pre-aggregated data directly
                    print("Using pre-aggregated data for count-based chart")
                    
                    fig = px.bar(df, x=x_col, y='count', title=title)
                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title='Count',
                        template='plotly_white'
                    )
                    return fig
                else:
                    # Create aggregation on-the-fly if needed
                    print("Creating on-the-fly aggregation for count-based chart")
                    value_counts = df[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'count']
                    
                    fig = px.bar(value_counts, x=x_col, y='count', title=title)
                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title='Count',
                        template='plotly_white'
                    )
                    return fig
            
            # Check if columns exist (for regular bar charts)
            if x_col not in df.columns:
                return self._create_error_chart(f"Column '{x_col}' not found in data")
            if y_col not in df.columns:
                return self._create_error_chart(f"Column '{y_col}' not found in data")
            
            # Try to create the chart with original data
            try:
                fig = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=title,
                    color_discrete_sequence=self.color_palette
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45, 
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as px_error:
                # Provide specific error information
                error_msg = f"Chart creation failed: {str(px_error)}\n\n"
                error_msg += "This usually means:\n"
                error_msg += f"• Column '{y_col}' contains mixed data types\n"
                error_msg += f"• Some values in '{y_col}' cannot be plotted\n"
                error_msg += "• Data needs to be cleaned before visualization"
                
                return self._create_error_chart(error_msg)
                
        # Line chart
        elif chart_type == 'line':
            if not x_col or not y_col:
                return self._create_error_chart("Both X and Y columns must be specified for line charts")
                
            if x_col not in df.columns or y_col not in df.columns:
                missing = []
                if x_col not in df.columns:
                    missing.append(f"X-axis column '{x_col}'")
                if y_col not in df.columns:
                    missing.append(f"Y-axis column '{y_col}'")
                return self._create_error_chart(f"Missing columns: {', '.join(missing)}")
            
            try:
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=title,
                    markers=True,
                    color_discrete_sequence=self.color_palette
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Line chart creation failed: {str(e)}")
                
        # Scatter plot
        elif chart_type == 'scatter':
            if not x_col or not y_col:
                return self._create_error_chart("Both X and Y columns must be specified for scatter plots")
                
            if x_col not in df.columns or y_col not in df.columns:
                missing = []
                if x_col not in df.columns:
                    missing.append(f"X-axis column '{x_col}'")
                if y_col not in df.columns:
                    missing.append(f"Y-axis column '{y_col}'")
                return self._create_error_chart(f"Missing columns: {', '.join(missing)}")
                
            size_col = chart_config.get('size')
            
            try:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    size=size_col if size_col and size_col in df.columns else None,
                    title=title,
                    color_discrete_sequence=self.color_palette
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Scatter plot creation failed: {str(e)}")
                
        # Pie chart
        elif chart_type == 'pie':
            # For pie charts we can use either x as names or specific 'names' param
            names_col = chart_config.get('names') or x_col
            values_col = chart_config.get('values') or y_col
            
            if not names_col:
                return self._create_error_chart("Category column (names) not specified")
                
            if names_col not in df.columns:
                return self._create_error_chart(f"Column '{names_col}' not found in data")
                
            try:
                if values_col and values_col in df.columns:
                    # If values column provided, use it
                    fig = px.pie(
                        df, 
                        names=names_col, 
                        values=values_col,
                        title=title,
                        color_discrete_sequence=self.color_palette
                    )
                else:
                    # Otherwise use counts
                    value_counts = df[names_col].value_counts().reset_index()
                    value_counts.columns = [names_col, 'count']
                    
                    fig = px.pie(
                        value_counts, 
                        names=names_col, 
                        values='count',
                        title=title,
                        color_discrete_sequence=self.color_palette
                    )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=500,
                    template='plotly_white',
                    uniformtext_minsize=12, 
                    uniformtext_mode='hide'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Pie chart creation failed: {str(e)}")
                
        # Histogram
        elif chart_type == 'histogram':
            x_col = chart_config.get('x')
            
            if not x_col:
                return self._create_error_chart("X-axis column not specified for histogram")
                
            if x_col not in df.columns:
                return self._create_error_chart(f"Column '{x_col}' not found in data")
                
            try:
                fig = px.histogram(
                    df, 
                    x=x_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=title,
                    color_discrete_sequence=self.color_palette
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Histogram creation failed: {str(e)}")
                
        # Box plot
        elif chart_type == 'box':
            if not x_col or not y_col:
                return self._create_error_chart("Both X and Y columns must be specified for box plots")
                
            if x_col not in df.columns or y_col not in df.columns:
                missing = []
                if x_col not in df.columns:
                    missing.append(f"X-axis column '{x_col}'")
                if y_col not in df.columns:
                    missing.append(f"Y-axis column '{y_col}'")
                return self._create_error_chart(f"Missing columns: {', '.join(missing)}")
                
            try:
                fig = px.box(
                    df, 
                    x=x_col, 
                    y=y_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    title=title,
                    color_discrete_sequence=self.color_palette
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Box plot creation failed: {str(e)}")
                
        # Heatmap
        elif chart_type == 'heatmap':
            # For heatmaps, we'll create a correlation matrix
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) < 2:
                return self._create_error_chart("At least two numeric columns are required for a correlation heatmap")
                
            try:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title=title or "Correlation Heatmap",
                    aspect="auto"
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white'
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Heatmap creation failed: {str(e)}")
                
        # Map visualization - simple default implementation
        elif chart_type == 'map':
            # Check for location columns
            lat_col = chart_config.get('lat') or next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
            lon_col = chart_config.get('lon') or next((col for col in df.columns if col.lower() in ['lon', 'lng', 'longitude']), None)
            
            if not lat_col or not lon_col:
                return self._create_error_chart("Latitude and longitude columns are required for map visualization")
                
            if lat_col not in df.columns or lon_col not in df.columns:
                missing = []
                if lat_col not in df.columns:
                    missing.append(f"Latitude column '{lat_col}'")
                if lon_col not in df.columns:
                    missing.append(f"Longitude column '{lon_col}'")
                return self._create_error_chart(f"Missing columns: {', '.join(missing)}")
                
            try:
                fig = px.scatter_mapbox(
                    df, 
                    lat=lat_col, 
                    lon=lon_col,
                    color=color_col if color_col and color_col in df.columns else None,
                    size=y_col if y_col and y_col in df.columns else None,
                    title=title,
                    mapbox_style="open-street-map",
                    color_discrete_sequence=self.color_palette,
                    zoom=3
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_white',
                    margin={"r": 0, "t": 30, "l": 0, "b": 0}
                )
                return fig
                
            except Exception as e:
                return self._create_error_chart(f"Map visualization failed: {str(e)}")
                
        # Fallback for unsupported chart types
        return self._create_error_chart(f"Chart type '{chart_type}' is not supported")
