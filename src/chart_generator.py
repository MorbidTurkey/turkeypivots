"""
Chart generation utilities using Plotly
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, List

class ChartGenerator:
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
            counts = df[x_col].value_counts().to_dict()
            if not counts:
                return f"No data available to count by {x_col}."
            top_items = sorted(counts.items(), key=lambda x: -x[1])[:5]
            summary = ', '.join([f"{k}: {v}" for k, v in top_items])
            return f"Top {x_col}s by count: {summary}."
        # Bar chart with y_col
        elif chart_type == 'bar' and x_col and y_col:
            if y_col in df.columns:
                top = df[[x_col, y_col]].sort_values(y_col, ascending=False).dropna().head(1)
                if not top.empty:
                    return f"The highest {y_col} by {x_col} is {top.iloc[0][x_col]} ({top.iloc[0][y_col]})."
            return f"This chart shows {y_col} by {x_col}."
        # Pie chart
        elif chart_type == 'pie' and chart_config.get('names'):
            names_col = chart_config['names']
            if names_col in df.columns:
                counts = df[names_col].value_counts().to_dict()
                top_items = sorted(counts.items(), key=lambda x: -x[1])[:3]
                summary = ', '.join([f"{k}: {v}" for k, v in top_items])
                return f"Top {names_col} categories: {summary}."
            return f"This pie chart shows the distribution of {names_col}."
        # Line chart
        elif chart_type == 'line' and x_col and y_col:
            return f"This line chart shows how {y_col} changes over {x_col}."
        # Scatter plot
        elif chart_type == 'scatter' and x_col and y_col:
            return f"This scatter plot shows the relationship between {x_col} and {y_col}."
        # Histogram
        elif chart_type == 'histogram' and x_col:
            return f"This histogram shows the distribution of {x_col}."
        # Box plot
        elif chart_type == 'box' and x_col:
            return f"This box plot shows the distribution and outliers for {x_col}."
        # Heatmap
        elif chart_type == 'heatmap':
            return "This heatmap shows the correlation or density between variables."
        # Map
        elif chart_type == 'map':
            return "This map shows geographic data distribution."
        # Fallback
        return title or "Chart generated."
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_chart(self, df: pd.DataFrame, chart_config: Dict[str, Any]) -> go.Figure:
        """
        Create a chart based on configuration
        
        Args:
            df: Data to visualize
            chart_config: Chart configuration dictionary
            
        Returns:
            plotly.graph_objects.Figure: Generated chart
        """
        chart_type = chart_config.get('type', 'bar')
        
        # Apply filters if specified
        filtered_df = self._apply_filters(df, chart_config.get('filters', {}))
        
        try:
            if chart_type == 'bar':
                return self._create_bar_chart(filtered_df, chart_config)
            elif chart_type == 'line':
                return self._create_line_chart(filtered_df, chart_config)
            elif chart_type == 'scatter':
                return self._create_scatter_plot(filtered_df, chart_config)
            elif chart_type == 'pie':
                return self._create_pie_chart(filtered_df, chart_config)
            elif chart_type == 'histogram':
                return self._create_histogram(filtered_df, chart_config)
            elif chart_type == 'box':
                return self._create_box_plot(filtered_df, chart_config)
            elif chart_type == 'heatmap':
                return self._create_heatmap(filtered_df, chart_config)
            elif chart_type == 'map':
                return self._create_map(filtered_df, chart_config)
            elif chart_type == 'violin':
                return self._create_violin_plot(filtered_df, chart_config)
            elif chart_type == 'sunburst':
                return self._create_sunburst(filtered_df, chart_config)
            elif chart_type == 'treemap':
                return self._create_treemap(filtered_df, chart_config)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
        
        except Exception as e:
            # Return error chart
            return self._create_error_chart(str(e))
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        if not filters:
            return df
        
        filtered_df = df.copy()

        for column, filter_config in filters.items():
            # Only apply filters for columns that exist in the DataFrame
            if column not in df.columns:
                # Ignore non-column filters like 'aggregation', 'sort', etc.
                continue

            if isinstance(filter_config, dict):
                if filter_config.get('exclude_missing'):
                    before_count = len(filtered_df)
                    filtered_df = filtered_df.dropna(subset=[column])
                    after_count = len(filtered_df)
                    print(f"Filter applied: Removed {before_count - after_count} rows with missing {column} values")

                if 'min_value' in filter_config:
                    min_val = filter_config['min_value']
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') >= min_val]

                if 'max_value' in filter_config:
                    max_val = filter_config['max_value']
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[column], errors='coerce') <= max_val]

                if 'values' in filter_config:
                    allowed_values = filter_config['values']
                    filtered_df = filtered_df[filtered_df[column].isin(allowed_values)]

        return filtered_df
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create bar chart with data validation and helpful error messages"""
        try:
            x_col = config.get('x')
            y_col = config.get('y')
            color_col = config.get('color')
            title = config.get('title', 'Bar Chart')
            
            if not x_col or not y_col:
                return self._create_error_chart("Missing x or y column for bar chart")
            
            # Handle count aggregation
            if y_col == 'count' or config.get('filters', {}).get('aggregation') == 'count':
                # Create count aggregation
                if x_col not in df.columns:
                    return self._create_error_chart(f"Column '{x_col}' not found in data")
                
                # Count occurrences by x_col
                count_data = df[x_col].value_counts().reset_index()
                count_data.columns = [x_col, 'count']
                
                fig = px.bar(count_data, x=x_col, y='count', title=title)
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
            
            # Data validation - don't fix, just report issues
            validation_issues = []
            
            # Check y-column data type
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                non_numeric_count = df[y_col].apply(lambda x: not str(x).replace('.', '').replace('-', '').isdigit() if pd.notna(x) else False).sum()
                if non_numeric_count > 0:
                    validation_issues.append(f"Y-axis column '{y_col}' contains {non_numeric_count} non-numeric values")
            
            # Check for missing values
            missing_x = df[x_col].isna().sum()
            missing_y = df[y_col].isna().sum()
            if missing_x > 0:
                validation_issues.append(f"X-axis column '{x_col}' has {missing_x} missing values")
            if missing_y > 0:
                validation_issues.append(f"Y-axis column '{y_col}' has {missing_y} missing values")
            
            # Check data size
            if len(df) == 0:
                return self._create_error_chart("No data available")
            
            # If there are validation issues, create an informative error chart
            if validation_issues:
                error_text = "Data Issues Found:\n\n" + "\n".join([f"• {issue}" for issue in validation_issues])
                error_text += "\n\nSuggestions:\n• Clean your data before uploading\n• Check for proper data types\n• Remove or fix missing values"
                
                return self._create_error_chart(error_text)
            
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
            
        except Exception as e:
            return self._create_error_chart(f"Error creating bar chart: {str(e)}")
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart with automatic aggregation (sum) for multiple entries per x/color group, sorted by x (chronological order if possible)."""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        title = config.get('title', f'{y} over {x}')
        
        # Get aggregation method from filters
        agg_method = config.get('filters', {}).get('aggregation', 'sum')
        print(f"Line chart using aggregation method: {agg_method}")

        group_cols = [x] if x else []
        if color:
            group_cols.append(color)
            
        if y and group_cols and y in df.columns:
            if agg_method == 'count':
                # For count aggregation
                grouped = df.groupby(group_cols, as_index=False).size()
                grouped.columns = [*group_cols, 'count']
                y = 'count'  # Update y to use the count column
            elif agg_method == 'mean':
                grouped = df.groupby(group_cols, as_index=False)[y].mean()
            elif agg_method == 'max':
                grouped = df.groupby(group_cols, as_index=False)[y].max()
            elif agg_method == 'min':
                grouped = df.groupby(group_cols, as_index=False)[y].min()
            else:  # Default to sum
                grouped = df.groupby(group_cols, as_index=False)[y].sum()
        else:
            grouped = df

        # Try to sort by x chronologically if possible
        if x in grouped.columns:
            try:
                grouped[x] = pd.to_datetime(grouped[x], errors='ignore')
                grouped = grouped.sort_values(x)
            except Exception:
                grouped = grouped.sort_values(x)

        fig = px.line(
            grouped, x=x, y=y, color=color,
            title=title,
            color_discrete_sequence=self.color_palette
        )

        fig.update_layout(
            template='plotly_white',
            height=400
        )

        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot"""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        size = config.get('size')
        title = config.get('title', f'{y} vs {x}')
        
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create pie chart with data validation"""
        try:
            names_col = config.get('names')
            values_col = config.get('values')
            title = config.get('title', 'Pie Chart')
            
            if not names_col:
                return self._create_error_chart("Pie chart requires a 'names' column (categories)")
            
            if names_col not in df.columns:
                return self._create_error_chart(f"Column '{names_col}' not found in data")
            
            validation_issues = []
            
            # Check for missing values in names column
            missing_names = df[names_col].isna().sum()
            if missing_names > 0:
                validation_issues.append(f"Names column '{names_col}' has {missing_names} missing values")
            
            # If values column specified, validate it
            if values_col:
                if values_col not in df.columns:
                    return self._create_error_chart(f"Values column '{values_col}' not found in data")
                
                if not pd.api.types.is_numeric_dtype(df[values_col]):
                    non_numeric_count = df[values_col].apply(lambda x: not str(x).replace('.', '').replace('-', '').isdigit() if pd.notna(x) else False).sum()
                    if non_numeric_count > 0:
                        validation_issues.append(f"Values column '{values_col}' contains {non_numeric_count} non-numeric values")
            
            if validation_issues:
                error_text = "Data Issues Found:\n\n" + "\n".join([f"• {issue}" for issue in validation_issues])
                error_text += "\n\nSuggestions:\n• Clean your data before uploading\n• Ensure values column contains only numbers"
                return self._create_error_chart(error_text)
            
            try:
                if values_col:
                    # Use specified values column
                    fig = px.pie(df, names=names_col, values=values_col, title=title,
                               color_discrete_sequence=self.color_palette)
                else:
                    # Use value counts of names column
                    value_counts = df[names_col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=title,
                               color_discrete_sequence=self.color_palette)
                
                fig.update_layout(template='plotly_white', height=500)
                return fig
                
            except Exception as px_error:
                error_msg = f"Pie chart creation failed: {str(px_error)}\n\n"
                error_msg += "This usually means:\n"
                error_msg += "• Data contains mixed or incompatible types\n"
                error_msg += "• Values cannot be aggregated properly\n"
                error_msg += "• Data needs to be cleaned before visualization"
                
                return self._create_error_chart(error_msg)
            
        except Exception as e:
            return self._create_error_chart(f"Error creating pie chart: {str(e)}")
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a line chart"""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        title = config.get('title', f'{y} over {x}')
        
        fig = px.line(
            df, x=x, y=y, color=color,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot"""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        size = config.get('size')
        title = config.get('title', f'{y} vs {x}')
        
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig

    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a histogram"""
        x = config.get('x')
        color = config.get('color')
        title = config.get('title', f'Distribution of {x}')
        
        fig = px.histogram(
            df, x=x, color=color,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a box plot"""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        title = config.get('title', f'Box Plot of {y}')
        
        fig = px.box(
            df, x=x, y=y, color=color,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a heatmap"""
        # Get numerical columns for correlation heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return self._create_error_chart("Not enough numerical columns for heatmap")
        
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title=config.get('title', 'Correlation Heatmap'),
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_map(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a map visualization"""
        lat = config.get('lat')
        lon = config.get('lon')
        color = config.get('color')
        size = config.get('size')
        hover_name = config.get('hover_name')
        title = config.get('title', 'Map Visualization')
        
        # Check if we have location data
        if not (lat and lon):
            # Try to find location columns
            location_cols = [col for col in df.columns if any(word in col.lower() for word in ['lat', 'lon', 'lng', 'latitude', 'longitude'])]
            if len(location_cols) >= 2:
                lat = location_cols[0]
                lon = location_cols[1]
            else:
                return self._create_error_chart("No latitude/longitude columns found for map")
        
        fig = px.scatter_mapbox(
            df, lat=lat, lon=lon, color=color, size=size,
            hover_name=hover_name,
            title=title,
            mapbox_style='open-street-map',
            height=400,
            zoom=3
        )
        
        fig.update_layout(
            template='plotly_white'
        )
        
        return fig
    
    def _create_violin_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a violin plot"""
        x = config.get('x')
        y = config.get('y')
        color = config.get('color')
        title = config.get('title', f'Violin Plot of {y}')
        
        fig = px.violin(
            df, x=x, y=y, color=color,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_sunburst(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a sunburst chart"""
        path = config.get('path', [])
        values = config.get('values')
        title = config.get('title', 'Sunburst Chart')
        
        if not path:
            return self._create_error_chart("Path columns required for sunburst chart")
        
        fig = px.sunburst(
            df, path=path, values=values,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_treemap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a treemap"""
        path = config.get('path', [])
        values = config.get('values')
        title = config.get('title', 'Treemap')
        
        if not path:
            return self._create_error_chart("Path columns required for treemap")
        
        fig = px.treemap(
            df, path=path, values=values,
            title=title,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an informative error chart with helpful messages"""
        fig = go.Figure()
        
        # Split long messages into multiple lines for better readability
        lines = error_message.split('\n')
        y_positions = [0.7 - i*0.08 for i in range(len(lines))]
        
        for i, line in enumerate(lines):
            if i < len(y_positions) and line.strip():  # Skip empty lines
                font_size = 14 if i == 0 else 12  # Larger font for first line
                font_weight = "bold" if line.startswith("Data Issues") or line.startswith("Suggestions") else "normal"
                color = "darkred" if "Error" in line or "failed" in line else "darkblue"
                
                fig.add_annotation(
                    text=line.strip(),
                    xref="paper", yref="paper",
                    x=0.5, y=y_positions[i],
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=font_size, color=color, weight=font_weight)
                )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(245,245,245,0.8)',
            paper_bgcolor='rgba(245,245,245,0.8)',
            height=400,
            title="Data Validation Issue"
        )
        
        return fig
    
    def suggest_charts(self, df: pd.DataFrame, column_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Suggest appropriate chart types based on data characteristics
        
        Args:
            df: DataFrame to analyze
            column_types: Dictionary of column semantic types
            
        Returns:
            List of suggested chart configurations
        """
        suggestions = []
        
        numerical_cols = [col for col, dtype in column_types.items() if 'numerical' in dtype]
        categorical_cols = [col for col, dtype in column_types.items() if dtype == 'categorical']
        datetime_cols = [col for col, dtype in column_types.items() if dtype == 'datetime']
        
        # Bar charts for categorical vs numerical
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                suggestions.append({
                    'type': 'bar',
                    'x': cat_col,
                    'y': num_col,
                    'title': f'{num_col} by {cat_col}',
                    'description': f'Compare {num_col} across different {cat_col} categories'
                })
        
        # Line charts for time series
        for date_col in datetime_cols:
            for num_col in numerical_cols:
                suggestions.append({
                    'type': 'line',
                    'x': date_col,
                    'y': num_col,
                    'title': f'{num_col} over time',
                    'description': f'Show how {num_col} changes over time'
                })
        
        # Scatter plots for numerical vs numerical
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                suggestions.append({
                    'type': 'scatter',
                    'x': col1,
                    'y': col2,
                    'title': f'{col2} vs {col1}',
                    'description': f'Explore relationship between {col1} and {col2}'
                })
        
        # Histograms for distributions
        for num_col in numerical_cols:
            suggestions.append({
                'type': 'histogram',
                'x': num_col,
                'title': f'Distribution of {num_col}',
                'description': f'Show the distribution of {num_col} values'
            })
        
        # Pie charts for categorical distributions
        for cat_col in categorical_cols:
            if df[cat_col].nunique() <= 10:  # Only for reasonable number of categories
                suggestions.append({
                    'type': 'pie',
                    'names': cat_col,
                    'title': f'Distribution of {cat_col}',
                    'description': f'Show the proportion of each {cat_col} category'
                })
        
        return suggestions[:10]  # Return top 10 suggestions
