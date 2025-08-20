"""
Turkey Pivots - Data Visualization Dashboard
Main application file with improved layout and functionality
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import datetime
import traceback
import plotly.graph_objects as go
import json
import os
import sys
import uuid
import traceback
from datetime import datetime
import base64
import io

# --- Import project modules ---
from src.data_processor import DataProcessor
from src.chart_generator_fix import ChartGenerator
from src.ai_assistant import ChartRequest
from src.database import DatabaseManager
from src.simple_ai_assistant import SimpleAIAssistant as AIAssistant

# --- Bootstrap Theme Configuration ---
BOOTSTRAP_THEMES = [
    "BOOTSTRAP", "CERULEAN", "COSMO", "CYBORG", "DARKLY", "FLATLY", "JOURNAL", "LITERA", 
    "LUMEN", "LUX", "MATERIA", "MINTY", "MORPH", "PULSE", "QUARTZ", "SANDSTONE", 
    "SIMPLEX", "SKETCHY", "SLATE", "SOLAR", "SPACELAB", "SUPERHERO", "UNITED", "VAPOR", "YETI", "ZEPHYR"
]
DEFAULT_THEME = "FLATLY"

def get_theme_url(theme_name):
    """Convert theme name to Bootstrap theme URL"""
    return getattr(dbc.themes, theme_name.upper(), dbc.themes.FLATLY)


# --- Initialize Dash App ---
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Turkey Pivots - Data Visualization Dashboard"

# Expose server for WSGI (Gunicorn)
server = app.server

# --- Initialize Components ---
data_processor = DataProcessor()
chart_generator = ChartGenerator()
db_manager = DatabaseManager()

# Initialize AI Assistant - Try OpenAI first, fallback to SimpleAI
try:
    from src.openai_assistant import OpenAIAssistant
    ai_assistant = OpenAIAssistant(model_name="gpt-4o-mini")
    print("Using OpenAI AI Assistant with gpt-4o-mini")
except (ImportError, ValueError) as e:
    # Fallback to SimpleAIAssistant
    from src.simple_ai_assistant import SimpleAIAssistant as AIAssistant
    ai_assistant = AIAssistant(model_name="gpt-oss:20b")
    print(f"Using Ollama AI Assistant (reason for OpenAI fallback: {str(e)})")

# --- App Layout ---
app.layout = dbc.Container([
    # --- Storage Components ---
    dcc.Store(id='session-data', storage_type='session'),  # Main session data
    dcc.Store(id='column-metadata', storage_type='session'),  # Column descriptions & settings
    dcc.Store(id='chart-library', storage_type='session', data=[]),  # All generated charts
    dcc.Store(id='theme-store', storage_type='local', data=DEFAULT_THEME),  # UI theme
    
    # --- Window States ---
    dcc.Store(id='chart-1-idx', storage_type='session', data=0),
    dcc.Store(id='chart-2-idx', storage_type='session', data=0),
    dcc.Store(id='chart-3-idx', storage_type='session', data=0),
    dcc.Store(id='chart-4-idx', storage_type='session', data=0),
    dcc.Store(id='chart-1-table', storage_type='session', data=False),
    dcc.Store(id='chart-2-table', storage_type='session', data=False),
    dcc.Store(id='chart-3-table', storage_type='session', data=False),
    dcc.Store(id='chart-4-table', storage_type='session', data=False),
    dcc.Store(id='chart-1-collapsed', storage_type='session', data=False),
    dcc.Store(id='chart-2-collapsed', storage_type='session', data=False),
    dcc.Store(id='chart-3-collapsed', storage_type='session', data=False),
    dcc.Store(id='chart-4-collapsed', storage_type='session', data=False),
    
    # --- Hidden Elements for Callbacks ---
    html.Div([
        html.Button(id=f'prev-{i}', style={'display': 'none'}) for i in range(1, 5)
    ] + [
        html.Button(id=f'next-{i}', style={'display': 'none'}) for i in range(1, 5)
    ] + [
        html.Button(id=f'table-{i}-toggle', style={'display': 'none'}) for i in range(1, 5)
    ] + [
        html.Button(id=f'collapse-{i}-toggle', style={'display': 'none'}) for i in range(1, 5)
    ] + [
        html.Div(id=f'chart-{i}-container', style={'display': 'none'}) for i in range(1, 5)
    ] + [
        html.Button(id=f'download-{i}-excel', style={'display': 'none'}) for i in range(1, 5)
    ], style={'display': 'none'}),
    
    # --- Header ---
    dbc.Row([
        dbc.Col([
            html.H1("Turkey Pivots", className="text-primary mb-2"),
            html.P("AI-powered data visualization and analysis", className="text-muted")
        ], width=8),
        dbc.Col([
            html.Div([
                html.Label("Theme:", className="me-2"),
                dcc.Dropdown(
                    id='theme-dropdown',
                    options=[{"label": t.title(), "value": t} for t in BOOTSTRAP_THEMES],
                    value=DEFAULT_THEME,
                    clearable=False,
                    style={"width": "200px"}
                )
            ], className="d-flex align-items-center justify-content-end")
        ], width=4, className="d-flex align-items-center")
    ], className="mb-4 pb-2 border-bottom"),
    
    # --- Main Content Area (different pages) ---
    html.Div(id='main-content'),
    
    # --- Footer ---
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Turkey Pivots © 2025 | AI-Powered Data Visualization", 
                   className="text-center text-muted small")
        ])
    ]),
    
    # --- Download Components ---
    dcc.Download(id='download-excel'),
    
    # --- Theme Reload Placeholder ---
    html.Div(id='dummy-theme-reload', style={'display': 'none'})
    
], fluid=True, className="px-2 px-sm-3 px-md-4 px-lg-5 px-xl-5")

# --- Page Creation Functions ---

def create_upload_page():
    """Create the initial file upload page"""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("Upload Your Data", className="text-center")),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.Div([
                                html.H4("Drag & Drop or Select a File"),
                                html.P("Support for CSV, Excel files (.xlsx, .xls)")
                            ])
                        ], className="d-flex align-items-center justify-content-center"),
                        style={
                            'width': '100%',
                            'height': '200px',
                            'lineHeight': '60px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'cursor': 'pointer'
                        },
                        multiple=False,
                        accept='.csv,.xlsx,.xls'
                    ),
                    html.Div(id='upload-output', className="mt-3")
                ])
            ], className="shadow-sm")
        ], width={"size": 8, "offset": 2})
    ])

def _determine_column_type(col_name, series=None):
    """Determine the most likely data type for a column"""
    # Check column name for hints
    col_lower = col_name.lower()
    
    # Check for date-related column names
    if any(date_hint in col_lower for date_hint in ['date', 'time', 'year', 'month', 'day']):
        return 'datetime'
    
    # Check for numeric-related column names
    if any(num_hint in col_lower for num_hint in ['price', 'sales', 'amount', 'qty', 'cost', 'revenue', 'number', 'count', 'total']):
        return 'numeric'
        
    # If we have data, check its type
    if series is not None:
        # Check if it's datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
            
        # Check if it's numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
            
        # Try to convert to datetime - some string columns might be dates
        try:
            # Drop NaNs to avoid conversion errors
            non_null = series.dropna()
            if len(non_null) > 0:
                # Check a few sample values for date patterns
                sample = non_null.head(5).astype(str)
                date_pattern = r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}'
                if sample.str.match(date_pattern).any():
                    return 'datetime'
        except:
            pass  # Continue if date detection fails
            
        # Check if it's categorical/text
        unique_pct = series.nunique() / len(series) if len(series) > 0 else 0
        
        # If low cardinality relative to row count, likely categorical
        if unique_pct < 0.5:  # Adjusted threshold
            return 'categorical'
            
        # If high cardinality, likely text data
        if unique_pct > 0.8:
            return 'text'
            
    # Default to categorical for unknown types
    return 'categorical'

def create_column_configuration(columns, sample_data=None):
    """Create column configuration page with descriptions and renaming options"""
    column_cards = []
    
    for col in columns:
        # Try to determine column type from data
        sample = None if sample_data is None else sample_data[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(x) for x in sample]) if sample else "No data available"
        
        missing_vals = 0 if sample_data is None else sample_data[col].isna().sum()
        missing_pct = 0 if sample_data is None else (missing_vals / len(sample_data)) * 100
        
        # Display missing data status
        if missing_vals == 0:
            missing_badge = html.Span("No Missing Values", className="badge bg-success")
        elif missing_pct < 5:
            missing_badge = html.Span(f"{missing_vals} Missing ({missing_pct:.1f}%)", className="badge bg-warning")
        else:
            missing_badge = html.Span(f"{missing_vals} Missing ({missing_pct:.1f}%)", className="badge bg-danger")
        
        column_cards.append(
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H5(col, className="mb-0"),
                        missing_badge
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Display Name", className="fw-bold"),
                            dbc.Input(
                                id={'type': 'column-name', 'index': col},
                                type="text",
                                value=col,
                                placeholder="Enter display name"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Data Type", className="fw-bold"),
                            dcc.Dropdown(
                                id={'type': 'column-type', 'index': col},
                                options=[
                                    {'label': 'Numeric', 'value': 'numeric'},
                                    {'label': 'Categorical', 'value': 'categorical'},
                                    {'label': 'Date/Time', 'value': 'datetime'},
                                    {'label': 'Text', 'value': 'text'},
                                    {'label': 'Ignore', 'value': 'ignore'}
                                ],
                                value=_determine_column_type(col, sample_data[col] if sample_data is not None else None),
                                clearable=False
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Description", className="fw-bold"),
                            dbc.Textarea(
                                id={'type': 'column-desc', 'index': col},
                                placeholder="Describe what this data represents...",
                                style={"height": "80px"}
                            )
                        ], width=12)
                    ]),
                    
                    html.Div([
                        html.Label("Sample Values:", className="fw-bold mt-3"),
                        html.P(sample_str, className="text-muted small")
                    ])
                ])
            ], className="mb-3")
        )
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Column Configuration", className="mb-4"),
                html.P([
                    "Customize your data columns to improve analysis and visualization. ",
                    "Add descriptions and rename columns as they'll appear in charts."
                ], className="lead mb-4"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(column_cards, id="column-cards-container", className="col-config")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("Continue to Dashboard", id="continue-btn", color="primary", size="lg", className="mt-4 mb-4")
            ], className="text-center")
        ])
    ])

def create_dashboard_page():
    """Create main dashboard with AI chat and visualization windows"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Button("← Back to Data Configuration", id="back-to-config-btn", color="secondary", size="sm"),
                html.H2("Dashboard", className="mt-3 mb-4")
            ])
        ]),
        
        # AI Assistant Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("AI Assistant", className="mb-0"),
                        dbc.Button(
                            "Pivot Builder", 
                            id="open-pivot-builder-btn", 
                            color="primary", 
                            size="sm",
                            className="float-end"
                        )
                    ]),
                    dbc.CardBody([
                        html.Div([
                            html.P(
                                "Ask questions or request charts based on your data. Example: 'Show me sales trends by month' or 'What are the top 5 products by revenue?'", 
                                className="text-muted mb-3"
                            ),
                            html.Div(id='chat-history', style={'height': '300px', 'overflow-y': 'auto'}),
                            dbc.InputGroup([
                                dbc.Input(id='chat-input', placeholder="Ask about your data or request a visualization..."),
                                dbc.Button("Send", id='send-btn', color="primary")
                            ], className="mt-3")
                        ])
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Visualization Windows (2x2 grid)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span(id='chart-1-title', children="Visual 1"),
                            dbc.ButtonGroup([
                                dbc.Button("◀", id='prev-1', size="sm", className="btn-sm"),
                                dbc.Button("▶", id='next-1', size="sm", className="btn-sm"),
                                dbc.Button(
                                    "Table", 
                                    id='table-1-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Toggle Table View"
                                ),
                                dbc.Button(
                                    "Excel", 
                                    id='download-1-excel', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Download as Excel"
                                ),
                                dbc.Button(
                                    "▲/▼", 
                                    id='collapse-1-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Collapse/Expand"
                                )
                            ], className="float-end")
                        ], className="d-flex justify-content-between")
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.Div(id='chart-1-container', style={'min-height': '400px', 'height': 'auto'})
                        ], style={'padding': '10px', 'overflow': 'visible'}),
                        id="collapse-1",
                        is_open=True
                    )
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span(id='chart-2-title', children="Visual 2"),
                            dbc.ButtonGroup([
                                dbc.Button("◀", id='prev-2', size="sm", className="btn-sm"),
                                dbc.Button("▶", id='next-2', size="sm", className="btn-sm"),
                                dbc.Button(
                                    "Table", 
                                    id='table-2-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Toggle Table View"
                                ),
                                dbc.Button(
                                    "Excel", 
                                    id='download-2-excel', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Download as Excel"
                                ),
                                dbc.Button(
                                    "▲/▼", 
                                    id='collapse-2-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Collapse/Expand"
                                )
                            ], className="float-end")
                        ], className="d-flex justify-content-between")
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.Div(id='chart-2-container', style={'min-height': '400px', 'height': 'auto'})
                        ], style={'padding': '10px', 'overflow': 'visible'}),
                        id="collapse-2",
                        is_open=True
                    )
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span(id='chart-3-title', children="Visual 3"),
                            dbc.ButtonGroup([
                                dbc.Button("◀", id='prev-3', size="sm", className="btn-sm"),
                                dbc.Button("▶", id='next-3', size="sm", className="btn-sm"),
                                dbc.Button(
                                    "Table", 
                                    id='table-3-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Toggle Table View"
                                ),
                                dbc.Button(
                                    "Excel", 
                                    id='download-3-excel', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Download as Excel"
                                ),
                                dbc.Button(
                                    "▲/▼", 
                                    id='collapse-3-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Collapse/Expand"
                                )
                            ], className="float-end")
                        ], className="d-flex justify-content-between")
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.Div(id='chart-3-container', style={'min-height': '400px', 'height': 'auto'})
                        ], style={'padding': '10px', 'overflow': 'visible'}),
                        id="collapse-3",
                        is_open=True
                    )
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span(id='chart-4-title', children="Visual 4"),
                            dbc.ButtonGroup([
                                dbc.Button("◀", id='prev-4', size="sm", className="btn-sm"),
                                dbc.Button("▶", id='next-4', size="sm", className="btn-sm"),
                                dbc.Button(
                                    "Table", 
                                    id='table-4-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Toggle Table View"
                                ),
                                dbc.Button(
                                    "Excel", 
                                    id='download-4-excel', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Download as Excel"
                                ),
                                dbc.Button(
                                    "▲/▼", 
                                    id='collapse-4-toggle', 
                                    size="sm",
                                    className="btn-sm ms-1",
                                    title="Collapse/Expand"
                                )
                            ], className="float-end")
                        ], className="d-flex justify-content-between")
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            html.Div(id='chart-4-container', style={'min-height': '400px', 'height': 'auto'})
                        ], style={'padding': '10px', 'overflow': 'visible'}),
                        id="collapse-4",
                        is_open=True
                    )
                ])
            ], width=6)
        ]),
        
        # Pivot Builder Modal
        dbc.Modal([
            dbc.ModalHeader("Custom Chart Builder"),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Chart Type"),
                        dcc.Dropdown(
                            id='chart-type-dropdown',
                            options=[
                                {'label': 'Bar Chart', 'value': 'bar'},
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Pie Chart', 'value': 'pie'},
                                {'label': 'Histogram', 'value': 'histogram'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Heatmap', 'value': 'heatmap'}
                            ],
                            placeholder="Select chart type"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Chart Title"),
                        dbc.Input(id='chart-title-input', placeholder="Enter a title for your chart")
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("X-Axis (Categories)"),
                        dcc.Dropdown(id='x-axis-dropdown')
                    ], width=6),
                    dbc.Col([
                        html.Label("Y-Axis (Values)"),
                        dcc.Dropdown(id='y-axis-dropdown')
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Color By"),
                        dcc.Dropdown(id='color-dropdown')
                    ], width=6),
                    dbc.Col([
                        html.Label("Aggregation"),
                        dcc.Dropdown(
                            id='agg-dropdown',
                            options=[
                                {'label': 'Sum', 'value': 'sum'},
                                {'label': 'Average', 'value': 'mean'},
                                {'label': 'Count', 'value': 'count'},
                                {'label': 'Minimum', 'value': 'min'},
                                {'label': 'Maximum', 'value': 'max'}
                            ],
                            value='sum'
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Filter (Optional)"),
                        dbc.Row([
                            dbc.Col(dcc.Dropdown(id='filter-column-dropdown', placeholder="Select column"), width=4),
                            dbc.Col(dcc.Dropdown(
                                id='filter-operation-dropdown',
                                options=[
                                    {'label': 'Equals', 'value': 'eq'},
                                    {'label': 'Not Equals', 'value': 'ne'},
                                    {'label': 'Greater Than', 'value': 'gt'},
                                    {'label': 'Less Than', 'value': 'lt'},
                                    {'label': 'In List', 'value': 'in'},
                                ],
                                placeholder="Operation"
                            ), width=3),
                            dbc.Col(dbc.Input(id='filter-value-input', placeholder="Value"), width=5)
                        ])
                    ], width=12)
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="close-pivot-builder-btn", className="me-2", color="secondary"),
                dbc.Button("Create Chart", id="create-chart-btn", color="primary")
            ])
        ], id="pivot-builder-modal", size="lg"),
    ])

def extract_data_from_figure(fig):
    """Extract data from plotly figure into pandas DataFrame"""
    if not fig or not hasattr(fig, 'data') or not fig.data:
        print("Warning: Figure has no data")
        return None
        
    all_data = []
    
    # Check if it's a count-based bar chart by examining axis titles
    is_count_chart = False
    
    # Log the figure type for debugging
    print(f"Extracting data from figure of type: {type(fig)}")
    
    if hasattr(fig, 'layout') and hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
        if hasattr(fig.layout.yaxis.title, 'text'):
            y_title = fig.layout.yaxis.title.text
            print(f"Y-axis title: {y_title}")
            if y_title == 'Count':
                is_count_chart = True
    
    for trace_idx, trace in enumerate(fig.data):
        trace_data = {}
        
        if hasattr(trace, 'x') and trace.x is not None:
            trace_data['x'] = trace.x
        if hasattr(trace, 'y') and trace.y is not None:
            trace_data['y'] = trace.y
        if hasattr(trace, 'name') and trace.name is not None:
            trace_data['series'] = trace.name
        else:
            trace_data['series'] = f"Series {trace_idx+1}"
            
        # Add other possible data elements
        if hasattr(trace, 'z') and trace.z is not None:
            trace_data['z'] = trace.z
        if hasattr(trace, 'text') and trace.text is not None:
            trace_data['text'] = trace.text
            
        # Get trace type for special handling
        trace_type = getattr(trace, 'type', None)
        print(f"Processing trace of type: {trace_type}")
            
        # For each trace create a row or multiple rows
        if trace_data:
            if 'x' in trace_data and 'y' in trace_data:
                # Convert to rows
                for i in range(len(trace_data['x'])):
                    if i < len(trace_data['y']):  # Ensure y value exists
                        row = {'x': trace_data['x'][i], 'y': trace_data['y'][i], 'series': trace_data['series']}
                        
                        # Clean up values for better Excel formatting
                        # Handle dates, numbers and other types properly
                        if isinstance(row['x'], (datetime.date, datetime.datetime)):
                            row['x'] = row['x'].strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Always add Count column for compatibility
                        row['Count'] = row['y']
                        
                        # If this is a count-based chart, also add 'count' column (lowercase)
                        if is_count_chart or trace.type == 'bar':
                            row['count'] = row['y']
                            
                        all_data.append(row)
            else:
                # Add trace data as is
                all_data.append(trace_data)
    
    # Combine all data into a single dataframe
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Always add these columns for compatibility with different code paths
        if 'y' in df.columns:
            # Add uppercase Count column
            if 'Count' not in df.columns:
                df['Count'] = df['y']
                
            # Add lowercase count column 
            if 'count' not in df.columns:
                df['count'] = df['y']
                
            # Add an aggregation column
            df['aggregation'] = 'count' if is_count_chart else 'sum'
            
        return df
    return None

# --- Helper Functions ---

def create_chart_config(chart_request, df, column_metadata=None):
    """Create appropriate chart configuration based on chart type and available data"""
    chart_type = chart_request.chart_type
    
    # Get column display names from metadata if available
    x_display = chart_request.x_column
    y_display = chart_request.y_column
    color_display = chart_request.color_column
    
    # Check for aggregation requests
    agg_func = None
    if chart_request.filters and 'aggregation' in chart_request.filters:
        agg_func = chart_request.filters['aggregation']
        print(f"Found aggregation in filters: {agg_func}")
    
    is_count_aggregation = (chart_request.y_column == 'count' or 
                          (chart_request.filters and chart_request.filters.get('aggregation') == 'count'))
    
    # For count aggregations, handle display names properly
    if is_count_aggregation:
        y_display = 'Count'
        print("Using Count as y_display for count aggregation")
    
    if column_metadata:
        if chart_request.x_column and chart_request.x_column in column_metadata:
            x_display = column_metadata[chart_request.x_column].get('display_name', chart_request.x_column)
        if chart_request.y_column and chart_request.y_column in column_metadata and not is_count_aggregation:
            y_display = column_metadata[chart_request.y_column].get('display_name', chart_request.y_column)
        if chart_request.color_column and chart_request.color_column in column_metadata:
            color_display = column_metadata[chart_request.color_column].get('display_name', chart_request.color_column)
    
    if chart_type == 'pie':
        # Pie charts need 'names' and 'values' instead of x/y
        names_col = chart_request.x_column  # Category column
        values_col = chart_request.y_column  # Value column
        
        # If no y_column specified, use value counts of x_column
        if not values_col and names_col:
            return {
                'type': 'pie',
                'names': names_col,
                'names_display': x_display,
                'values': None,  # Will use value counts
                'title': chart_request.title or f"Distribution of {x_display}",
                'filters': chart_request.filters
            }
        else:
            return {
                'type': 'pie',
                'names': names_col,
                'names_display': x_display,
                'values': values_col,
                'values_display': y_display,
                'title': chart_request.title or f"{y_display} by {x_display}",
                'filters': chart_request.filters
            }
    
    elif chart_type == 'histogram':
        # Histograms only need x column (the distribution variable)
        return {
            'type': 'histogram',
            'x': chart_request.x_column or chart_request.y_column,  # Use whichever is available
            'x_display': x_display or y_display,
            'color': chart_request.color_column,
            'color_display': color_display,
            'title': chart_request.title or f"Distribution of {x_display or y_display}",
            'filters': chart_request.filters
        }
    
    elif chart_type == 'heatmap':
        # Heatmaps typically show correlation between numerical columns
        return {
            'type': 'heatmap',
            'title': chart_request.title or "Correlation Heatmap",
            'filters': chart_request.filters
        }
    
    elif chart_type == 'scatter':
        # Scatter plots can have size parameter
        return {
            'type': 'scatter',
            'x': chart_request.x_column,
            'x_display': x_display,
            'y': chart_request.y_column,
            'y_display': y_display,
            'color': chart_request.color_column,
            'color_display': color_display,
            'size': chart_request.size_column if hasattr(chart_request, 'size_column') else None,
            'title': chart_request.title or f"{y_display} vs {x_display}",
            'filters': chart_request.filters
        }
    
    else:
        # Standard x/y charts (bar, line, box)
        return {
            'type': chart_type,
            'x': chart_request.x_column,
            'x_display': x_display,
            'y': chart_request.y_column,
            'y_display': y_display,
            'color': chart_request.color_column,
            'color_display': color_display,
            'title': chart_request.title or f"{chart_type.title()} Chart: {y_display} by {x_display}",
            'filters': chart_request.filters
        }

def register_window_callbacks(app):
    """Register all callbacks for chart windows"""
    for i in range(1, 5):
        # Chart/table display
        @app.callback(
            Output(f'chart-{i}-container', 'children'),
            [Input(f'chart-{i}-idx', 'data'),
             Input(f'chart-library', 'data'),
             Input(f'chart-{i}-table', 'data')],
            prevent_initial_call=True
        )
        def update_chart_window(idx, chart_library, table_view, window=i):
            import pandas as pd
            if not chart_library or idx is None or idx < 0 or idx >= len(chart_library):
                return html.Div("No chart available.", className="text-center text-muted p-5")
            
            entry = chart_library[idx]
            config = entry.get('config')
            title = entry.get('title', '')
            fig = entry.get('figure')
            insight = entry.get('insight', None)

            # Update window title
            app.clientside_callback(
                """function(title) { 
                    return title;
                }""",
                Output(f'chart-{window}-title', 'children', allow_duplicate=True),
                [Input({'index': window, 'chart_title': title}, 'id')],
                prevent_initial_call=True
            )

            # Table view
            if table_view:
                try:
                    # Extract data from figure
                    df = extract_data_from_figure(fig)
                    
                    if df is not None:
                        return html.Div([
                            html.P(insight, className="text-muted mb-3 small") if insight else None,
                            dash_table.DataTable(
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.to_dict('records'),
                                filter_action="native",
                                sort_action="native",
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'minWidth': '80px'
                                },
                                style_header={
                                    'backgroundColor': '#f8f9fa',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[{
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }]
                            ),
                            html.Div([
                                html.Hr(),
                                html.Button("Back to Chart", id={'type': f'table-back-{window}', 'index': 0}, 
                                            n_clicks=0, className="btn btn-outline-secondary btn-sm")
                            ], className="mt-3")
                        ])
                    else:
                        return html.Div([
                            html.P("Table view is not available for this chart type.", className="text-danger"),
                            html.Button("Back to Chart", id={'type': f'table-back-{window}', 'index': 0}, 
                                        n_clicks=0, className="btn btn-outline-secondary btn-sm mt-3")
                        ])
                except Exception as e:
                    print(f"Error creating table view: {str(e)}")
                    return html.Div([
                        html.P(f"Error creating table view: {str(e)}", className="text-danger"),
                        html.Button("Back to Chart", id={'type': f'table-back-{window}', 'index': 0}, 
                                    n_clicks=0, className="btn btn-outline-secondary btn-sm mt-3")
                    ])
            
            # Chart view
            return html.Div([
                dcc.Graph(figure=fig, className="chart-graph"),
                html.P(insight, className="text-muted mt-2 small") if insight else None
            ], className="chart-window")

        # Arrow buttons
        @app.callback(
            Output(f'chart-{i}-idx', 'data'),
            [Input(f'prev-{i}', 'n_clicks'),
             Input(f'next-{i}', 'n_clicks'),
             Input('chart-library', 'data')],
            [State(f'chart-{i}-idx', 'data')],
            prevent_initial_call=True
        )
        def update_chart_idx(prev, next_, chart_library, current_idx, window=i):
            ctx = dash.callback_context
            n = len(chart_library) if chart_library else 0
            
            if n == 0:
                return 0
                
            # If no current index, use 0
            if current_idx is None:
                current_idx = 0
                
            # Ensure current index is valid
            current_idx = max(0, min(current_idx, n-1))
                
            if ctx.triggered and ctx.triggered[0]['prop_id'].startswith(f'prev-{window}'):
                return (current_idx - 1) % n
            elif ctx.triggered and ctx.triggered[0]['prop_id'].startswith(f'next-{window}'):
                return (current_idx + 1) % n
            
            return current_idx

        # Table view toggle
        @app.callback(
            Output(f'chart-{i}-table', 'data'),
            [Input(f'table-{i}-toggle', 'n_clicks'),
             Input({'type': f'table-back-{i}', 'index': ALL}, 'n_clicks')],
            [State(f'chart-{i}-table', 'data'),
             State(f'chart-{i}-idx', 'data'),
             State('chart-library', 'data')],
            prevent_initial_call=True
        )
        def toggle_table_view(toggle_click, back_click_list, current_state, chart_idx, chart_library, window=i):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_state or False
                
            trigger_id = ctx.triggered[0]['prop_id']
            print(f"Table view toggle triggered by: {trigger_id}")
            
            # Check if any back button was clicked
            back_clicked = False
            if back_click_list and any(click for click in back_click_list if click):
                print(f"Back button clicked for window {window}")
                back_clicked = True
                
            if 'table-back' in trigger_id or back_clicked:
                print(f"Switching to chart view in window {window}")
                return False
            elif 'table-toggle' in trigger_id:
                # Print info about toggle for debugging
                if chart_library and chart_idx is not None and isinstance(chart_idx, int) and 0 <= chart_idx < len(chart_library):
                    chart_entry = chart_library[chart_idx]
                    print(f"Toggling to table view in window {window} for chart: {chart_entry.get('title', 'Unnamed')}")
                else:
                    print(f"Warning: Invalid chart_idx {chart_idx} for window {window}, chart_library length: {len(chart_library) if chart_library else 0}")
                
                # Toggle the current state
                return not current_state
                
            return current_state or False

        # Collapse/Expand toggle
        @app.callback(
            [Output(f'collapse-{i}', 'is_open'),
             Output(f'chart-{i}-collapsed', 'data')],
            [Input(f'collapse-{i}-toggle', 'n_clicks')],
            [State(f'collapse-{i}', 'is_open'),
             State(f'chart-{i}-collapsed', 'data')],
            prevent_initial_call=True
        )
        def toggle_collapse(n_clicks, is_open, collapsed, window=i):
            if n_clicks:
                return not is_open, not collapsed
            return is_open, collapsed

# --- Main Page Routing ---

@app.callback(
    Output('main-content', 'children'),
    Input('session-data', 'data')
)
def update_main_content(session_data):
    """Route to different pages based on session data"""
    if not session_data:
        return create_upload_page()
    elif 'columns' in session_data and 'configured' not in session_data:
        # Try to load sample data for column config
        sample_data = None
        try:
            sample_data = data_processor.load_saved_data(session_data['file_path'])
        except:
            pass
        return create_column_configuration(session_data['columns'], sample_data)
    elif 'configured' in session_data:
        return create_dashboard_page()
    else:
        return create_upload_page()

# --- File Upload Callback ---

@app.callback(
    [Output('session-data', 'data'), Output('upload-output', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('session-data', 'data')
)
def handle_file_upload(contents, filename, session_data):
    """Process uploaded file"""
    if contents is not None:
        try:
            print(f"Processing file: {filename}")
            # Process uploaded file
            df = data_processor.process_upload(contents, filename)
            columns = list(df.columns)
            print(f"Successfully processed {filename} with columns: {columns}")
            
            # Save data temporarily
            session_id = str(uuid.uuid4())
            file_path = f"temp/{session_id}.parquet"
            os.makedirs("temp", exist_ok=True)
            
            # Try to save as parquet, fallback to pickle if needed
            try:
                df.to_parquet(file_path)
            except Exception as parquet_error:
                print(f"Parquet save failed, using pickle: {parquet_error}")
                file_path = f"temp/{session_id}.pkl"
                df.to_pickle(file_path)
            
            session_data = session_data or {}
            session_data.update({
                'session_id': session_id,
                'filename': filename,
                'columns': columns,
                'file_path': file_path
            })
            
            success_message = dbc.Alert(
                f"Successfully uploaded {filename}! Now configure your data columns.", 
                color="success",
                duration=4000  # Show for 4 seconds
            )
            return session_data, success_message
        
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            traceback.print_exc()
            error_message = dbc.Alert(
                f"Error processing file: {str(e)}", 
                color="danger",
                duration=6000  # Show error for 6 seconds
            )
            return session_data, error_message
    
    return dash.no_update, ""

# --- Column Configuration Callback ---

@app.callback(
    [Output('session-data', 'data', allow_duplicate=True),
     Output('column-metadata', 'data')],
    Input('continue-btn', 'n_clicks'),
    [State('session-data', 'data'),
     State({'type': 'column-type', 'index': ALL}, 'value'),
     State({'type': 'column-type', 'index': ALL}, 'id'),
     State({'type': 'column-desc', 'index': ALL}, 'value'),
     State({'type': 'column-name', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def handle_continue_to_dashboard(n_clicks, session_data, column_types, column_ids, column_descs, column_names):
    """Save column configuration and metadata"""
    if n_clicks and session_data:
        # Save column configuration
        column_metadata = {}
        for i, col_id in enumerate(column_ids):
            col_name = col_id['index']
            col_type = column_types[i] if column_types and i < len(column_types) else 'categorical'
            col_desc = column_descs[i] if column_descs and i < len(column_descs) else ""
            col_display = column_names[i] if column_names and i < len(column_names) else col_name
            
            column_metadata[col_name] = {
                'type': col_type,
                'description': col_desc,
                'display_name': col_display
            }
        
        session_data['configured'] = True
        print(f"Moving to dashboard with column metadata")
        return session_data, column_metadata
    return dash.no_update, {}

# --- Back to Config Callback ---

@app.callback(
    Output('session-data', 'data', allow_duplicate=True),
    Input('back-to-config-btn', 'n_clicks'),
    State('session-data', 'data'),
    prevent_initial_call=True
)
def handle_back_to_config(n_clicks, session_data):
    """Go back to column configuration"""
    if n_clicks:
        if session_data and 'configured' in session_data:
            session_data.pop('configured', None)
        return session_data
    return dash.no_update

# --- AI Chat Callback ---

@app.callback(
    [Output('chat-history', 'children'), 
     Output('chart-library', 'data'), 
     Output('chat-input', 'value')],
    Input('send-btn', 'n_clicks'),
    [State('chat-input', 'value'),
     State('session-data', 'data'),
     State('column-metadata', 'data'),
     State('chat-history', 'children'),
     State('chart-library', 'data')],
    prevent_initial_call=True
)
def handle_chat_message(n_clicks, message, session_data, column_metadata, current_history, chart_library):
    """Process AI assistant chat messages and create charts"""
    if n_clicks and message and session_data:
        print(f"Processing AI request: {message}")
        
        # Load the data
        try:
            df = data_processor.load_saved_data(session_data['file_path'])
            
            # Prepare column info for AI assistant including user descriptions
            column_info = {}
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                sample_values = df[col].dropna().head(3).tolist()
                
                user_description = ""
                display_name = col
                if column_metadata and col in column_metadata:
                    user_description = column_metadata[col].get('description', "")
                    display_name = column_metadata[col].get('display_name', col)
                
                column_info[col] = {
                    'dtype': dtype,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'user_description': user_description,
                    'display_name': display_name
                }
            
            # Use AI assistant to parse the request
            chart_request = ai_assistant.parse_chart_request(
                message, 
                column_info,
                column_metadata or {}
            )
            
            if chart_request and chart_request.chart_type:
                print(f"AI parsed request: {chart_request}")

                # Handle specific count-based chart requests
                if chart_request.y_column == 'count' or (
                    chart_request.filters and chart_request.filters.get('aggregation') == 'count'):
                    # Force a proper count aggregation
                    print("Detected count-based chart request, setting up proper aggregation")
                    
                    # Create a pre-aggregated DataFrame for count-based charts
                    if chart_request.x_column in df.columns:
                        # Use value_counts for simple count aggregation
                        value_counts = df[chart_request.x_column].value_counts().reset_index()
                        value_counts.columns = [chart_request.x_column, 'count']
                        
                        # Replace the original dataframe with the pre-aggregated one
                        df_to_use = value_counts
                        
                        # Make sure the chart config knows to use the 'count' column
                        chart_request.y_column = 'count'
                        
                        # Ensure the chart config knows this is a count aggregation
                        if not chart_request.filters:
                            chart_request.filters = {}
                        chart_request.filters['aggregation'] = 'count'
                        
                        print(f"Created count-based dataframe with columns: {df_to_use.columns.tolist()}")
                        print(f"Sample data: {df_to_use.head(3).to_dict('records')}")
                    else:
                        df_to_use = df
                        print(f"Warning: Cannot create count aggregation, x_column '{chart_request.x_column}' not found")
                else:
                    df_to_use = df

                # Convert ChartRequest to dictionary format for chart generator
                chart_config = create_chart_config(chart_request, df_to_use, column_metadata)

                # Create the chart - use the pre-aggregated dataframe
                fig = chart_generator.create_chart(df_to_use, chart_config)

                # Generate insight/summary - make sure to use the same dataframe used for the chart
                insight = chart_generator.generate_insight(df_to_use, chart_config)

                # Add chart to library, including insight
                chart_library = chart_library or []
                chart_entry = {
                    'id': len(chart_library),
                    'title': chart_config.get('title', 'Chart'),
                    'request': message,
                    'config': chart_config,
                    'figure': fig,
                    'insight': insight
                }
                chart_library.append(chart_entry)

                # Add to chat history
                current_history = current_history or []

                # User message
                user_msg = html.Div([
                    html.Div([
                        html.Span("You", className="fw-bold me-2"),
                        html.Span(message)
                    ], className="d-flex")
                ], className="chat-message user-message mb-2")

                # AI response with generated insight
                ai_msg = html.Div([
                    html.Div([
                        html.Span("AI", className="fw-bold me-2"),
                        html.Span(insight)
                    ], className="d-flex")
                ], className="chat-message ai-message mb-2")

                current_history = current_history + [user_msg, ai_msg]
                
            else:
                # Failed to parse
                current_history = current_history or []
                
                user_msg = html.Div([
                    html.Div([
                        html.Span("You", className="fw-bold me-2"),
                        html.Span(message)
                    ], className="d-flex")
                ], className="chat-message user-message mb-2")
                
                ai_msg = html.Div([
                    html.Div([
                        html.Span("AI", className="fw-bold me-2"),
                        html.Span("I couldn't understand that request. Try something like 'Show me sales by month' or 'Create a bar chart of top products by revenue'")
                    ], className="d-flex")
                ], className="chat-message ai-message error-message mb-2")
                
                current_history = current_history + [user_msg, ai_msg]
                
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            traceback.print_exc()
            current_history = current_history or []
            
            # Add user message
            user_msg = html.Div([
                html.Div([
                    html.Span("You", className="fw-bold me-2"),
                    html.Span(message)
                ], className="d-flex")
            ], className="chat-message user-message mb-2")
            
            # Add error message
            error_msg = html.Div([
                html.Div([
                    html.Span("Error", className="fw-bold me-2 text-danger"),
                    html.Span(f"An error occurred: {str(e)}")
                ], className="d-flex")
            ], className="chat-message error-message mb-2")
            
            current_history = current_history + [user_msg, error_msg]
        
        return current_history, chart_library, ""  # Clear input
    
    return current_history or [], chart_library or [], message or ""

# --- Pivot Builder Modal Callbacks ---

@app.callback(
    Output("pivot-builder-modal", "is_open", allow_duplicate=True),
    [Input("open-pivot-builder-btn", "n_clicks"), 
     Input("close-pivot-builder-btn", "n_clicks"),
     Input("create-chart-btn", "n_clicks")],
    [State("pivot-builder-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_pivot_builder_modal(open_clicks, close_clicks, create_clicks, is_open):
    """Toggle pivot builder modal"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "open-pivot-builder-btn":
        return True
    elif trigger_id == "close-pivot-builder-btn":  # Don't handle create-chart-btn here to avoid duplicate callback
        return False
    return is_open

@app.callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('color-dropdown', 'options'),
     Output('filter-column-dropdown', 'options')],
    [Input('session-data', 'data'),
     Input('column-metadata', 'data'),
     Input("pivot-builder-modal", "is_open")]
)
def update_dropdown_options(session_data, column_metadata, is_open):
    """Update dropdown options based on column metadata"""
    if is_open and session_data and 'columns' in session_data:
        columns = session_data['columns']
        options = []
        
        # Use column metadata if available
        if column_metadata:
            for col in columns:
                if col in column_metadata:
                    display_name = column_metadata[col].get('display_name', col)
                    options.append({'label': display_name, 'value': col})
                else:
                    options.append({'label': col, 'value': col})
        else:
            options = [{'label': col, 'value': col} for col in columns]
            
        return options, options, options, options
    return [], [], [], []

@app.callback(
    [Output('chart-library', 'data', allow_duplicate=True),
     Output('pivot-builder-modal', 'is_open', allow_duplicate=True),
     Output('chart-1-idx', 'data', allow_duplicate=True)],
    Input('create-chart-btn', 'n_clicks'),
    [State('chart-type-dropdown', 'value'),
     State('chart-title-input', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('color-dropdown', 'value'),
     State('agg-dropdown', 'value'),
     State('filter-column-dropdown', 'value'),
     State('filter-operation-dropdown', 'value'),
     State('filter-value-input', 'value'),
     State('session-data', 'data'),
     State('column-metadata', 'data'),
     State('chart-library', 'data')],
    prevent_initial_call=True
)
def create_custom_chart(n_clicks, chart_type, title, x_col, y_col, color_col, agg_func,
                        filter_col, filter_op, filter_val, session_data, column_metadata, chart_library):
    """Create custom chart from pivot builder"""
    if not n_clicks:
        return chart_library or [], dash.no_update, dash.no_update
    if n_clicks and session_data:
        try:
            # Set defaults for missing values
            if not chart_type:
                chart_type = 'bar'
            
            if not x_col and not y_col:
                # If neither axis is specified, use first two columns
                df = data_processor.load_saved_data(session_data['file_path'])
                if len(df.columns) > 0:
                    x_col = df.columns[0]
                    if len(df.columns) > 1:
                        y_col = df.columns[1]
                    else:
                        # If only one column exists, use it for x and do a count aggregation
                        y_col = 'count'
            df = data_processor.load_saved_data(session_data['file_path'])
            
            # Apply filter if specified
            filters = {}
            if filter_col and filter_op and filter_val is not None:
                filters[filter_col] = {
                    'operation': filter_op,
                    'value': filter_val
                }
            
            # Create a ChartRequest object
            # Add aggregation to filters instead of as a separate parameter
            if agg_func:
                if not filters:
                    filters = {}
                filters['aggregation'] = agg_func
                
            chart_request = ChartRequest(
                chart_type=chart_type,
                x_column=x_col,
                y_column=y_col,
                color_column=color_col,
                title=title or f"{chart_type.title()} Chart",
                filters=filters
            )
            
            # Use smart chart config
            chart_config = create_chart_config(chart_request, df, column_metadata)
            
            # Create chart
            fig = chart_generator.create_chart(df, chart_config)
            
            # Generate insight/summary
            insight = chart_generator.generate_insight(df, chart_config, include_aggregation=True)
            
            # Add chart to library
            chart_library = chart_library or []
            chart_entry = {
                'id': len(chart_library),
                'title': chart_config.get('title', 'Custom Chart'),
                'request': f"Custom {chart_type} chart",
                'config': chart_config,
                'figure': fig,
                'insight': insight
            }
            chart_library.append(chart_entry)
            
            print(f"Created custom chart: {chart_entry['title']}")
            print(f"Added to chart library at position {len(chart_library)-1}")
            
            # Return chart library, close modal, and set chart index to new chart
            # Forcibly close the modal by setting is_open to False
            return chart_library, False, len(chart_library) - 1
            
        except Exception as e:
            print(f"Error creating custom chart: {str(e)}")
            traceback.print_exc()
            # Keep modal open in case of error
            return chart_library or [], True, dash.no_update
    
    return chart_library or [], True, dash.no_update

# --- Excel Download Callbacks ---

@app.callback(
    Output('download-excel', 'data'),
    [Input('download-1-excel', 'n_clicks'),
     Input('download-2-excel', 'n_clicks'),
     Input('download-3-excel', 'n_clicks'),
     Input('download-4-excel', 'n_clicks')],
    [State('chart-1-idx', 'data'),
     State('chart-2-idx', 'data'),
     State('chart-3-idx', 'data'),
     State('chart-4-idx', 'data'),
     State('chart-library', 'data')],
    prevent_initial_call=True
)
def download_chart_data(btn1, btn2, btn3, btn4, idx1, idx2, idx3, idx4, chart_library):
    """Download chart data as Excel"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger == 'download-1-excel':
        window_idx = 1
        chart_idx = idx1
    elif trigger == 'download-2-excel':
        window_idx = 2
        chart_idx = idx2
    elif trigger == 'download-3-excel':
        window_idx = 3
        chart_idx = idx3
    elif trigger == 'download-4-excel':
        window_idx = 4
        chart_idx = idx4
    else:
        return dash.no_update
    
    if not chart_library or chart_idx is None or chart_idx >= len(chart_library):
        return dash.no_update
    
    try:
        print(f"Attempting to download Excel for chart index {chart_idx} from window {window_idx}")
        chart_entry = chart_library[chart_idx]
        fig = chart_entry.get('figure')
        title = chart_entry.get('title', f'Chart-{window_idx}')
        
        # Extract data from figure
        df = extract_data_from_figure(fig)
        
        if df is None:
            print(f"Warning: Could not extract data from figure for chart {chart_idx}")
            return dash.no_update
        
        print(f"Successfully extracted data for chart '{title}' with columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")
        
        # Return Excel file
        return dcc.send_data_frame(df.to_excel, f"{title.replace(' ', '_')}.xlsx", sheet_name="ChartData")
    except Exception as e:
        print(f"Error downloading Excel: {str(e)}")
        traceback.print_exc()
        return dash.no_update

# --- Theme Switching Callback ---

@app.callback(
    Output('theme-store', 'data'),
    Input('theme-dropdown', 'value'),
    prevent_initial_call=True
)
def update_theme(selected_theme):
    """Store selected theme"""
    return selected_theme

@app.callback(
    Output('dummy-theme-reload', 'children'),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)
def reload_theme(theme_name):
    """Reload page with new theme"""
    ctx = dash.callback_context
    if ctx.triggered_id == 'theme-store':
        return dcc.Location(href='/', id='theme-reload-location')
    return None

# --- Register all window callbacks ---
register_window_callbacks(app)

# --- Main ---
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    
    # Initialize database
    db_manager.init_db()
    
    # Test AI connection
    print("🦃 Starting Turkey Pivots...")
    
    # Check what type of AI assistant we're using
    if hasattr(ai_assistant, 'test_openai_connection'):
        # Using OpenAI
        if ai_assistant.test_openai_connection():
            print(f"✅ OpenAI Assistant ready with {ai_assistant.model_name}")
        else:
            print("⚠️  OpenAI connection failed - will use rule-based parsing")
            ai_assistant.ai_provider = "rules"
    else:
        # Using Ollama
        if ai_assistant.test_ollama_connection():
            print(f"✅ AI Assistant ready with {ai_assistant.model_name}")
        else:
            print("⚠️  AI Assistant will use rule-based parsing (Ollama not available)")
            ai_assistant.ai_provider = "rules"  # Fallback to rules
    
    # Get port from environment variable (for Render compatibility)
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting web server at http://{host}:{port}")
    app.run_server(debug=False, host=host, port=port)
