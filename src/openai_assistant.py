"""
OpenAI-powered AI Assistant for Turkey Pivots
"""
#
import os
import json
import time
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv

# Try importing pydantic-ai
try:
    from pydantic_ai import Agent, OpenAIModel
    from pydantic import BaseModel, Field
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    print("PydanticAI not available. Will use fallback parsing.")

# Load environment variables
load_dotenv()

@dataclass
class ChartRequest:
    """Structured chart request from AI parsing"""
    chart_type: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    title: Optional[str] = None
    filters: Dict[str, Any] = None


# Only define Pydantic models if pydantic is available
if PYDANTIC_AI_AVAILABLE:
    class ColumnMapping(BaseModel):
        """Mapping from user request to data columns"""
        explanation: str = Field(..., description="Detailed explanation of the column mapping and why it makes sense")
        x_column: Optional[str] = None
        y_column: Optional[str] = None
        color_column: Optional[str] = None
        size_column: Optional[str] = None
        filters: Dict[str, Any] = Field(default_factory=dict)

    class ChartTypeSelection(BaseModel):
        """Chart type selection with confidence and reasoning"""
        chart_type: str = Field(..., description="The selected chart type (bar, line, scatter, pie, etc.)")
        confidence: float = Field(..., ge=0, le=100, description="Confidence score (0-100)")
        reasoning: str = Field(..., description="Detailed reasoning for this chart type selection")

    class FinalChartConfiguration(BaseModel):
        """Final chart configuration combining all elements"""
        chart_type: str
        x_column: Optional[str] = None
        y_column: Optional[str] = None
        color_column: Optional[str] = None
        size_column: Optional[str] = None
        title: str
        filters: Optional[Dict[str, Any]] = None
        reasoning: str = Field(..., description="Final explanation of the complete chart setup")

class OpenAIAssistant:
    """OpenAI-powered AI Assistant using PydanticAI"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("PydanticAI not available. Install with: pip install pydantic-ai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to your .env file.")
        
        # Create OpenAI model
        self.openai_model = OpenAIModel(
            model_name=model_name,
            api_key=self.api_key,
        )
        
        # Agent 1: Column Mapping
        self.column_agent = Agent(
            model=self.openai_model,
            output_type=ColumnMapping,
            system_prompt="""You are a data column mapping expert.

Your job: Analyze the user's request and available data columns to determine what data should be used for visualization.

Key tasks:
1. Understand what the user wants to see/measure
2. Map their intent to specific columns
3. Detect if aggregation is needed (count, sum, average, etc.)
4. Identify any filters needed

For requests like "number of posts per channel":
- This needs COUNT aggregation
- X-axis: Channel (what to group by)
- Y-axis: count (what to measure)
- Aggregation: "count"

For requests like "average engagement by channel":
- X-axis: Channel
- Y-axis: Engagement column
- Aggregation: "mean"

Always explain your reasoning clearly."""
        )
        
        # Agent 2: Chart Type Selection
        self.chart_agent = Agent(
            model=self.openai_model,
            output_type=ChartTypeSelection,
            system_prompt="""You are a data visualization expert.

Your job: Determine the BEST chart type for the given data columns and user request.

Available chart types:
- bar: Compare categorical data
- line: Show trends over time or ordered categories
- scatter: Display relationship between two numeric variables
- pie: Show proportions of a whole (use sparingly, only for parts of a whole)
- histogram: Distribution of a numeric variable
- box: Statistical distribution and outliers
- heatmap: Show patterns in a matrix of values

Choose based on:
1. The nature of the data (categorical vs numeric)
2. The user's intent (comparison, trend, distribution, etc.)
3. Data visualization best practices

Provide your confidence level (0-100%) and explain your reasoning in detail."""
        )
        
        # Agent 3: Final Chart Configuration
        self.config_agent = Agent(
            model=self.openai_model,
            output_type=FinalChartConfiguration,
            system_prompt="""You are a chart configuration specialist.

Your job: Create the final chart specification based on the column mapping and chart type.

Tasks:
1. Define a clear, concise chart title
2. Confirm all column selections are appropriate
3. Add any necessary filters or data transformations
4. Ensure the final configuration will create an effective visualization

Focus on creating a chart that answers the user's original question clearly and accurately.
Explain your final configuration choices."""
        )
        
        # For backwards compatibility
        self.ai_provider = "openai"
        
    def test_openai_connection(self):
        """Test connection to OpenAI API"""
        try:
            # Simple model call to test connection
            from pydantic_ai import OpenAIModel
            model = OpenAIModel(model_name=self.model_name, api_key=self.api_key)
            response = model.generate_text("Hello, are you working?")
            return True
        except Exception as e:
            logging.error(f"OpenAI connection failed: {str(e)}")
            return False
    
    def process_chart_request(self, query: str, df: pd.DataFrame, column_metadata=None):
        """Process natural language query into chart request"""
        try:
            # Prepare column info for agents
            columns_desc = []
            
            # Prepare column descriptions
            if column_metadata:
                # Use provided metadata
                for col_meta in column_metadata:
                    col_type = col_meta.get("type", "unknown")
                    is_category = col_meta.get("is_category", False)
                    if is_category:
                        col_type = "category"
                    
                    desc = col_meta.get("description", "")
                    display_name = col_meta.get("display_name", col_meta.get("name", ""))
                    
                    columns_desc.append(f"- {col_meta.get('name')}: {display_name} ({col_type}) {desc}")
            else:
                # Generate basic column info
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    if col_type == 'object':
                        col_type = 'text/category'
                    elif col_type.startswith('int') or col_type.startswith('float'):
                        col_type = 'numeric'
                    columns_desc.append(f"- {col}: ({col_type})")
            
            # Add sample data
            sample_data = df.head(3).to_string()
            
            # Step 1: Column Mapping
            print("🔍 Step 1: Analyzing columns...")
            column_mapping = self.column_agent.generate(
                f"""User Request: {query}

Available Columns:
{chr(10).join(columns_desc)}

Sample Data:
{sample_data}

Determine which columns should be used for x-axis, y-axis, color, etc. to best visualize what the user is asking for."""
            )
            print(f"   Column mapping: {column_mapping.explanation}")
            
            # Step 2: Chart Type Selection
            print("📊 Step 2: Selecting chart type...")
            chart_selection = self.chart_agent.generate(
                f"""User Request: {query}

Column Mapping:
{column_mapping.explanation}

Available Columns:
{chr(10).join(columns_desc)}

Sample Data:
{sample_data}

Select the best chart type to visualize this data."""
            )
            print(f"   Chart type: {chart_selection.chart_type} ({chart_selection.confidence}% confidence)")
            print(f"   Reasoning: {chart_selection.reasoning}")
            
            # Step 3: Final Chart Configuration
            print("🎯 Step 3: Assembling final chart...")
            final_config = self.config_agent.generate(
                f"""User Request: {query}

Column Mapping:
{column_mapping.explanation}

Chart Type Selection:
{chart_selection.chart_type} ({chart_selection.confidence}% confidence)
{chart_selection.reasoning}

Available Columns:
{chr(10).join(columns_desc)}

Create the final chart configuration."""
            )
            print(f"   Final chart: {final_config.reasoning}")
            
            # Construct ChartRequest
            filters = {}
            if column_mapping.filters:
                filters = column_mapping.filters
            
            # Add aggregation info from column mapping to filters
            aggregation = None
            if "count" in column_mapping.explanation.lower():
                if "average" in column_mapping.explanation.lower() or "mean" in column_mapping.explanation.lower():
                    aggregation = "mean"
                elif "sum" in column_mapping.explanation.lower():
                    aggregation = "sum"
                else:
                    aggregation = "count"
            
            if aggregation:
                filters["aggregation"] = aggregation
            
            # Create and return chart request
            print(f"AI parsed request: {final_config}")
            
            chart_request = ChartRequest(
                chart_type=chart_selection.chart_type,
                x_column=column_mapping.x_column,
                y_column=column_mapping.y_column,
                color_column=column_mapping.color_column,
                size_column=column_mapping.size_column,
                title=final_config.title,
                filters=filters
            )
            
            return chart_request
            
        except Exception as e:
            print(f"Error processing chart request: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # Fallback methods for compatibility with SimpleAIAssistant
    def test_ollama_connection(self):
        """Compatibility method - redirects to OpenAI test"""
        return self.test_openai_connection()
        
# For backwards compatibility
AIAssistant = OpenAIAssistant
