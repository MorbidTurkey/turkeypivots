"""
Simplified AI Assistant using PydanticAI 3-Agent Architecture
Much cleaner than the complex rule-based system
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import pandas as pd

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.ollama import OllamaProvider
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

# Simple ChartRequest for compatibility
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

# PydanticAI Models
class ColumnMapping(BaseModel):
    """Agent 1 Output: Column assignments for the visualization"""
    x_column: Optional[str] = Field(None, description="Column for X-axis (categories, time, etc)")
    y_column: Optional[str] = Field(None, description="Column for Y-axis (values to measure)")
    color_column: Optional[str] = Field(None, description="Column for color coding/grouping")
    aggregation: Optional[str] = Field(None, description="count, sum, mean, max, min if needed")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters to apply")
    reasoning: str = Field(..., description="Why these columns were chosen")

class ChartSelection(BaseModel):
    """Agent 2 Output: Chart type selection"""
    chart_type: str = Field(..., description="bar, line, scatter, pie, histogram, box, heatmap, map")
    confidence: float = Field(..., description="Confidence in this choice (0-1)")
    reasoning: str = Field(..., description="Why this chart type fits the data and question")
    alternatives: List[str] = Field(default=[], description="Other chart types that could work")

class FinalChartRequest(BaseModel):
    """Agent 3 Output: Complete chart configuration"""
    chart_type: str
    x_column: Optional[str]
    y_column: Optional[str] 
    color_column: Optional[str]
    title: str
    filters: Optional[Dict[str, Any]]
    reasoning: str = Field(..., description="Final explanation of the complete chart setup")

class SimpleAIAssistant:
    """Simplified 3-Agent PydanticAI System"""
    
    def __init__(self, model_name: str = "gpt-oss:20b", ollama_url: str = "http://localhost:11434/v1"):
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("PydanticAI not available. Install with: pip install pydantic-ai")
        
        self.model_name = model_name
        
        # Create Ollama model
        self.ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=ollama_url),
        )
        
        # Agent 1: Column Mapping
        self.column_agent = Agent(
            model=self.ollama_model,
            output_type=ColumnMapping,
            system_prompt="""You are a data column mapping expert.

Your job: Analyze the user's question and available data columns to determine what data should be used for visualization.

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
            model=self.ollama_model,
            output_type=ChartSelection,
            system_prompt="""You are a data visualization expert.

Your job: Given the user's intent and column mappings, choose the best chart type.

Chart types available:
- bar: Compare categories, show counts/totals
- line: Trends over time, continuous data
- scatter: Relationships between two numeric variables
- pie: Composition/proportions (use sparingly, <8 categories)
- histogram: Distribution of single numeric variable
- box: Statistical distribution, outliers
- heatmap: Correlation matrix, density
- map: Geographic data

Guidelines:
- Count/aggregation by category → bar chart
- Time series → line chart
- Two numeric variables → scatter plot
- Proportions/percentages → pie chart
- Single variable distribution → histogram

Be confident but also suggest alternatives."""
        )
        
        # Agent 3: Final Assembly
        self.assembly_agent = Agent(
            model=self.ollama_model,
            output_type=FinalChartRequest,
            system_prompt="""You are a chart configuration expert.

Your job: Take the column mappings and chart selection to create the final chart configuration.

Tasks:
1. Combine the column mapping and chart type
2. Create a clear, descriptive title
3. Ensure all components work together
4. Add any final filters or adjustments

Make sure the title describes what the chart shows, not just the data.
Example: "Number of Posts by Channel" not "Channel vs Count"
"""
        )
    
    def parse_chart_request(self, user_input: str, column_info: Dict[str, Dict], 
                          column_descriptions: Dict[str, str] = None) -> ChartRequest:
        """
        Simple 3-step process to create charts
        """
        try:
            # Prepare context
            available_columns = list(column_info.keys())
            
            # Format column information
            column_details = []
            for col, info in column_info.items():
                detail = f"- {col}: {info.get('dtype', 'unknown')} type"
                if 'sample_values' in info:
                    detail += f", examples: {info['sample_values'][:3]}"
                if column_descriptions and col in column_descriptions:
                    detail += f", description: {column_descriptions[col]}"
                column_details.append(detail)
            
            context = f"""
User Request: "{user_input}"

Available Columns:
{chr(10).join(column_details)}
"""
            
            # Step 1: Column Mapping
            print("🔍 Step 1: Analyzing columns...")
            column_result = self.column_agent.run_sync(context)
            column_mapping = column_result.output
            print(f"   Column mapping: {column_mapping.reasoning}")
            
            # Step 2: Chart Selection
            print("📊 Step 2: Selecting chart type...")
            chart_context = f"""
{context}

Column Mapping Result:
- X-axis: {column_mapping.x_column}
- Y-axis: {column_mapping.y_column}
- Color: {column_mapping.color_column}
- Aggregation needed: {column_mapping.aggregation}
- Reasoning: {column_mapping.reasoning}

What chart type best shows this data?
"""
            chart_result = self.chart_agent.run_sync(chart_context)
            chart_selection = chart_result.output
            print(f"   Chart type: {chart_selection.chart_type} ({chart_selection.confidence:.1%} confidence)")
            print(f"   Reasoning: {chart_selection.reasoning}")
            
            # Step 3: Final Assembly
            print("🎯 Step 3: Assembling final chart...")
            final_context = f"""
{context}

Column Mapping:
{column_mapping.model_dump()}

Chart Selection:
{chart_selection.model_dump()}

Create the final chart configuration.
"""
            final_result = self.assembly_agent.run_sync(final_context)
            final_chart = final_result.output
            print(f"   Final chart: {final_chart.reasoning}")
            
            # Handle aggregation (for chart generator)
            filters = final_chart.filters or {}
            if column_mapping.aggregation:
                filters['aggregation'] = column_mapping.aggregation
            
            # Convert to legacy format
            return ChartRequest(
                chart_type=final_chart.chart_type,
                x_column=final_chart.x_column,
                y_column=final_chart.y_column if final_chart.y_column != 'count' else 'count',
                color_column=final_chart.color_column,
                title=final_chart.title,
                filters=filters if filters else None
            )
            
        except Exception as e:
            print(f"❌ PydanticAI failed: {e}")
            # Simple fallback
            return self._simple_fallback(user_input, column_info)
    
    def _simple_fallback(self, user_input: str, column_info: Dict[str, Dict]) -> ChartRequest:
        """Ultra-simple fallback for when AI fails"""
        available_columns = list(column_info.keys())
        
        # Detect aggregation
        if any(word in user_input.lower() for word in ['count', 'number of', 'how many']):
            # Find first text column for grouping
            x_col = None
            for col in available_columns:
                if column_info[col].get('dtype', '').startswith('object'):
                    x_col = col
                    break
            
            return ChartRequest(
                chart_type='bar',
                x_column=x_col,
                y_column='count',
                title=f"Count by {x_col}" if x_col else "Data Count",
                filters={'aggregation': 'count'}
            )
        
        # Default: first two columns
        x_col = available_columns[0] if available_columns else None
        y_col = available_columns[1] if len(available_columns) > 1 else None
        
        return ChartRequest(
            chart_type='bar',
            x_column=x_col,
            y_column=y_col,
            title=f"{y_col} by {x_col}" if x_col and y_col else "Simple Chart"
        )

    # Compatibility methods for existing app
    @property
    def ai_provider(self):
        return "pydantic_ai"
    
    @ai_provider.setter  
    def ai_provider(self, value):
        pass  # Ignore - we always use PydanticAI
    
    def test_ollama_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            result = self.column_agent.run_sync("test")
            return True
        except:
            return False

# For backward compatibility
AIAssistant = SimpleAIAssistant
