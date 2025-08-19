"""
PydanticAI-based Chart Selector for Turkey Pivots
Simplified 3-step process: column matching → chart selection → chart generation
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass

from .chart_registry import CHART_REGISTRY, ChartType, ChartDefinition

class ColumnMapping(BaseModel):
    """Column mapping for chart generation"""
    x_column: Optional[str] = Field(None, description="Column for X-axis")
    y_column: Optional[str] = Field(None, description="Column for Y-axis") 
    color_column: Optional[str] = Field(None, description="Column for color grouping")
    size_column: Optional[str] = Field(None, description="Column for size mapping")
    
class ChartSelection(BaseModel):
    """Chart selection with reasoning"""
    chart_type: ChartType = Field(..., description="Selected chart type")
    confidence: float = Field(..., description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(..., description="Why this chart type was selected")
    alternative_charts: List[ChartType] = Field(default=[], description="Alternative chart suggestions")

class ChartRequest(BaseModel):
    """Complete chart request with all parameters"""
    chart_type: ChartType = Field(..., description="Type of chart to create")
    column_mapping: ColumnMapping = Field(..., description="Column assignments")
    title: Optional[str] = Field(None, description="Chart title")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters to apply")
    reasoning: str = Field(..., description="Explanation of chart choice")

@dataclass
class DataContext:
    """Context about available data"""
    column_info: Dict[str, Dict]
    sample_data: Optional[pd.DataFrame] = None
    user_descriptions: Optional[Dict[str, str]] = None

class PydanticChartSelector:
    """PydanticAI-based chart selection system"""
    
    def __init__(self, model_name: str = "gpt-oss:20b", ollama_url: str = "http://localhost:11434/v1"):
        """Initialize the chart selector with PydanticAI"""
        
        # Create Ollama model
        self.ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=ollama_url),
        )
        
        # Create specialized agents for each step
        self.column_mapper_agent = self._create_column_mapper_agent()
        self.chart_selector_agent = self._create_chart_selector_agent()
        self.request_builder_agent = self._create_request_builder_agent()
    
    def _create_column_mapper_agent(self) -> Agent:
        """Create agent for intelligent column mapping"""
        return Agent(
            model=self.ollama_model,
            output_type=ColumnMapping,
            system_prompt="""You are an expert at mapping data columns to chart axes.
            
Your task is to analyze the user's request and available data columns to suggest the best column mappings for visualization.

For trading card data, common patterns:
- Value columns: MARKET, MID, TREND, AVG, LOW, Price Bought (for Y-axis)
- Category columns: Card Name, Set Name, Condition, Language (for X-axis)  
- Grouping columns: Set Name, Condition, Printing (for color)

Consider:
1. User intent (what they want to see)
2. Data types (numeric vs categorical)
3. Meaningful relationships
4. Visualization best practices

Return optimal column assignments or null if no good mapping exists."""
        )
    
    def _create_chart_selector_agent(self) -> Agent:
        """Create agent for chart type selection"""
        return Agent(
            model=self.ollama_model,
            output_type=ChartSelection,
            system_prompt="""You are an expert data visualization consultant.

Your task is to select the best chart type based on:
1. User intent and request
2. Available data structure
3. Visualization best practices
4. Data storytelling goals

Available chart types: bar, line, scatter, pie, histogram, box, heatmap, map, violin, sunburst, treemap

Consider:
- What story does the user want to tell?
- What type of comparison or analysis?
- How many data points and categories?
- What insights should be highlighted?

Provide reasoning for your choice and suggest alternatives."""
        )
    
    def _create_request_builder_agent(self) -> Agent:
        """Create agent for building complete chart requests"""
        return Agent(
            model=self.ollama_model,
            output_type=ChartRequest,
            system_prompt="""You are a chart configuration expert.

Your task is to build a complete chart request by combining:
1. Selected chart type
2. Column mappings
3. Appropriate title
4. Any necessary filters

For trading card data, consider filters like:
- Excluding missing values (common for market prices)
- Value ranges (e.g., cards over $10)
- Category filters (specific sets, conditions)

Create meaningful titles that describe what the chart shows.
Suggest filters that improve data quality and focus."""
        )
    
    async def analyze_request(self, user_request: str, data_context: DataContext) -> ChartRequest:
        """
        Main method: analyze user request and return complete chart configuration
        
        Args:
            user_request: Natural language chart request
            data_context: Information about available data
            
        Returns:
            ChartRequest: Complete chart configuration
        """
        
        # Step 1: Intelligent column mapping
        column_mapping = await self._map_columns(user_request, data_context)
        
        # Step 2: Chart type selection
        chart_selection = await self._select_chart_type(user_request, data_context, column_mapping)
        
        # Step 3: Build complete request
        chart_request = await self._build_request(user_request, data_context, column_mapping, chart_selection)
        
        return chart_request
    
    async def _map_columns(self, user_request: str, data_context: DataContext) -> ColumnMapping:
        """Step 1: Map user intent to appropriate columns"""
        
        # Create context for the agent
        column_info_str = self._format_column_info(data_context.column_info)
        sample_data_str = self._format_sample_data(data_context.sample_data) if data_context.sample_data is not None else ""
        
        prompt = f"""
User Request: "{user_request}"

Available Columns:
{column_info_str}

{sample_data_str}

Map the user's intent to appropriate column assignments for visualization.
"""
        
        result = await self.column_mapper_agent.run(prompt)
        return result.output
    
    async def _select_chart_type(self, user_request: str, data_context: DataContext, 
                                column_mapping: ColumnMapping) -> ChartSelection:
        """Step 2: Select best chart type based on intent and data"""
        
        # Get compatible charts from registry
        compatible_charts = CHART_REGISTRY.find_charts_by_intent(user_request, data_context.column_info)
        
        if not compatible_charts:
            # Fallback to all compatible charts
            compatible_charts = CHART_REGISTRY.find_compatible_charts(data_context.column_info)
        
        chart_options = [chart_def.chart_type.value for chart_def, _ in compatible_charts[:5]]  # Top 5
        
        prompt = f"""
User Request: "{user_request}"

Column Mapping: {column_mapping.model_dump()}

Compatible Chart Types: {chart_options}

Chart Type Definitions:
{self._format_chart_definitions(compatible_charts)}

Select the best chart type for this visualization request.
"""
        
        result = await self.chart_selector_agent.run(prompt)
        return result.output
    
    async def _build_request(self, user_request: str, data_context: DataContext,
                           column_mapping: ColumnMapping, chart_selection: ChartSelection) -> ChartRequest:
        """Step 3: Build complete chart request with filters and title"""
        
        prompt = f"""
User Request: "{user_request}"
Selected Chart Type: {chart_selection.chart_type}
Column Mapping: {column_mapping.model_dump()}
Reasoning: {chart_selection.reasoning}

Available Data Info:
{self._format_column_info(data_context.column_info)}

Build a complete chart request with:
1. Appropriate title that describes what the chart shows
2. Any beneficial filters (especially for data quality)
3. Final reasoning for the configuration

For trading card data, common beneficial filters:
- exclude_missing: true (for price columns with missing values)
- min_value filters for focusing on valuable cards
- category filters for specific sets or conditions
"""
        
        result = await self.request_builder_agent.run(prompt)
        return result.output
    
    def _format_column_info(self, column_info: Dict[str, Dict]) -> str:
        """Format column information for prompts"""
        formatted = []
        for col_name, info in column_info.items():
            dtype = info.get('dtype', 'unknown')
            non_null = info.get('non_null_count', 0)
            total = info.get('total_count', 0)
            sample_values = info.get('sample_values', [])
            
            formatted.append(f"- {col_name} ({dtype}): {non_null}/{total} non-null")
            if sample_values:
                formatted.append(f"  Sample values: {sample_values[:3]}")
        
        return "\n".join(formatted)
    
    def _format_sample_data(self, sample_data: pd.DataFrame) -> str:
        """Format sample data for prompts"""
        if sample_data is None or sample_data.empty:
            return ""
        
        return f"\nSample Data (first 3 rows):\n{sample_data.head(3).to_string()}"
    
    def _format_chart_definitions(self, compatible_charts: List[tuple]) -> str:
        """Format chart definitions for prompts"""
        formatted = []
        for chart_def, compatibility in compatible_charts:
            formatted.append(f"- {chart_def.chart_type.value}: {chart_def.description}")
            formatted.append(f"  Use cases: {', '.join(chart_def.use_cases[:3])}")
            formatted.append(f"  Confidence: {compatibility['confidence']:.2f}")
        
        return "\n".join(formatted)

class SimpleChartSelector:
    """Simplified non-AI version for fallback"""
    
    def __init__(self):
        self.registry = CHART_REGISTRY
    
    def analyze_request(self, user_request: str, data_context: DataContext) -> ChartRequest:
        """Fallback rule-based analysis"""
        
        # Find compatible charts
        compatible_charts = self.registry.find_charts_by_intent(user_request, data_context.column_info)
        
        if not compatible_charts:
            compatible_charts = self.registry.find_compatible_charts(data_context.column_info)
        
        if not compatible_charts:
            raise ValueError("No compatible charts found for the available data")
        
        # Select best chart
        best_chart, compatibility = compatible_charts[0]
        
        # Simple column mapping
        suggested_mappings = compatibility["suggested_mappings"]
        column_mapping = ColumnMapping(
            x_column=suggested_mappings.get("x"),
            y_column=suggested_mappings.get("y"),
            color_column=suggested_mappings.get("color"),
            size_column=suggested_mappings.get("size")
        )
        
        # Basic filters for trading card data
        filters = {}
        if "missing" in user_request.lower() or "null" in user_request.lower():
            if column_mapping.y_column:
                filters[column_mapping.y_column] = {"exclude_missing": True}
        
        return ChartRequest(
            chart_type=best_chart.chart_type,
            column_mapping=column_mapping,
            title=f"{best_chart.name}: {column_mapping.y_column} by {column_mapping.x_column}",
            filters=filters if filters else None,
            reasoning=f"Selected {best_chart.name} based on use case matching and data compatibility"
        )
