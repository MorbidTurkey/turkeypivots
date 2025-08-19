"""
Master Chart Registry for Turkey Pivots
Defines all supported chart types with their use cases, requirements, and parameters
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

class ChartType(str, Enum):
    """Enumeration of supported chart types"""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    MAP = "map"
    VIOLIN = "violin"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"

class DataType(str, Enum):
    """Data types for column classification"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    GEOGRAPHIC = "geographic"

@dataclass
class ColumnRequirement:
    """Defines requirements for chart columns"""
    name: str
    data_type: DataType
    required: bool = True
    description: str = ""

@dataclass
class ChartDefinition:
    """Complete definition of a chart type"""
    chart_type: ChartType
    name: str
    description: str
    use_cases: List[str]
    column_requirements: List[ColumnRequirement]
    min_data_points: int = 1
    max_categories: Optional[int] = None
    examples: List[str] = None
    
    def matches_use_case(self, user_intent: str) -> bool:
        """Check if this chart matches the user's intent"""
        user_intent_lower = user_intent.lower()
        return any(use_case.lower() in user_intent_lower for use_case in self.use_cases)
    
    def validate_data_compatibility(self, column_info: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate if available data is compatible with this chart type"""
        result = {
            "compatible": False,
            "missing_requirements": [],
            "suggested_mappings": {},
            "confidence": 0.0
        }
        
        available_columns = list(column_info.keys())
        suggested_mappings = {}
        missing_requirements = []
        
        for req in self.column_requirements:
            compatible_columns = []
            
            # Find columns that match the data type requirement
            for col_name, col_info in column_info.items():
                col_dtype = col_info.get('dtype', 'object')
                
                if req.data_type == DataType.NUMERIC and col_dtype in ['int64', 'float64', 'int32', 'float32']:
                    compatible_columns.append(col_name)
                elif req.data_type == DataType.CATEGORICAL and col_dtype in ['object', 'category', 'string']:
                    compatible_columns.append(col_name)
                elif req.data_type == DataType.DATETIME and 'datetime' in str(col_dtype).lower():
                    compatible_columns.append(col_name)
                elif req.data_type == DataType.GEOGRAPHIC and any(geo_word in col_name.lower() 
                    for geo_word in ['lat', 'lon', 'latitude', 'longitude', 'coord', 'location']):
                    compatible_columns.append(col_name)
            
            if compatible_columns:
                # Suggest the best matching column
                suggested_mappings[req.name] = compatible_columns[0]
            elif req.required:
                missing_requirements.append(req.name)
        
        # Calculate compatibility
        total_required = sum(1 for req in self.column_requirements if req.required)
        satisfied_required = total_required - len(missing_requirements)
        
        if total_required > 0:
            result["confidence"] = satisfied_required / total_required
            result["compatible"] = len(missing_requirements) == 0
        
        result["missing_requirements"] = missing_requirements
        result["suggested_mappings"] = suggested_mappings
        
        return result

class ChartRegistry:
    """Master registry of all chart definitions"""
    
    def __init__(self):
        self.charts = self._initialize_chart_definitions()
    
    def _initialize_chart_definitions(self) -> Dict[ChartType, ChartDefinition]:
        """Initialize all chart definitions"""
        return {
            ChartType.BAR: ChartDefinition(
                chart_type=ChartType.BAR,
                name="Bar Chart",
                description="Compare values across categories using rectangular bars",
                use_cases=[
                    "compare values", "show differences", "rank items", "most valuable",
                    "highest", "lowest", "compare across categories", "breakdown by",
                    "which is best", "top cards", "expensive", "valuable"
                ],
                column_requirements=[
                    ColumnRequirement("x", DataType.CATEGORICAL, True, "Category column (e.g., Card Name, Set Name)"),
                    ColumnRequirement("y", DataType.NUMERIC, True, "Value column (e.g., Market Value, Price)")
                ],
                max_categories=50,
                examples=[
                    "show most valuable cards",
                    "compare market values by set",
                    "which cards are most expensive"
                ]
            ),
            
            ChartType.LINE: ChartDefinition(
                chart_type=ChartType.LINE,
                name="Line Chart",
                description="Show trends and changes over time or continuous data",
                use_cases=[
                    "trend", "over time", "change", "growth", "progression",
                    "timeline", "time series", "evolution", "track", "monitor"
                ],
                column_requirements=[
                    ColumnRequirement("x", DataType.DATETIME, True, "Time/sequence column"),
                    ColumnRequirement("y", DataType.NUMERIC, True, "Value column to track")
                ],
                examples=[
                    "card value trends over time",
                    "price changes by date",
                    "market progression"
                ]
            ),
            
            ChartType.SCATTER: ChartDefinition(
                chart_type=ChartType.SCATTER,
                name="Scatter Plot",
                description="Explore relationships between two numeric variables",
                use_cases=[
                    "relationship", "correlation", "vs", "versus", "against",
                    "compare two", "relationship between", "how does", "affect"
                ],
                column_requirements=[
                    ColumnRequirement("x", DataType.NUMERIC, True, "First numeric variable"),
                    ColumnRequirement("y", DataType.NUMERIC, True, "Second numeric variable")
                ],
                examples=[
                    "market value vs trend value",
                    "compare two price metrics",
                    "relationship between values"
                ]
            ),
            
            ChartType.PIE: ChartDefinition(
                chart_type=ChartType.PIE,
                name="Pie Chart",
                description="Show proportions and composition of a whole",
                use_cases=[
                    "distribution", "proportion", "percentage", "share",
                    "composition", "breakdown", "what percentage", "how much of"
                ],
                column_requirements=[
                    ColumnRequirement("values", DataType.NUMERIC, True, "Values to show proportions"),
                    ColumnRequirement("labels", DataType.CATEGORICAL, True, "Category labels")
                ],
                max_categories=10,
                examples=[
                    "distribution of cards by condition",
                    "percentage by set",
                    "share of total value"
                ]
            ),
            
            ChartType.HISTOGRAM: ChartDefinition(
                chart_type=ChartType.HISTOGRAM,
                name="Histogram",
                description="Show frequency distribution of numeric data",
                use_cases=[
                    "distribution", "frequency", "histogram", "bins", "spread",
                    "range", "how many", "count by range"
                ],
                column_requirements=[
                    ColumnRequirement("x", DataType.NUMERIC, True, "Numeric column to analyze distribution")
                ],
                examples=[
                    "distribution of card prices",
                    "frequency of market values",
                    "price range analysis"
                ]
            ),
            
            ChartType.BOX: ChartDefinition(
                chart_type=ChartType.BOX,
                name="Box Plot",
                description="Show statistical distribution with quartiles and outliers",
                use_cases=[
                    "outliers", "quartiles", "statistical", "spread", "median",
                    "box plot", "distribution summary", "detect outliers"
                ],
                column_requirements=[
                    ColumnRequirement("y", DataType.NUMERIC, True, "Numeric values to analyze"),
                    ColumnRequirement("x", DataType.CATEGORICAL, False, "Optional grouping variable")
                ],
                examples=[
                    "price distribution by condition",
                    "detect outlier card values",
                    "statistical summary of prices"
                ]
            ),
            
            ChartType.HEATMAP: ChartDefinition(
                chart_type=ChartType.HEATMAP,
                name="Heatmap",
                description="Show correlation or intensity between two categorical variables",
                use_cases=[
                    "correlation", "heatmap", "intensity", "matrix",
                    "pattern", "relationship matrix", "heat map"
                ],
                column_requirements=[
                    ColumnRequirement("x", DataType.CATEGORICAL, True, "First categorical variable"),
                    ColumnRequirement("y", DataType.CATEGORICAL, True, "Second categorical variable"),
                    ColumnRequirement("values", DataType.NUMERIC, True, "Values to map intensity")
                ],
                examples=[
                    "correlation between metrics",
                    "pattern analysis",
                    "intensity mapping"
                ]
            ),
            
            ChartType.MAP: ChartDefinition(
                chart_type=ChartType.MAP,
                name="Geographic Map",
                description="Visualize data on geographic locations",
                use_cases=[
                    "map", "geographic", "location", "coordinates", "geo",
                    "spatial", "where", "geography", "region"
                ],
                column_requirements=[
                    ColumnRequirement("lat", DataType.GEOGRAPHIC, True, "Latitude coordinates"),
                    ColumnRequirement("lon", DataType.GEOGRAPHIC, True, "Longitude coordinates"),
                    ColumnRequirement("values", DataType.NUMERIC, False, "Optional values to display")
                ],
                examples=[
                    "card locations on map",
                    "geographic distribution",
                    "spatial analysis"
                ]
            ),
            
            ChartType.VIOLIN: ChartDefinition(
                chart_type=ChartType.VIOLIN,
                name="Violin Plot",
                description="Show distribution shape and density of numeric data",
                use_cases=[
                    "density", "violin", "distribution shape", "detailed distribution",
                    "probability density", "shape analysis"
                ],
                column_requirements=[
                    ColumnRequirement("y", DataType.NUMERIC, True, "Numeric values to analyze"),
                    ColumnRequirement("x", DataType.CATEGORICAL, False, "Optional grouping variable")
                ],
                examples=[
                    "detailed price distribution",
                    "density analysis of values",
                    "distribution shape comparison"
                ]
            ),
            
            ChartType.SUNBURST: ChartDefinition(
                chart_type=ChartType.SUNBURST,
                name="Sunburst Chart",
                description="Show hierarchical data with nested categories",
                use_cases=[
                    "hierarchy", "nested", "sunburst", "hierarchical breakdown",
                    "tree structure", "multilevel", "categorical hierarchy"
                ],
                column_requirements=[
                    ColumnRequirement("path", DataType.CATEGORICAL, True, "Hierarchical path columns"),
                    ColumnRequirement("values", DataType.NUMERIC, True, "Values for each segment")
                ],
                examples=[
                    "hierarchical card breakdown",
                    "nested category analysis",
                    "multilevel grouping"
                ]
            ),
            
            ChartType.TREEMAP: ChartDefinition(
                chart_type=ChartType.TREEMAP,
                name="Treemap",
                description="Show hierarchical data as nested rectangles sized by value",
                use_cases=[
                    "treemap", "nested rectangles", "hierarchical sizing",
                    "proportional hierarchy", "space-filling", "tree map"
                ],
                column_requirements=[
                    ColumnRequirement("path", DataType.CATEGORICAL, True, "Hierarchical path columns"),
                    ColumnRequirement("values", DataType.NUMERIC, True, "Values to size rectangles")
                ],
                examples=[
                    "proportional card value breakdown",
                    "hierarchical sizing",
                    "nested value visualization"
                ]
            )
        }
    
    def get_chart_definition(self, chart_type: ChartType) -> Optional[ChartDefinition]:
        """Get chart definition by type"""
        return self.charts.get(chart_type)
    
    def find_compatible_charts(self, column_info: Dict[str, Dict], 
                             min_confidence: float = 0.5) -> List[tuple[ChartDefinition, Dict]]:
        """Find all charts compatible with the available data"""
        compatible_charts = []
        
        for chart_def in self.charts.values():
            compatibility = chart_def.validate_data_compatibility(column_info)
            if compatibility["compatible"] and compatibility["confidence"] >= min_confidence:
                compatible_charts.append((chart_def, compatibility))
        
        # Sort by confidence score (highest first)
        compatible_charts.sort(key=lambda x: x[1]["confidence"], reverse=True)
        return compatible_charts
    
    def find_charts_by_intent(self, user_intent: str, column_info: Dict[str, Dict]) -> List[tuple[ChartDefinition, Dict]]:
        """Find charts that match user intent and are compatible with data"""
        matching_charts = []
        
        for chart_def in self.charts.values():
            if chart_def.matches_use_case(user_intent):
                compatibility = chart_def.validate_data_compatibility(column_info)
                if compatibility["compatible"]:
                    matching_charts.append((chart_def, compatibility))
        
        # Sort by confidence score (highest first)
        matching_charts.sort(key=lambda x: x[1]["confidence"], reverse=True)
        return matching_charts
    
    def get_all_chart_types(self) -> List[ChartType]:
        """Get list of all supported chart types"""
        return list(self.charts.keys())
    
    def get_chart_examples(self, chart_type: ChartType) -> List[str]:
        """Get usage examples for a specific chart type"""
        chart_def = self.get_chart_definition(chart_type)
        return chart_def.examples if chart_def else []

# Global registry instance
CHART_REGISTRY = ChartRegistry()
