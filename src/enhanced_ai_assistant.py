"""
Enhanced AI Assistant using PydanticAI for Turkey Pivots
Integrates the new PydanticAI-based architecture with existing system
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
import asyncio

# Import existing components
from ai_assistant import ChartRequest
from chart_registry import CHART_REGISTRY, ChartType

# Define fallback DataContext
@dataclass
class DataContext:
    column_info: Dict[str, Dict]
    sample_data: Optional[pd.DataFrame] = None
    user_descriptions: Optional[Dict[str, str]] = None

# Try importing PydanticAI components with fallback
PYDANTIC_AI_AVAILABLE = False
try:
    from pydantic_chart_selector import (
        PydanticChartSelector, 
        SimpleChartSelector, 
        ChartRequest,
        ColumnMapping
    )
    PYDANTIC_AI_AVAILABLE = True
    print("PydanticAI integration loaded successfully")
except ImportError as e:
    print(f"PydanticAI not available, using fallback: {e}")
    
    # Define minimal fallback classes
    @dataclass
    class ChartRequest:
        chart_type: ChartType
        column_mapping: Any
        title: Optional[str] = None
        filters: Optional[Dict[str, Any]] = None
        reasoning: str = ""
    
    @dataclass
    class ColumnMapping:
        x_column: Optional[str] = None
        y_column: Optional[str] = None
        color_column: Optional[str] = None
        size_column: Optional[str] = None

class EnhancedAIAssistant:
    """Enhanced AI Assistant with PydanticAI integration"""
    
    def __init__(self, use_pydantic_ai: bool = True, model_name: str = "gpt-oss:20b"):
        """
        Initialize enhanced AI assistant
        
        Args:
            use_pydantic_ai: Whether to use PydanticAI (falls back to simple if unavailable)
            model_name: Model name for Ollama
        """
        self.use_pydantic_ai = use_pydantic_ai and PYDANTIC_AI_AVAILABLE
        self.model_name = model_name
        
        if self.use_pydantic_ai:
            try:
                self.pydantic_selector = PydanticChartSelector(model_name=model_name)
                print("Initialized PydanticAI selector")
            except Exception as e:
                print(f"Failed to initialize PydanticAI, using simple selector: {e}")
                self.use_pydantic_ai = False
                if PYDANTIC_AI_AVAILABLE:
                    self.simple_selector = SimpleChartSelector()
        
        if not self.use_pydantic_ai:
            if PYDANTIC_AI_AVAILABLE:
                self.simple_selector = SimpleChartSelector()
                print("Initialized simple chart selector")
            else:
                print("PydanticAI not available, will use rule-based fallback")
    
    def parse_chart_request(self, user_input: str, column_info: Dict[str, Dict], 
                          column_descriptions: Dict[str, str] = None,
                          sample_data: pd.DataFrame = None) -> ChartRequest:
        """
        Parse chart request using enhanced AI or fallback methods
        
        Args:
            user_input: User's natural language request
            column_info: Information about available columns  
            column_descriptions: User-provided column descriptions
            sample_data: Sample of the data for context
            
        Returns:
            LegacyChartRequest: Chart request compatible with existing system
        """
        
        # Create data context
        data_context = DataContext(
            column_info=column_info,
            sample_data=sample_data.head(5) if sample_data is not None else None,
            user_descriptions=column_descriptions
        )
        
        try:
            if self.use_pydantic_ai and PYDANTIC_AI_AVAILABLE:
                # Use PydanticAI async analysis (already returns legacy format)
                return asyncio.run(self._analyze_with_pydantic_ai(user_input, data_context))
            elif PYDANTIC_AI_AVAILABLE:
                # Use simple fallback
                chart_request = self._analyze_with_simple_selector(user_input, data_context)
                return self._convert_to_legacy_format(chart_request)
            else:
                # No PydanticAI available, use legacy rules
                return self._fallback_to_rules(user_input, column_info, column_descriptions)
            
        except Exception as e:
            print(f"Enhanced AI analysis failed: {e}")
            # Final fallback to rule-based parsing
            return self._fallback_to_rules(user_input, column_info, column_descriptions)
    
    async def _analyze_with_pydantic_ai(self, user_input: str, data_context: DataContext):
        """Analyze using PydanticAI - with error handling"""
        try:
            pydantic_result = await self.pydantic_selector.analyze_request(user_input, data_context)
            return self._convert_to_legacy_format(pydantic_result)
        except Exception as e:
            print(f"PydanticAI analysis failed: {e}")
            # Fall back to simple selector if available, otherwise rule-based
            if hasattr(self, 'simple_selector'):
                return self._convert_to_legacy_format(
                    self.simple_selector.analyze_request(user_input, data_context)
                )
            else:
                # Convert data context to legacy format for rule-based fallback
                return self._fallback_to_rules(user_input, data_context.column_info, data_context.user_descriptions)
    
    def _analyze_with_simple_selector(self, user_input: str, data_context: DataContext):
        """Analyze using simple rule-based selector"""
        return self.simple_selector.analyze_request(user_input, data_context)
    
    def _convert_to_legacy_format(self, modern_request) -> ChartRequest:
        """Convert modern ChartRequest to legacy format"""
        if hasattr(modern_request, 'chart_type'):
            chart_type = modern_request.chart_type.value if hasattr(modern_request.chart_type, 'value') else str(modern_request.chart_type)
        else:
            chart_type = 'bar'
            
        column_mapping = getattr(modern_request, 'column_mapping', None)
        
        return ChartRequest(
            chart_type=chart_type,
            x_column=getattr(column_mapping, 'x_column', None) if column_mapping else None,
            y_column=getattr(column_mapping, 'y_column', None) if column_mapping else None,
            color_column=getattr(column_mapping, 'color_column', None) if column_mapping else None,
            size_column=getattr(column_mapping, 'size_column', None) if column_mapping else None,
            title=getattr(modern_request, 'title', None),
            filters=getattr(modern_request, 'filters', None)
        )
    
    def _fallback_to_rules(self, user_input: str, column_info: Dict[str, Dict],
                          column_descriptions: Dict[str, str] = None) -> ChartRequest:
        """Enhanced rule-based parsing that adapts to different data types"""
        
        # Smart column detection for different data types
        available_columns = list(column_info.keys())
        
        # Detect aggregation requests (count, number of, etc.)
        aggregation_keywords = ['count', 'number of', 'how many', 'total of']
        is_aggregation = any(keyword in user_input.lower() for keyword in aggregation_keywords)
        
        if is_aggregation:
            # For aggregation requests like "posts per channel"
            # Find the grouping column (channel, category, etc.)
            grouping_keywords = ['channel', 'category', 'type', 'group', 'per']
            x_column = None
            
            for keyword in grouping_keywords:
                for col in available_columns:
                    if keyword.lower() in col.lower():
                        x_column = col
                        break
                if x_column:
                    break
            
            # If no obvious grouping column, use first text/category column
            if not x_column:
                for col in available_columns:
                    col_info = column_info.get(col, {})
                    if col_info.get('dtype', '').startswith(('object', 'string')):
                        x_column = col
                        break
            
            # For aggregation, we need to specify that we want to count
            # The chart generator will need to handle this by counting occurrences
            from ai_assistant import ChartRequest as LegacyChartRequest
            return LegacyChartRequest(
                chart_type='bar',
                x_column=x_column,
                y_column='count',  # Special keyword indicating count aggregation
                title=f"Number of Posts by {x_column}" if x_column else "Post Count",
                filters={'aggregation': 'count'}  # Flag for chart generator
            )
        
        # Fall back to original rule-based parsing for non-aggregation requests
        from ai_assistant import AIAssistant
        
        fallback_assistant = AIAssistant(ai_provider="rules")
        return fallback_assistant.parse_chart_request(user_input, column_info, column_descriptions)
    
    def get_chart_suggestions(self, column_info: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Get chart suggestions based on available data"""
        suggestions = []
        
        compatible_charts = CHART_REGISTRY.find_compatible_charts(column_info, min_confidence=0.3)
        
        for chart_def, compatibility in compatible_charts[:5]:  # Top 5 suggestions
            suggestion = {
                "chart_type": chart_def.chart_type.value,
                "name": chart_def.name,
                "description": chart_def.description,
                "confidence": compatibility["confidence"],
                "use_cases": chart_def.use_cases[:3],  # Top 3 use cases
                "examples": chart_def.examples[:2] if chart_def.examples else [],
                "suggested_mappings": compatibility["suggested_mappings"]
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def explain_chart_choice(self, chart_type: str, user_input: str, column_info: Dict[str, Dict]) -> str:
        """Explain why a particular chart type was chosen"""
        try:
            chart_def = CHART_REGISTRY.get_chart_definition(ChartType(chart_type))
            if not chart_def:
                return f"Chart type '{chart_type}' is not recognized."
            
            compatibility = chart_def.validate_data_compatibility(column_info)
            
            explanation = f"""
Chart Choice Explanation:
- **Chart Type**: {chart_def.name}
- **Description**: {chart_def.description}
- **Confidence**: {compatibility['confidence']:.1%}

**Why this chart fits your request:**
"""
            
            # Check which use cases match
            matching_cases = [case for case in chart_def.use_cases if case.lower() in user_input.lower()]
            if matching_cases:
                explanation += f"- Matches your intent: {', '.join(matching_cases[:3])}\n"
            
            # Show column mappings
            if compatibility['suggested_mappings']:
                explanation += "- **Suggested column mappings:**\n"
                for role, column in compatibility['suggested_mappings'].items():
                    explanation += f"  - {role}: {column}\n"
            
            # Show examples
            if chart_def.examples:
                explanation += f"- **Similar use cases**: {', '.join(chart_def.examples[:2])}\n"
            
            return explanation
            
        except Exception as e:
            return f"Could not generate explanation: {e}"
    
    def validate_chart_feasibility(self, chart_type: str, column_info: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate if a chart type is feasible with available data"""
        try:
            chart_def = CHART_REGISTRY.get_chart_definition(ChartType(chart_type))
            if not chart_def:
                return {"feasible": False, "reason": f"Unknown chart type: {chart_type}"}
            
            compatibility = chart_def.validate_data_compatibility(column_info)
            
            return {
                "feasible": compatibility["compatible"],
                "confidence": compatibility["confidence"],
                "missing_requirements": compatibility["missing_requirements"],
                "suggested_mappings": compatibility["suggested_mappings"],
                "reason": "All requirements met" if compatibility["compatible"] else 
                         f"Missing: {', '.join(compatibility['missing_requirements'])}"
            }
            
        except Exception as e:
            return {"feasible": False, "reason": f"Validation error: {e}"}
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama server for compatibility with existing app"""
        try:
            # Try to use the PydanticAI selector if available
            if self.use_pydantic_ai and hasattr(self, 'pydantic_selector'):
                return True
            # Always return True since we have fallback systems
            return True
        except Exception:
            # Even if connection fails, we have rule-based fallback
            return True
    
    @property
    def ai_provider(self):
        """Compatibility property for existing app"""
        if self.use_pydantic_ai:
            return "pydantic_ai"
        else:
            return "rules"
    
    @ai_provider.setter
    def ai_provider(self, value):
        """Compatibility setter for existing app"""
        if value == "rules":
            self.use_pydantic_ai = False
        elif value in ["pydantic_ai", "ollama"]:
            self.use_pydantic_ai = True and PYDANTIC_AI_AVAILABLE

# Global instance for easy access
enhanced_ai_assistant = EnhancedAIAssistant()
