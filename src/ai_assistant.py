"""
AI Assistant for natural language chart generation
"""

import pandas as pd
import re
import json
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass

# AI service availability flags
OLLAMA_AVAILABLE = True  # We'll enable Ollama
OPENAI_AVAILABLE = False  # Keep OpenAI disabled for now

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

class AIAssistant:
    def __init__(self, ai_provider: str = "ollama"):
        """
        Initialize AI Assistant
        
        Args:
            ai_provider: "ollama" for local LLM, "rules" for rule-based parsing, or "openai" for OpenAI API
        """
        self.ai_provider = ai_provider
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "gpt-oss:20b"  # Using the available gpt-oss model
        
        # Chart type keywords mapping (used as fallback)
        self.chart_keywords = {
            'bar': ['bar', 'column', 'compare', 'comparison', 'by', 'across', 'breakdown'],
            'line': ['line', 'trend', 'over time', 'time series', 'timeline', 'change', 'growth'],
            'scatter': ['scatter', 'relationship', 'correlation', 'vs', 'against', 'versus'],
            'pie': ['pie', 'distribution', 'proportion', 'percentage', 'share', 'composition'],
            'histogram': ['histogram', 'distribution', 'frequency', 'bins', 'range'],
            'box': ['box', 'boxplot', 'quartiles', 'outliers', 'spread'],
            'heatmap': ['heatmap', 'correlation', 'matrix', 'heat map'],
            'map': ['map', 'geographic', 'location', 'coordinates', 'geo', 'spatial']
        }
    
    def parse_chart_request(self, user_input: str, column_info: Dict[str, Dict], 
                          column_descriptions: Dict[str, str] = None) -> ChartRequest:
        """
        Parse natural language input into structured chart request
        
        Args:
            user_input: User's natural language request
            column_info: Information about available columns
            column_descriptions: User-provided column descriptions
            
        Returns:
            ChartRequest: Structured chart request
        """
        user_input = user_input.lower().strip()
        
        # Try AI parsing first if available, fallback to rules
        if self.ai_provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                return self._parse_with_ollama(user_input, column_info, column_descriptions)
            except Exception as e:
                print(f"Ollama parsing failed, using rules: {e}")
                return self._parse_with_rules(user_input, column_info, column_descriptions)
        else:
            # Fallback to rule-based parsing
            return self._parse_with_rules(user_input, column_info, column_descriptions)
    
    def _parse_with_rules(self, user_input: str, column_info: Dict[str, Dict], 
                         column_descriptions: Dict[str, str] = None) -> ChartRequest:
        """Enhanced rule-based parsing with better error guidance and filter support"""
        user_input = user_input.lower()
        available_columns = list(column_info.keys())
        
        print(f"Available columns: {available_columns}")
        
        # Parse filter requests
        filters = {}
        filter_keywords = ['filter', 'exclude', 'remove', 'ignore', 'without', 'drop', 'skip']
        
        if any(keyword in user_input for keyword in filter_keywords):
            # Look for "filter out missing values" or "exclude missing" patterns
            if 'missing' in user_input or 'null' in user_input or 'empty' in user_input:
                # Find which column to filter
                for col in available_columns:
                    if col.lower() in user_input:
                        filters[col] = {'exclude_missing': True}
                        print(f"Added filter: exclude missing values from {col}")
                        break
                # If no specific column mentioned but MARKET/value columns exist, assume those
                if not filters:
                    value_columns = ['MARKET', 'MID', 'TREND', 'AVG', 'LOW']
                    for col in value_columns:
                        if col in available_columns:
                            filters[col] = {'exclude_missing': True}
                            print(f"Added default filter: exclude missing values from {col}")
                            break
        
        # Enhanced value detection for trading cards
        value_keywords = ['valuable', 'value', 'price', 'cost', 'worth', 'expensive', 'cheap']
        value_columns = ['MARKET', 'MID', 'TREND', 'AVG', 'LOW', 'Price Bought']
        
        # Find available value columns
        available_value_columns = [col for col in value_columns if col in available_columns]
        
        # Enhanced category detection
        category_columns = ['Card Name', 'Set Name', 'Condition', 'Printing', 'Language', 'Set Code']
        available_category_columns = [col for col in category_columns if col in available_columns]
        
        print(f"Available value columns: {available_value_columns}")
        print(f"Available category columns: {available_category_columns}")
        print(f"Parsed filters: {filters}")
        
        # Find best matching columns
        y_column = None
        x_column = None
        
        # For "most valuable" requests
        if any(keyword in user_input for keyword in value_keywords):
            if available_value_columns:
                # Prioritize MARKET for "most valuable" requests
                y_column = next((col for col in ['MARKET', 'MID', 'TREND', 'AVG'] if col in available_value_columns), available_value_columns[0])
            
            if available_category_columns:
                # Prioritize Card Name for trading cards
                x_column = next((col for col in ['Card Name', 'Set Name'] if col in available_category_columns), available_category_columns[0])
        
        # Check if we found valid columns
        if not y_column or not x_column:
            print(f"Could not determine appropriate columns. Y: {y_column}, X: {x_column}")
            return None
        
        # Determine chart type
        chart_type = self._detect_chart_type(user_input)
        if not chart_type:
            chart_type = 'bar'  # Default for "most valuable" requests
        
        title = self._generate_title(chart_type, x_column, y_column, user_input)
        
        print(f"Rule-based parsing result: chart_type={chart_type}, x={x_column}, y={y_column}, filters={filters}")
        
        return ChartRequest(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=None,
            title=title,
            filters=filters if filters else None
        )
    
    def _parse_with_ollama(self, user_input: str, column_info: Dict[str, Dict], 
                          column_descriptions: Dict[str, str] = None) -> ChartRequest:
        """Parse using local Ollama LLM - Enhanced for intelligent column mapping"""
        try:
            prompt = self._create_advanced_parsing_prompt(user_input, column_info, column_descriptions)
            
            response = requests.post(self.ollama_url, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '')
                return self._extract_chart_config(ai_response, column_info)
            else:
                print(f"Ollama request failed: {response.status_code}")
                return self._parse_with_rules(user_input, column_info, column_descriptions)
                
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return self._parse_with_rules(user_input, column_info, column_descriptions)
    
    # def _parse_with_openai(self, user_input: str, column_info: Dict[str, Dict], 
    #                       column_descriptions: Dict[str, str] = None) -> ChartRequest:
    #     """Parse using OpenAI API - Currently disabled"""
    #     print("OpenAI parsing is currently disabled")
    #     return self._parse_with_rules(user_input, column_info, column_descriptions)
    
    def _create_advanced_parsing_prompt(self, user_input: str, column_info: Dict[str, Dict], 
                                       column_descriptions: Dict[str, str] = None) -> str:
        """Create an advanced prompt for AI parsing with intelligent column mapping"""
        
        # Analyze columns and create detailed descriptions
        column_analysis = []
        for col, info in column_info.items():
            analysis = f"- {col}: "
            analysis += f"Type: {info['dtype']}, "
            analysis += f"Unique values: {info['unique_count']}, "
            analysis += f"Sample: {info.get('sample_values', [])[:3]}"
            
            # Add user description if available
            if column_descriptions and col in column_descriptions:
                analysis += f", User description: '{column_descriptions[col]}'"
            
            # Infer semantic meaning from column name and data
            semantic_hints = []
            col_lower = col.lower()
            if any(word in col_lower for word in ['date', 'time', 'year', 'month', 'day']):
                semantic_hints.append("likely temporal data")
            if any(word in col_lower for word in ['price', 'cost', 'amount', 'value', 'revenue', 'income', 'sales']):
                semantic_hints.append("likely monetary/numeric data")
            if any(word in col_lower for word in ['category', 'type', 'group', 'class', 'name', 'status']):
                semantic_hints.append("likely categorical data")
            if any(word in col_lower for word in ['lat', 'lon', 'latitude', 'longitude', 'location']):
                semantic_hints.append("likely geographic data")
            if any(word in col_lower for word in ['count', 'number', 'quantity', 'total']):
                semantic_hints.append("likely count/quantity data")
                
            if semantic_hints:
                analysis += f", Likely: {', '.join(semantic_hints)}"
            
            column_analysis.append(analysis)

        prompt = f"""You are an expert data analyst. Analyze this request and create the best visualization.

USER REQUEST: "{user_input}"

AVAILABLE COLUMNS:
{chr(10).join(column_analysis)}

Your task:
1. Understand what the user wants to see (even if vague)
2. Choose the most appropriate chart type
3. Intelligently map columns to chart axes/properties
4. Handle common variations (e.g., "income" might match "revenue", "salary", "earnings")
5. Parse any filtering requests (e.g., "filter out missing values", "exclude nulls")

RULES FOR COLUMN MATCHING:
- Look for semantic similarity, not just exact matches
- "over time" → find date/time columns
- "income/revenue/sales/earnings" → find monetary columns  
- "by category/type/group" → find categorical columns
- Consider column names, data types, and sample values
- If user says "show X by Y", X goes to y-axis, Y goes to x-axis
- For time series, automatically use line charts
- For comparisons across categories, use bar charts
- For relationships between numbers, use scatter plots

FILTERING RULES:
- If user mentions "filter", "exclude", "remove", "without", "missing", "null", "empty" values
- Extract which column(s) to filter and what type of filter to apply
- Common filters: exclude missing values, min/max ranges, specific value lists

CHART TYPE GUIDELINES:
- Line chart: trends over time, continuous data progression
- Bar chart: comparing categories, discrete comparisons
- Scatter plot: relationship between two numeric variables
- Pie chart: composition/proportions (only if <10 categories)
- Histogram: distribution of single numeric variable
- Heatmap: correlation between multiple numeric variables
- Box plot: statistical distribution, outliers
- Map: geographic data with lat/lon coordinates

Return ONLY a JSON object with this exact structure:
{{
    "chart_type": "bar|line|scatter|pie|histogram|box|heatmap|map",
    "x_column": "exact_column_name_or_null",
    "y_column": "exact_column_name_or_null", 
    "color_column": "exact_column_name_or_null",
    "size_column": "exact_column_name_or_null",
    "title": "Descriptive chart title",
    "filters": {{"column_name": {{"exclude_missing": true}} }} or null,
    "reasoning": "Brief explanation of choices"
}}

FILTER EXAMPLES:
- "filter out missing Market values" → {{"MARKET": {{"exclude_missing": true}}}}
- "exclude cards under $5" → {{"MARKET": {{"min_value": 5}}}}
- "show only Good condition cards" → {{"Condition": {{"values": ["Good"]}}}}

IMPORTANT: 
- Use EXACT column names from the list above
- If no good match exists, use null
- Be smart about fuzzy matching (income→revenue, date→timestamp, etc.)
- Choose the chart type that best shows the relationship the user wants to see
- Parse filters from natural language and include in "filters" field

JSON Response:"""

        return prompt
    
    def _extract_chart_config(self, ai_response: str, column_info: Dict[str, Dict]) -> ChartRequest:
        """Extract chart configuration from AI response with enhanced parsing"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                config = json.loads(json_match.group())
                
                # Validate that columns exist in the dataset
                available_columns = list(column_info.keys())
                
                def validate_column(col_name):
                    if col_name is None or col_name == "null":
                        return None
                    if col_name in available_columns:
                        return col_name
                    # Try fuzzy matching
                    col_lower = col_name.lower()
                    for available_col in available_columns:
                        if col_lower == available_col.lower():
                            return available_col
                    return None
                
                x_col = validate_column(config.get('x_column'))
                y_col = validate_column(config.get('y_column'))
                color_col = validate_column(config.get('color_column'))
                size_col = validate_column(config.get('size_column'))
                
                # Extract filters if present
                filters = config.get('filters')
                if filters:
                    # Validate filter column names
                    validated_filters = {}
                    for col_name, filter_config in filters.items():
                        validated_col = validate_column(col_name)
                        if validated_col:
                            validated_filters[validated_col] = filter_config
                    filters = validated_filters if validated_filters else None
                
                chart_request = ChartRequest(
                    chart_type=config.get('chart_type', 'bar'),
                    x_column=x_col,
                    y_column=y_col,
                    color_column=color_col,
                    size_column=size_col,
                    title=config.get('title', 'Generated Chart'),
                    filters=filters
                )
                
                # Log AI reasoning if available
                if 'reasoning' in config:
                    print(f"AI reasoning: {config['reasoning']}")
                
                return chart_request
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"AI response: {ai_response}")
        except Exception as e:
            print(f"Error parsing AI response: {e}")
        
        # Fallback to rule-based parsing if AI parsing fails
        return self._parse_with_rules(ai_response, column_info)
    
    def _detect_chart_type(self, user_input: str) -> str:
        """Detect chart type from user input using keywords"""
        scores = {}
        
        for chart_type, keywords in self.chart_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input)
            if score > 0:
                scores[chart_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'bar'  # Default to bar chart
    
    def _assign_columns(self, chart_type: str, mentioned_columns: List[str], 
                       column_info: Dict[str, Dict]) -> tuple:
        """Assign columns to x, y, color based on chart type and data types"""
        
        numerical_cols = [col for col in mentioned_columns 
                         if column_info[col]['dtype'] in ['int64', 'float64']]
        categorical_cols = [col for col in mentioned_columns 
                           if column_info[col]['dtype'] == 'object']
        
        x_column = y_column = color_column = None
        
        if chart_type in ['bar', 'box']:
            if categorical_cols and numerical_cols:
                x_column = categorical_cols[0]
                y_column = numerical_cols[0]
                if len(categorical_cols) > 1:
                    color_column = categorical_cols[1]
        
        elif chart_type == 'line':
            if len(mentioned_columns) >= 2:
                x_column = mentioned_columns[0]
                y_column = mentioned_columns[1]
        
        elif chart_type == 'scatter':
            if len(numerical_cols) >= 2:
                x_column = numerical_cols[0]
                y_column = numerical_cols[1]
                if categorical_cols:
                    color_column = categorical_cols[0]
        
        elif chart_type == 'pie':
            if categorical_cols:
                x_column = categorical_cols[0]  # Names
                if numerical_cols:
                    y_column = numerical_cols[0]  # Values
        
        elif chart_type == 'histogram':
            if numerical_cols:
                x_column = numerical_cols[0]
        
        return x_column, y_column, color_column
    
    def _generate_title(self, chart_type: str, x_column: str, y_column: str, user_input: str) -> str:
        """Generate appropriate title for the chart"""
        if 'title:' in user_input:
            # Extract explicit title
            title_match = re.search(r'title:\s*([^,\n]+)', user_input, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
        
        # Generate based on columns and chart type
        if chart_type == 'bar' and x_column and y_column:
            return f'{y_column} by {x_column}'
        elif chart_type == 'line' and x_column and y_column:
            return f'{y_column} over {x_column}'
        elif chart_type == 'scatter' and x_column and y_column:
            return f'{y_column} vs {x_column}'
        elif chart_type == 'pie' and x_column:
            return f'Distribution of {x_column}'
        elif chart_type == 'histogram' and x_column:
            return f'Distribution of {x_column}'
        else:
            return f'{chart_type.title()} Chart'
    
    def get_chart_suggestions(self, column_info: Dict[str, Dict], 
                             column_descriptions: Dict[str, str] = None) -> List[str]:
        """Generate suggested chart ideas based on available data"""
        
        suggestions = []
        columns = list(column_info.keys())
        
        numerical_cols = [col for col, info in column_info.items() 
                         if info['dtype'] in ['int64', 'float64']]
        categorical_cols = [col for col, info in column_info.items() 
                           if info['dtype'] == 'object']
        
        # Generate natural language suggestions
        if numerical_cols and categorical_cols:
            suggestions.append(f"Show me {numerical_cols[0]} by {categorical_cols[0]}")
            suggestions.append(f"Create a bar chart of {numerical_cols[0]} for each {categorical_cols[0]}")
        
        if len(numerical_cols) >= 2:
            suggestions.append(f"Plot {numerical_cols[1]} vs {numerical_cols[0]}")
            suggestions.append(f"Show the correlation between {numerical_cols[0]} and {numerical_cols[1]}")
        
        for num_col in numerical_cols[:2]:
            suggestions.append(f"Show the distribution of {num_col}")
        
        for cat_col in categorical_cols[:2]:
            suggestions.append(f"Create a pie chart of {cat_col}")
        
        # Time-based suggestions if we detect time columns
        time_cols = [col for col in columns if any(word in col.lower() 
                    for word in ['date', 'time', 'year', 'month', 'day'])]
        if time_cols and numerical_cols:
            suggestions.append(f"Show {numerical_cols[0]} over time")
        
        return suggestions[:5]
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI API - Currently disabled"""
        print("OpenAI integration is currently disabled. Using rule-based parsing only.")
        print("To enable OpenAI: pip install openai, then update the code.")
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model_name,
                "prompt": "Hello",
                "stream": False
            }, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running and the model is installed:")
            print(f"  ollama pull {self.model_name}")
            print("  ollama serve")
            return False
