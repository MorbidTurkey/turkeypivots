"""
Data processing utilities for handling file uploads and data manipulation
"""

import pandas as pd
import base64
import io
import os
from typing import Optional, Dict, Any

class DataProcessor:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def process_upload(self, contents: str, filename: str) -> pd.DataFrame:
        """
        Process uploaded file contents and return a pandas DataFrame
        
        Args:
            contents: Base64 encoded file contents
            filename: Name of the uploaded file
            
        Returns:
            pd.DataFrame: Processed data
        """
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            if filename.endswith('.csv'):
                # Try different encodings for CSV files
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                raise ValueError(f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}")
            
            # Basic data cleaning
            df = self.clean_dataframe(df)
            return df
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.str.strip()  # Remove whitespace
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        
        return df
    
    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about each column
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Column information including data types, unique values, etc.
        """
        column_info = {}
        
        for col in df.columns:
            info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            # Add specific info based on data type
            if df[col].dtype in ['int64', 'float64']:
                info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
            elif df[col].dtype == 'object':
                info['most_common'] = df[col].value_counts().head(5).to_dict()
            
            column_info[col] = info
        
        return column_info
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer the semantic type of each column (categorical, numerical, datetime, etc.)
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Column semantic types
        """
        column_types = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                column_types[col] = 'empty'
                continue
            
            # Check for datetime
            if series.dtype == 'object':
                try:
                    pd.to_datetime(series.head(100))
                    column_types[col] = 'datetime'
                    continue
                except:
                    pass
            
            # Check for numerical
            if series.dtype in ['int64', 'float64']:
                if series.nunique() > len(series) * 0.9:
                    column_types[col] = 'numerical_continuous'
                else:
                    column_types[col] = 'numerical_discrete'
            
            # Check for categorical
            elif series.dtype == 'object':
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.1:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
            
            # Boolean
            elif series.dtype == 'bool':
                column_types[col] = 'boolean'
            
            else:
                column_types[col] = 'other'
        
        return column_types

    def load_saved_data(self, file_path: str) -> pd.DataFrame:
        """
        Load previously saved data from temporary storage
        
        Args:
            file_path: Path to the saved data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Try to load based on file extension first
            if file_path.endswith('.parquet'):
                try:
                    return pd.read_parquet(file_path)
                except ImportError:
                    # pyarrow not available, try pickle fallback
                    pickle_path = file_path.replace('.parquet', '.pkl')
                    if os.path.exists(pickle_path):
                        print(f"Parquet not available, loading pickle fallback: {pickle_path}")
                        return pd.read_pickle(pickle_path)
                    else:
                        raise Exception("Parquet file cannot be read and no pickle fallback found")
            elif file_path.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:
                # Try both formats if extension is unclear
                try:
                    return pd.read_parquet(file_path)
                except:
                    try:
                        return pd.read_pickle(file_path)
                    except:
                        raise ValueError(f"Cannot read file in any supported format: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading saved data: {str(e)}")
