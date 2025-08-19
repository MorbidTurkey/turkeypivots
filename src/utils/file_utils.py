"""
Utility functions for file handling and other common operations
"""

import os
import pandas as pd
import uuid
from typing import List, Dict, Any
import shutil

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def generate_chart_id() -> str:
    """Generate a unique chart ID"""
    return str(uuid.uuid4())

def ensure_directory_exists(directory_path: str):
    """Ensure a directory exists, create if it doesn't"""
    os.makedirs(directory_path, exist_ok=True)

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    return filename

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def cleanup_temp_files(temp_dir: str = "temp", max_age_hours: int = 24):
    """Clean up temporary files older than specified hours"""
    import time
    
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old temp file: {filename}")
                except OSError as e:
                    print(f"Error removing temp file {filename}: {e}")

def save_dataframe_safely(df: pd.DataFrame, file_path: str, format: str = "parquet") -> bool:
    """Save DataFrame with error handling"""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        
        if format.lower() == "parquet":
            df.to_parquet(file_path, index=False)
        elif format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif format.lower() == "excel":
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return True
    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        return False

def load_dataframe_safely(file_path: str) -> pd.DataFrame:
    """Load DataFrame with error handling"""
    try:
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading DataFrame from {file_path}: {e}")
        return pd.DataFrame()

def validate_column_names(columns: List[str]) -> List[str]:
    """Validate and clean column names"""
    cleaned_columns = []
    seen_names = set()
    
    for col in columns:
        # Clean the column name
        clean_col = str(col).strip()
        clean_col = clean_col.replace(' ', '_')
        clean_col = ''.join(c for c in clean_col if c.isalnum() or c in ['_', '-'])
        
        # Ensure uniqueness
        original_col = clean_col
        counter = 1
        while clean_col in seen_names:
            clean_col = f"{original_col}_{counter}"
            counter += 1
        
        seen_names.add(clean_col)
        cleaned_columns.append(clean_col)
    
    return cleaned_columns

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_data_sample(df: pd.DataFrame, n_rows: int = 5) -> Dict[str, Any]:
    """Get a sample of the data for preview"""
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'head': df.head(n_rows).to_dict('records'),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality metrics"""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime']).columns)
    }

def export_chart_config(chart_config: Dict[str, Any], file_path: str) -> bool:
    """Export chart configuration to JSON file"""
    try:
        import json
        with open(file_path, 'w') as f:
            json.dump(chart_config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error exporting chart config: {e}")
        return False

def import_chart_config(file_path: str) -> Dict[str, Any]:
    """Import chart configuration from JSON file"""
    try:
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error importing chart config: {e}")
        return {}

def create_backup(source_path: str, backup_dir: str = "backups") -> bool:
    """Create a backup of a file"""
    try:
        ensure_directory_exists(backup_dir)
        filename = os.path.basename(source_path)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{timestamp}_{filename}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        shutil.copy2(source_path, backup_path)
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
