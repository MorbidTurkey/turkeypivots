"""
Database management for storing user sessions and chart configurations
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "data/turkeypivots.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    columns TEXT NOT NULL,
                    column_descriptions TEXT,
                    column_renames TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Charts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS charts (
                    chart_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    chart_config TEXT NOT NULL,
                    chart_title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Chart library table (for user's saved charts)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chart_library (
                    library_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    chart_id TEXT NOT NULL,
                    window_position INTEGER,
                    is_active BOOLEAN DEFAULT FALSE,
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                    FOREIGN KEY (chart_id) REFERENCES charts (chart_id)
                )
            """)
            
            conn.commit()
    
    def save_session(self, session_id: str, filename: str, file_path: str, 
                    columns: List[str], column_descriptions: Dict[str, str] = None,
                    column_renames: Dict[str, str] = None) -> bool:
        """Save a user session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, filename, file_path, columns, column_descriptions, column_renames, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    filename,
                    file_path,
                    json.dumps(columns),
                    json.dumps(column_descriptions) if column_descriptions else None,
                    json.dumps(column_renames) if column_renames else None,
                    datetime.now()
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT session_id, filename, file_path, columns, column_descriptions, 
                           column_renames, created_at, last_accessed
                    FROM sessions WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'session_id': row[0],
                        'filename': row[1],
                        'file_path': row[2],
                        'columns': json.loads(row[3]),
                        'column_descriptions': json.loads(row[4]) if row[4] else {},
                        'column_renames': json.loads(row[5]) if row[5] else {},
                        'created_at': row[6],
                        'last_accessed': row[7]
                    }
                return None
        except Exception as e:
            print(f"Error retrieving session: {e}")
            return None
    
    def save_chart(self, chart_id: str, session_id: str, chart_config: Dict[str, Any], 
                   chart_title: str = None) -> bool:
        """Save a chart configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO charts 
                    (chart_id, session_id, chart_config, chart_title)
                    VALUES (?, ?, ?, ?)
                """, (
                    chart_id,
                    session_id,
                    json.dumps(chart_config),
                    chart_title
                ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving chart: {e}")
            return False
    
    def get_chart(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chart configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT chart_id, session_id, chart_config, chart_title, created_at
                    FROM charts WHERE chart_id = ?
                """, (chart_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'chart_id': row[0],
                        'session_id': row[1],
                        'chart_config': json.loads(row[2]),
                        'chart_title': row[3],
                        'created_at': row[4]
                    }
                return None
        except Exception as e:
            print(f"Error retrieving chart: {e}")
            return None
    
    def get_session_charts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all charts for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT chart_id, chart_config, chart_title, created_at
                    FROM charts WHERE session_id = ?
                    ORDER BY created_at DESC
                """, (session_id,))
                
                rows = cursor.fetchall()
                return [{
                    'chart_id': row[0],
                    'chart_config': json.loads(row[1]),
                    'chart_title': row[2],
                    'created_at': row[3]
                } for row in rows]
        except Exception as e:
            print(f"Error retrieving session charts: {e}")
            return []
    
    def save_to_library(self, library_id: str, session_id: str, chart_id: str, 
                       window_position: int = None) -> bool:
        """Save a chart to user's library"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO chart_library 
                    (library_id, session_id, chart_id, window_position)
                    VALUES (?, ?, ?, ?)
                """, (library_id, session_id, chart_id, window_position))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving to library: {e}")
            return False
    
    def get_library_charts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all charts in user's library"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT cl.library_id, cl.chart_id, cl.window_position, cl.saved_at,
                           c.chart_config, c.chart_title
                    FROM chart_library cl
                    JOIN charts c ON cl.chart_id = c.chart_id
                    WHERE cl.session_id = ?
                    ORDER BY cl.saved_at DESC
                """, (session_id,))
                
                rows = cursor.fetchall()
                return [{
                    'library_id': row[0],
                    'chart_id': row[1],
                    'window_position': row[2],
                    'saved_at': row[3],
                    'chart_config': json.loads(row[4]),
                    'chart_title': row[5]
                } for row in rows]
        except Exception as e:
            print(f"Error retrieving library charts: {e}")
            return []
    
    def update_session_access(self, session_id: str):
        """Update last accessed time for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE sessions SET last_accessed = ? WHERE session_id = ?
                """, (datetime.now(), session_id))
                
                conn.commit()
        except Exception as e:
            print(f"Error updating session access: {e}")
    
    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up sessions older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old sessions and related data
                cursor.execute("""
                    DELETE FROM chart_library 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE last_accessed < datetime('now', '-{} days')
                    )
                """.format(days_old))
                
                cursor.execute("""
                    DELETE FROM charts 
                    WHERE session_id IN (
                        SELECT session_id FROM sessions 
                        WHERE last_accessed < datetime('now', '-{} days')
                    )
                """.format(days_old))
                
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE last_accessed < datetime('now', '-{} days')
                """.format(days_old))
                
                conn.commit()
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                cursor.execute("SELECT COUNT(*) FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM charts")
                stats['total_charts'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chart_library")
                stats['library_items'] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
