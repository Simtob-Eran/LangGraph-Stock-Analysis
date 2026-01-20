"""SQLite database operations for analysis logging and caching."""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("database")


class Database:
    """SQLite database manager for stock analysis system."""

    def __init__(self, db_path: str = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or settings.DATABASE_PATH
        self._ensure_db_directory()
        self._init_schema()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema if not exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Analysis runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    user_query TEXT,
                    execution_time_seconds REAL,
                    status TEXT,
                    error_message TEXT
                )
            """)

            # Agent executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    agent_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    input_data TEXT,
                    output_data TEXT,
                    reasoning TEXT,
                    confidence REAL,
                    execution_time_ms INTEGER,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            """)

            # Data cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_cache (
                    ticker TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    data TEXT NOT NULL,
                    expires_at DATETIME
                )
            """)

            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    ticker TEXT NOT NULL,
                    overall_score REAL,
                    recommendation TEXT,
                    markdown_report TEXT,
                    json_data TEXT,
                    created_at DATETIME,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_ticker
                ON analysis_runs(ticker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_timestamp
                ON analysis_runs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_ticker
                ON data_cache(ticker)
            """)

            logger.info("Database schema initialized successfully")

    def create_analysis_run(
        self,
        run_id: str,
        ticker: str,
        analysis_type: str,
        user_query: str
    ) -> str:
        """
        Create a new analysis run record.

        Args:
            run_id: Unique run identifier
            ticker: Stock ticker
            analysis_type: Type of analysis
            user_query: Original user query

        Returns:
            Run ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_runs
                (id, ticker, analysis_type, timestamp, user_query, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, ticker, analysis_type, datetime.now(), user_query, "running"))

        logger.debug(f"Created analysis run: {run_id} for {ticker}")
        return run_id

    def update_analysis_run(
        self,
        run_id: str,
        execution_time: float,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update analysis run with completion details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE analysis_runs
                SET execution_time_seconds = ?, status = ?, error_message = ?
                WHERE id = ?
            """, (execution_time, status, error_message, run_id))

        logger.debug(f"Updated analysis run: {run_id} - Status: {status}")

    def log_agent_execution(
        self,
        execution_id: str,
        run_id: str,
        agent_name: str,
        ticker: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning: str,
        confidence: float,
        execution_time_ms: int
    ):
        """Log an agent execution."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_executions
                (id, run_id, agent_name, ticker, timestamp, input_data,
                 output_data, reasoning, confidence, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                run_id,
                agent_name,
                ticker,
                datetime.now(),
                json.dumps(input_data),
                json.dumps(output_data, default=str),
                reasoning,
                confidence,
                execution_time_ms
            ))

        logger.debug(f"Logged execution for agent: {agent_name}")

    def cache_data(self, ticker: str, data: Dict[str, Any], hours: int = 24):
        """
        Cache data for a ticker.

        Args:
            ticker: Stock ticker
            data: Data to cache
            hours: Hours until expiration
        """
        expires_at = datetime.now() + timedelta(hours=hours)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO data_cache
                (ticker, timestamp, data, expires_at)
                VALUES (?, ?, ?, ?)
            """, (ticker, datetime.now(), json.dumps(data, default=str), expires_at))

        logger.debug(f"Cached data for {ticker} (expires: {expires_at})")

    def get_cached_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a ticker if not expired.

        Args:
            ticker: Stock ticker

        Returns:
            Cached data or None if not found/expired
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT data, expires_at FROM data_cache
                WHERE ticker = ?
            """, (ticker,))

            row = cursor.fetchone()
            if not row:
                return None

            expires_at = datetime.fromisoformat(row['expires_at'])
            if expires_at < datetime.now():
                logger.debug(f"Cache expired for {ticker}")
                return None

            logger.debug(f"Cache hit for {ticker}")
            return json.loads(row['data'])

    def save_report(
        self,
        report_id: str,
        run_id: str,
        ticker: str,
        overall_score: float,
        recommendation: str,
        markdown_report: str,
        json_data: Dict[str, Any]
    ):
        """Save a final analysis report."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports
                (id, run_id, ticker, overall_score, recommendation,
                 markdown_report, json_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                run_id,
                ticker,
                overall_score,
                recommendation,
                markdown_report,
                json.dumps(json_data, default=str),
                datetime.now()
            ))

        logger.info(f"Saved report: {report_id} for {ticker}")

    def get_recent_analyses(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent analyses for a ticker."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analysis_runs
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (ticker, limit))

            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_cache(self, days: int = 7):
        """Remove cache entries older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM data_cache
                WHERE expires_at < ?
            """, (cutoff,))

            deleted = cursor.rowcount

        logger.info(f"Cleaned up {deleted} old cache entries")
        return deleted
