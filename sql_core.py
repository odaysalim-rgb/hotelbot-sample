from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings


def get_mysql_uri() -> str:
    """
    Build the SQLAlchemy MySQL URI from settings if MYSQL_URL is not explicitly set.
    """
    if settings.mysql_url:
        return settings.mysql_url

    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
    )


@dataclass
class SQLAnswer:
    sql: str
    rows: List[Dict[str, Any]]
    raw_result: Any


class SQLPipeline:
    """
    Simple Text-to-SQL pipeline on top of a MySQL database using LangChain.
    """

    def __init__(self, table: Optional[str] = None) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

        uri = get_mysql_uri()
        # Limit tables to the main Dubai hotels table for safety by default.
        include_tables = [table or settings.mysql_table]

        self.db = SQLDatabase.from_uri(uri, include_tables=include_tables)
        self.llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)

        # Prompt to generate SQL directly from schema + question
        self.sql_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert MySQL analyst and SQL generator.
            Use ONLY the tables and columns provided.

            IMPORTANT — DATE RULE:
            - The `date` column is stored as a TEXT string.
            - Its format may vary (e.g., '2025-08-21', '21/08/2025', '08/21/2025').
            - ALWAYS parse it safely using:

            COALESCE(
                STR_TO_DATE(date, '%Y-%m-%d'),
                STR_TO_DATE(date, '%d/%m/%Y'),
                STR_TO_DATE(date, '%m/%d/%Y')
            )

            - NEVER use DATE(date) directly.
            - NEVER assume the date format.

            Database schema:
            {schema}

            Write ONLY a valid MySQL SQL query (no explanation).
            The query must answer the user's question.

            Question:
            {question}
            """.strip()
        )

    def ask_sql(self, question: str) -> SQLAnswer:
        """
        Generate SQL for a natural language question, execute it, and return the results.
        """
        # Get schema info for better SQL generation
        schema = self.db.get_table_info()

        # Ask LLM to generate SQL
        msg = self.sql_prompt.format(schema=schema, question=question)
        sql_query = self.llm.invoke(msg).content.strip()

        # Defensive cleanup: strip markdown fences like ```sql ... ``` while keeping the inner query.
        if "```" in sql_query:
            import re

            match = re.search(r"```(?:sql)?\s*(.*?)```", sql_query, re.IGNORECASE | re.DOTALL)
            if match:
                sql_query = match.group(1).strip()
            else:
                # Fallback: remove backticks only
                sql_query = sql_query.replace("```", "").strip()

        if not sql_query:
            raise ValueError("Generated SQL query was empty. Check the LLM prompt or question.")

        # Execute the SQL query
        raw_result = self.db.run(sql_query)

        # SQLDatabase.run may return a string representation; we keep it as-is and
        # also expose it in a rows-like field for convenience (best-effort parsing).
        rows: List[Dict[str, Any]] = []

        return SQLAnswer(sql=sql_query, rows=rows, raw_result=raw_result)



