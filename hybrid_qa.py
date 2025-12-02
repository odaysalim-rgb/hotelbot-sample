from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from rag_core import RAGPipeline
from sql_core import SQLPipeline, SQLAnswer


Route = Literal["sql", "rag", "sql+rag"]


router_prompt = ChatPromptTemplate.from_template(
    """
You are a routing classifier deciding whether a question should be answered using SQL or RAG.

Always choose **SQL** when:
- The question asks for counts (e.g., "how many...", "number of...", "count all...")
- The question asks to list items (e.g., "list hotels", "show all hotels", "names of...")
- The question asks about any data found in database tables
- The question can be answered directly from structured data

Choose **RAG** only when:
- The question requires long-form description, comparison, hotel features, amenities, views, areas, etc.
- The answer must come from PDF documents and not from SQL tables.

Classify this question strictly as either "sql" or "rag":

Question: {question}
""".strip()
)
@dataclass
class HybridAnswer:
    route: Route
    answer: str
    sql_query: Optional[str] = None
    sql_raw_result: Optional[str] = None


class HybridQAPipeline:
    """
    Orchestrates between:
    - SQLPipeline (Text-to-SQL over MySQL)
    - RAGPipeline (vector search + LLM over documents)
    """

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

        self.router_llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        self.sql_pipeline = SQLPipeline()
        self.rag_pipeline = RAGPipeline()

    def _route(self, question: str) -> Route:
        msg = router_prompt.format(question=question)
        route = self.router_llm.invoke(msg).content.strip().lower()
        if route not in {"sql", "rag", "sql+rag"}:
            # Fallback to rag for safety
            return "rag"
        return route  # type: ignore[return-value]

    def _answer_with_sql(self, question: str) -> HybridAnswer:
        try:
            sql_result: SQLAnswer = self.sql_pipeline.ask_sql(question)
        except Exception as exc:
            # If SQL generation/execution fails, fall back to RAG so the user still gets an answer.
            rag_fallback = self.rag_pipeline.ask(question)
            return HybridAnswer(
                route="rag",
                answer=f"(SQL route failed: {exc})\n\n{rag_fallback}",
            )

        # Ask the LLM to explain the SQL result in natural language
        explanation_prompt = ChatPromptTemplate.from_template(
            """
You are given a user's question and the raw result of an SQL query that answers it.
Explain the answer clearly and concisely in natural language.

Question:
{question}

SQL query:
{sql_query}

SQL result:
{sql_result}

Natural language answer:
""".strip()
        )
        msg = explanation_prompt.format(
            question=question,
            sql_query=sql_result.sql,
            sql_result=str(sql_result.raw_result),
        )
        answer_text = self.router_llm.invoke(msg).content.strip()

        return HybridAnswer(
            route="sql",
            answer=answer_text,
            sql_query=sql_result.sql,
            sql_raw_result=str(sql_result.raw_result),
        )

    def _answer_with_rag(self, question: str) -> HybridAnswer:
        answer_text = self.rag_pipeline.ask(question)
        return HybridAnswer(route="rag", answer=answer_text)

    def _answer_with_sql_and_rag(self, question: str) -> HybridAnswer:
        # Get numeric / tabular data from SQL
        sql_result: SQLAnswer = self.sql_pipeline.ask_sql(question)
        sql_str = str(sql_result.raw_result)

        # Get contextual information from RAG
        rag_answer = self.rag_pipeline.ask(question)

        # Compose a final answer using both
        combo_prompt = ChatPromptTemplate.from_template(
            """
You are given:
- A user's question
- The result of an SQL query that computes exact numeric values relevant to the question
- A contextual answer from a RAG system over documents

Combine them into a single, coherent answer:
- Use the SQL result as the source of truth for any numeric values.
- Use the RAG answer only for qualitative/contextual explanation.
- If there is a conflict, trust the SQL numbers.

Question:
{question}

SQL query:
{sql_query}

SQL result:
{sql_result}

RAG context answer:
{rag_answer}

Final answer:
""".strip()
        )
        msg = combo_prompt.format(
            question=question,
            sql_query=sql_result.sql,
            sql_result=sql_str,
            rag_answer=rag_answer,
        )
        final_answer = self.router_llm.invoke(msg).content.strip()

        return HybridAnswer(
            route="sql+rag",
            answer=final_answer,
            sql_query=sql_result.sql,
            sql_raw_result=sql_str,
        )

    def ask(self, question: str) -> HybridAnswer:
        route = self._route(question)

        if route == "sql":
            return self._answer_with_sql(question)
        if route == "sql+rag":
            return self._answer_with_sql_and_rag(question)
        # default: rag
        return self._answer_with_rag(question)



