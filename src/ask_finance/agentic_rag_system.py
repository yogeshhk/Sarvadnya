"""
Agentic RAG system using LangGraph for multi-modal financial document analysis.

Workflow:
  user query
    → analyze_query  : classify intent (text / table / image / mixed)
    → route_query    : decide which retrieval paths to activate
    → retrieve_text  : semantic vector search (always runs)
    → retrieve_table : SQL + vector search (if table intent detected)
    → retrieve_image : image description search (if image intent detected)
    → generate_answer: synthesise all retrieved context into a final answer

Each node receives and returns an AgentState dict, as required by LangGraph.
"""

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from document_parser import Chunk, load_chunks
from multimodal_embeddings import MultiModalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Pydantic schemas
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"


class QueryIntent(BaseModel):
    """Structured output produced by the query-analysis node."""
    primary_intent: QueryType = Field(description="Primary content type to search")
    secondary_intents: List[QueryType] = Field(default=[], description="Secondary content types")
    requires_calculation: bool = Field(default=False, description="Whether query needs calculations")
    requires_comparison: bool = Field(default=False, description="Whether query needs comparisons")
    key_entities: List[str] = Field(default=[], description="Key financial entities mentioned")
    temporal_scope: Optional[str] = Field(default=None, description="Time period, if specified")


class SQLQuery(BaseModel):
    """Structured SQL query produced by the table-retrieval node."""
    query: str = Field(description="SQL query to execute")
    explanation: str = Field(description="Human-readable explanation of the query")
    confidence: float = Field(description="Confidence score (0-1)")


class AgentState(BaseModel):
    """Shared state passed between every node in the LangGraph workflow."""
    messages: List[Any] = Field(default_factory=list)
    user_query: str = ""
    query_intent: Optional[QueryIntent] = None
    retrieved_chunks: List[Dict] = Field(default_factory=list)
    table_results: List[Dict] = Field(default_factory=list)
    image_descriptions: List[str] = Field(default_factory=list)
    sql_queries: List[Dict] = Field(default_factory=list)
    final_answer: str = ""
    metadata: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FinancialRAGAgent:
    """Multi-modal RAG agent for financial documents."""

    def __init__(
        self,
        vector_store: MultiModalVectorStore,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
    ):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=model_name,
            temperature=0.1,
        )
        self.intent_parser = PydanticOutputParser(pydantic_object=QueryIntent)
        self.sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node: analyze_query
    # ------------------------------------------------------------------

    async def analyze_query(self, state: AgentState) -> AgentState:
        """Classify the user query into a QueryIntent."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a financial document analyst. "
                "Classify the user's query and respond ONLY with valid JSON "
                "matching this schema:\n"
                + self.intent_parser.get_format_instructions()
            )),
            HumanMessage(content=state.user_query),
        ])
        chain = prompt | self.llm | self.intent_parser
        try:
            intent: QueryIntent = await chain.ainvoke({})
        except Exception as e:
            logger.warning("Intent classification failed (%s); defaulting to TEXT.", e)
            intent = QueryIntent(primary_intent=QueryType.TEXT)

        state.query_intent = intent
        state.messages.append(
            AIMessage(content=f"Query classified as: {intent.primary_intent}")
        )
        return state

    # ------------------------------------------------------------------
    # Node: route_query
    # ------------------------------------------------------------------

    async def route_query(self, state: AgentState) -> AgentState:
        """Decide which downstream retrieval nodes to activate.

        LangGraph uses the return value of the *conditional* edge function
        (see _route_decision below) to branch, not this node's output.
        This node just logs the routing decision.
        """
        intent = state.query_intent
        targets = ["retrieve_text"]
        if intent and intent.primary_intent in (QueryType.TABLE, QueryType.MIXED):
            targets.append("retrieve_table")
        if intent and intent.primary_intent in (QueryType.IMAGE, QueryType.MIXED):
            targets.append("retrieve_image")

        state.metadata["routing_targets"] = targets
        logger.info("Routing to: %s", targets)
        return state

    def _route_decision(self, state: AgentState) -> str:
        """Return the next node name based on routing metadata.

        LangGraph calls this as the conditional edge function after route_query.
        It must return one of the node names registered in the graph.
        """
        targets: List[str] = state.metadata.get("routing_targets", ["retrieve_text"])
        # We run retrieve_text first; table/image retrieval happens inside
        # retrieve_text if needed (simpler than dynamic fan-out for a PoC).
        return "retrieve_text"

    # ------------------------------------------------------------------
    # Node: retrieve_text
    # ------------------------------------------------------------------

    async def retrieve_text(self, state: AgentState) -> AgentState:
        """Semantic vector search for relevant text chunks."""
        try:
            results = await self.vector_store.search(
                state.user_query,
                content_types=["text"],
                n_results=5,
            )
            state.retrieved_chunks.extend(results)
        except Exception as e:
            logger.error("Text retrieval failed: %s", e)

        # Also trigger table/image retrieval if they were routed
        targets = state.metadata.get("routing_targets", [])
        if "retrieve_table" in targets:
            state = await self.retrieve_table(state)
        if "retrieve_image" in targets:
            state = await self.retrieve_image(state)

        return state

    # ------------------------------------------------------------------
    # Node: retrieve_table
    # ------------------------------------------------------------------

    async def retrieve_table(self, state: AgentState) -> AgentState:
        """Vector search restricted to table chunks, plus a SQL query attempt."""
        try:
            table_chunks = await self.vector_store.search(
                state.user_query,
                content_types=["table"],
                n_results=3,
            )
            state.retrieved_chunks.extend(table_chunks)

            # Attempt to generate and run a SQL query for structured data
            if table_chunks:
                sql_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=(
                        "Generate a SQL query that answers the user's question "
                        "given this table schema. Respond with JSON only.\n"
                        + self.sql_parser.get_format_instructions()
                    )),
                    HumanMessage(content=(
                        f"Question: {state.user_query}\n"
                        f"Table metadata: {json.dumps(table_chunks[0]['metadata'])}"
                    )),
                ])
                chain = sql_prompt | self.llm | self.sql_parser
                sql_obj: SQLQuery = await chain.ainvoke({})

                table_name = table_chunks[0]["metadata"].get("sql_table_name")
                if table_name:
                    rows = self.vector_store.execute_sql_query(sql_obj.query, table_name)
                    state.table_results.extend(rows)
                    state.sql_queries.append({
                        "query": sql_obj.query,
                        "explanation": sql_obj.explanation,
                        "results": rows,
                    })
        except Exception as e:
            logger.error("Table retrieval failed: %s", e)

        return state

    # ------------------------------------------------------------------
    # Node: retrieve_image
    # ------------------------------------------------------------------

    async def retrieve_image(self, state: AgentState) -> AgentState:
        """Vector search restricted to image chunks."""
        try:
            image_chunks = await self.vector_store.search(
                state.user_query,
                content_types=["image"],
                n_results=3,
            )
            for chunk in image_chunks:
                desc = chunk["metadata"].get("description", "Financial chart")
                state.image_descriptions.append(desc)
            state.retrieved_chunks.extend(image_chunks)
        except Exception as e:
            logger.error("Image retrieval failed: %s", e)

        return state

    # ------------------------------------------------------------------
    # Node: generate_answer
    # ------------------------------------------------------------------

    async def generate_answer(self, state: AgentState) -> AgentState:
        """Synthesise all retrieved context into the final answer."""
        context_parts: List[str] = []

        # Text chunks
        for chunk in state.retrieved_chunks:
            if chunk.get("metadata", {}).get("content_type") == "text":
                context_parts.append(chunk["content"][:500])

        # Table results
        if state.table_results:
            context_parts.append(
                "Table data:\n" + json.dumps(state.table_results[:5], indent=2)
            )

        # Image descriptions
        if state.image_descriptions:
            context_parts.append(
                "Visual content:\n" + "\n".join(state.image_descriptions)
            )

        context = "\n\n---\n\n".join(context_parts) or "No relevant context found."

        synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a financial analyst. Answer the question using ONLY "
                "the provided context. Be precise and cite specific figures."
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state.user_query}"),
        ])
        chain = synthesis_prompt | self.llm
        response = await chain.ainvoke({})
        state.final_answer = response.content
        state.messages.append(AIMessage(content=state.final_answer))
        return state

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Wire up the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("retrieve_text", self.retrieve_text)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "route_query")
        workflow.add_conditional_edges(
            "route_query",
            self._route_decision,
            {"retrieve_text": "retrieve_text"},
        )
        workflow.add_edge("retrieve_text", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query through the full agentic workflow.

        Args:
            query: Natural-language question about the financial documents.

        Returns:
            Dict with keys: answer, retrieved_chunks, table_results,
            sql_queries, image_descriptions, query_intent.
        """
        initial_state = AgentState(
            user_query=query,
            messages=[HumanMessage(content=query)],
        )
        final_state: AgentState = await self.graph.ainvoke(initial_state)
        return {
            "answer": final_state.final_answer,
            "retrieved_chunks": final_state.retrieved_chunks,
            "table_results": final_state.table_results,
            "sql_queries": final_state.sql_queries,
            "image_descriptions": final_state.image_descriptions,
            "query_intent": (
                final_state.query_intent.dict() if final_state.query_intent else None
            ),
        }
