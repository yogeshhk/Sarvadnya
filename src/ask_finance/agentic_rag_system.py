"""
Agentic RAG system using LangGraph for multi-modal financial document analysis
Includes router agent, query expansion, and specialized retrieval strategies
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import sqlparse

from document_parser import load_chunks, Chunk
from multimodal_embeddings import MultiModalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"

class QueryIntent(BaseModel):
    """Structured output for query analysis"""
    primary_intent: QueryType = Field(description="Primary type of content to search")
    secondary_intents: List[QueryType] = Field(default=[], description="Secondary content types")
    requires_calculation: bool = Field(default=False, description="Whether query needs calculations")
    requires_comparison: bool = Field(default=False, description="Whether query needs comparisons")
    key_entities: List[str] = Field(default=[], description="Key financial entities mentioned")
    temporal_scope: Optional[str] = Field(default=None, description="Time period if specified")

class SQLQuery(BaseModel):
    """Structured SQL query output"""
    query: str = Field(description="SQL query to execute")
    explanation: str = Field(description="Human-readable explanation of the query")
    confidence: float = Field(description="Confidence in query correctness (0-1)")

class AgentState(BaseModel):
    """State maintained across the agent workflow"""
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

class FinancialRAGAgent:
    """Multi-modal RAG agent for financial documents"""
    
    def __init__(self, vector_store: MultiModalVectorStore, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4-turbo-preview"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=0.1
        )
        
        # Initialize parsers
        self.intent_parser = PydanticOutputParser(pydantic_object=QueryIntent)
        self.sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("route_query", self.route