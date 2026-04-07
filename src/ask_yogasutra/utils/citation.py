"""
Shared citation-aware wrappers for LlamaIndex query and conversation engines.

Both GraphRAG and LinearRAG backends use these wrappers to append sutra
reference IDs to every response, giving the user traceability back to
the source nodes in the knowledge base.
"""

from typing import List, Dict, Any
from llama_index.core import Response
from llama_index.core.base.llms.types import ChatMessage


class CitationQueryEngine:
    """Wraps a LlamaIndex query engine to append source sutra IDs to responses."""

    def __init__(self, base_query_engine):
        self.base_query_engine = base_query_engine

    def query(self, query_str: str) -> Response:
        response = self.base_query_engine.query(query_str)
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []

        referenced_ids = set()
        for node in source_nodes:
            if hasattr(node, 'metadata') and 'id' in node.metadata:
                referenced_ids.add(node.metadata['id'])

        formatted_response = f"{response.response}\n\nReferences: {', '.join(sorted(referenced_ids))}"
        return Response(response=formatted_response, source_nodes=response.source_nodes)


class CitationConversationEngine:
    """Wraps a LlamaIndex chat engine to append source sutra IDs to responses."""

    def __init__(self, base_conversation_engine):
        self.base_conversation_engine = base_conversation_engine

    def query(self, query: str, msgs: List[ChatMessage]) -> Response:
        response = self.base_conversation_engine.chat(query, msgs)
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        return Response(response=response.response, source_nodes=source_nodes)


def extract_text_from_node(node_data: Dict[str, Any]) -> tuple:
    """Extract Sanskrit text fields and essential metadata from a Yogasutra graph node.

    Only the three most information-dense fields are indexed to keep chunk
    size manageable and avoid embedding noise from less relevant fields.

    Returns:
        (text, metadata) where text is a concatenation of the relevant fields
        and metadata carries the sutra ID, number, and chapter for citation.
    """
    relevant_fields = [
        'Sanskrit_Text',
        'Word_for_Word_Analysis',
        'Vyasa_commentary'
    ]

    text_parts = []
    essential_metadata = {
        'id': node_data.get('id'),
        'sutra_number': node_data.get('sutra_number', ''),
        'chapter': node_data.get('chapter', '')
    }

    for field in relevant_fields:
        if field in node_data and node_data[field]:
            text_parts.append(f"{field}: {node_data[field]}")

    return " ".join(text_parts), essential_metadata
