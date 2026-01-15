"""
RAG (Retrieval-Augmented Generation) Engine for Floor Plans
Combines vector search with LLM-based query understanding
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import openai

load_dotenv()


class QueryParser:
    """Parse natural language queries into structured constraints"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize query parser with OpenAI API"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def parse_query(self, query: str) -> Dict:
        """
        Parse natural language query into structured constraints
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with 'query_text', 'filters', and 'constraints'
        """
        # Use GPT to extract constraints
        system_prompt = """You are a query parser for architectural floor plan searches.
Extract structured information from natural language queries.

Return JSON with these fields:
- semantic_query: The semantic/conceptual part of the query
- filters: Dictionary of exact filters (bedroom_count, bathroom_count, total_area, etc.)
- room_types: List of room types mentioned
- features: List of features mentioned (balcony, open concept, etc.)

Example:
Query: "Find 2 bedroom apartments with 2 bathrooms under 1000 sq ft"
Response: {
  "semantic_query": "modern residential apartments",
  "filters": {
    "bedroom_count": 2,
    "bathroom_count": 2,
    "total_area": {"$lte": 92.9}
  },
  "room_types": ["bedroom", "bathroom"],
  "features": []
}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            parsed = json.loads(response.choices[0].message.content)
            return parsed
        
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback to simple parsing
            return self._simple_parse(query)
    
    def _simple_parse(self, query: str) -> Dict:
        """Simple rule-based parsing as fallback"""
        query_lower = query.lower()
        
        filters = {}
        features = []
        room_types = []
        
        # Extract bedroom count
        bedroom_match = re.search(r'(\d+)\s*bed', query_lower)
        if bedroom_match:
            filters['bedroom_count'] = int(bedroom_match.group(1))
            room_types.append('bedroom')
        
        # Extract bathroom count
        bathroom_match = re.search(r'(\d+)\s*(bath|washroom)', query_lower)
        if bathroom_match:
            filters['bathroom_count'] = int(bathroom_match.group(1))
            room_types.append('bathroom')
        
        # Extract area constraints
        area_patterns = [
            (r'under\s*(\d+)\s*(sq\s*ft|sqft|square\s*feet)', '$lte'),
            (r'less\s*than\s*(\d+)\s*(sq\s*ft|sqft)', '$lte'),
            (r'above\s*(\d+)\s*(sq\s*ft|sqft)', '$gte'),
            (r'more\s*than\s*(\d+)\s*(sq\s*ft|sqft)', '$gte'),
        ]
        
        for pattern, operator in area_patterns:
            match = re.search(pattern, query_lower)
            if match:
                area_sqft = int(match.group(1))
                area_sqm = area_sqft * 0.092903  # Convert to square meters
                filters['total_area'] = {operator: area_sqm}
                break
        
        # Detect features
        if 'balcony' in query_lower:
            features.append('balcony')
        if 'open concept' in query_lower or 'open kitchen' in query_lower:
            features.append('open_concept')
        if 'ensuite' in query_lower or 'attached bathroom' in query_lower:
            features.append('ensuite')
        
        return {
            'semantic_query': query,
            'filters': filters,
            'room_types': room_types,
            'features': features
        }


class RAGEngine:
    """Retrieval-Augmented Generation engine for floor plans"""
    
    def __init__(
        self, 
        vector_search,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_search: FloorPlanVectorSearch instance
            api_key: OpenAI API key
        """
        self.vector_search = vector_search
        self.query_parser = QueryParser(api_key)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        return_context: bool = False
    ) -> Dict:
        """
        Search for floor plans using RAG
        
        Args:
            query: Natural language query
            k: Number of results
            return_context: Whether to return full context
            
        Returns:
            Dict with results and optional context
        """
        # Parse query into structured constraints
        parsed = self.query_parser.parse_query(query)
        
        # Perform vector search
        results = self.vector_search.search(
            query=parsed.get('semantic_query', query),
            k=k,
            filters=parsed.get('filters')
        )
        
        # Enhance results with additional filtering
        filtered_results = self._filter_by_features(
            results, 
            parsed.get('features', [])
        )
        
        response = {
            'query': query,
            'parsed_query': parsed,
            'results': filtered_results,
            'count': len(filtered_results)
        }
        
        if return_context:
            response['context'] = self._generate_context(filtered_results)
        
        return response
    
    def _filter_by_features(
        self, 
        results: List[Dict], 
        features: List[str]
    ) -> List[Dict]:
        """Filter results by required features"""
        if not features:
            return results
        
        filtered = []
        for result in results:
            # Feature matching would require loading full plan
            # For now, just return all results
            filtered.append(result)
        
        return filtered
    
    def _generate_context(self, results: List[Dict]) -> str:
        """Generate context summary for LLM"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            context_parts.append(
                f"{i}. {metadata.get('name', 'Unnamed')} - "
                f"{metadata.get('total_area', 0):.1f} sqm, "
                f"{metadata.get('bedroom_count', 0)} bedrooms, "
                f"{metadata.get('bathroom_count', 0)} bathrooms"
            )
        
        return "\n".join(context_parts)
    
    def answer_question(
        self, 
        query: str, 
        k: int = 5
    ) -> str:
        """
        Answer a question about floor plans using RAG
        
        Args:
            query: Natural language question
            k: Number of plans to retrieve
            
        Returns:
            Generated answer
        """
        # Retrieve relevant floor plans
        search_results = self.search(query, k=k, return_context=True)
        
        # Generate answer using GPT
        system_prompt = """You are an architectural assistant helping users find floor plans.
Based on the retrieved floor plans, provide a helpful answer to the user's query.
Be specific and reference the floor plans by name."""
        
        user_prompt = f"""Query: {query}

Retrieved Floor Plans:
{search_results['context']}

Please provide a helpful answer based on these floor plans."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            return answer
        
        except Exception as e:
            return f"Error generating answer: {e}"


class HybridSearch:
    """Hybrid search combining vector similarity and exact filters"""
    
    def __init__(self, vector_search):
        """Initialize hybrid search"""
        self.vector_search = vector_search
        self.query_parser = QueryParser()
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Perform hybrid search
        
        Args:
            query: Natural language query
            k: Number of results
            rerank: Whether to rerank results
            
        Returns:
            Ranked list of results
        """
        # Parse query
        parsed = self.query_parser.parse_query(query)
        
        # Get more results for reranking
        initial_k = k * 3 if rerank else k
        
        # Vector search with filters
        results = self.vector_search.search(
            query=parsed.get('semantic_query', query),
            k=initial_k,
            filters=parsed.get('filters')
        )
        
        if not rerank:
            return results[:k]
        
        # Rerank based on exact matches
        reranked = self._rerank_results(results, parsed)
        
        return reranked[:k]
    
    def _rerank_results(
        self, 
        results: List[Dict], 
        parsed_query: Dict
    ) -> List[Dict]:
        """Rerank results based on query constraints"""
        scored_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            # Bonus for exact matches
            bonus = 0
            
            filters = parsed_query.get('filters', {})
            
            # Bedroom count match
            if 'bedroom_count' in filters:
                if metadata.get('bedroom_count') == filters['bedroom_count']:
                    bonus += 0.5
            
            # Bathroom count match
            if 'bathroom_count' in filters:
                if metadata.get('bathroom_count') == filters['bathroom_count']:
                    bonus += 0.5
            
            # Area constraints
            if 'total_area' in filters:
                area_filter = filters['total_area']
                area = metadata.get('total_area', 0)
                
                if isinstance(area_filter, dict):
                    matches = True
                    if '$lte' in area_filter and area > area_filter['$lte']:
                        matches = False
                    if '$gte' in area_filter and area < area_filter['$gte']:
                        matches = False
                    if matches:
                        bonus += 0.3
            
            # Adjust score (lower is better for L2 distance)
            adjusted_score = score - bonus
            
            scored_results.append({
                **result,
                'adjusted_score': adjusted_score,
                'bonus': bonus
            })
        
        # Sort by adjusted score
        scored_results.sort(key=lambda x: x['adjusted_score'])
        
        return scored_results


# Example usage
if __name__ == "__main__":
    from vector_store import FloorPlanVectorSearch
    from floor_plan_schema import EXAMPLE_FLOOR_PLAN
    
    # Initialize system
    vector_search = FloorPlanVectorSearch(store_type="faiss")
    vector_search.index_floor_plan(EXAMPLE_FLOOR_PLAN)
    
    # Initialize RAG engine
    rag = RAGEngine(vector_search)
    
    # Test queries
    test_queries = [
        "Find 2 bedroom apartments with 2 bathrooms",
        "Show me apartments under 1000 sq ft",
        "I need a 3 bedroom house with open kitchen",
        "What apartments do you have with balconies?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = rag.search(query, k=3)
        
        print(f"\nParsed query:")
        print(json.dumps(results['parsed_query'], indent=2))
        
        print(f"\nFound {results['count']} results:")
        for result in results['results']:
            metadata = result['metadata']
            print(f"\n  {metadata.get('name', 'Unnamed')}")
            print(f"    Area: {metadata.get('total_area', 0)} sqm")
            print(f"    Bedrooms: {metadata.get('bedroom_count', 0)}")
            print(f"    Bathrooms: {metadata.get('bathroom_count', 0)}")
            print(f"    Score: {result.get('score', 0):.3f}")
