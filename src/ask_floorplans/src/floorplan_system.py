"""
Main Floor Plan Management System
Integrates storage, retrieval, and AI copilot functionality
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
from src.storage.vector_store import FloorPlanVectorSearch
from src.retrieval.rag_engine import RAGEngine, HybridSearch
from src.generation.copilot import ArchitecturalCopilot
from src.schemas.floor_plan_schema import FloorPlan

load_dotenv()


class FloorPlanSystem:
    """
    Complete floor plan management system with:
    - Semantic storage
    - RAG-based retrieval
    - AI copilot for generation and modification
    """
    
    def __init__(
        self,
        vector_store_type: str = "faiss",
        index_name: str = "floor-plans",
        api_key: Optional[str] = None
    ):
        """
        Initialize the floor plan system
        
        Args:
            vector_store_type: 'faiss' or 'pinecone'
            index_name: Name of the vector index
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize components
        self.vector_search = FloorPlanVectorSearch(
            store_type=vector_store_type,
            index_name=index_name
        )
        
        self.rag_engine = RAGEngine(
            vector_search=self.vector_search,
            api_key=self.api_key
        )
        
        self.hybrid_search = HybridSearch(
            vector_search=self.vector_search
        )
        
        self.copilot = ArchitecturalCopilot(api_key=self.api_key)
        
        # In-memory plan cache
        self.plan_cache = {}
    
    def store_floor_plan(
        self, 
        plan_data: Dict,
        validate: bool = True
    ) -> str:
        """
        Store a floor plan with semantic metadata
        
        Args:
            plan_data: Floor plan dictionary
            validate: Whether to validate against schema
            
        Returns:
            Floor plan ID
        """
        # Validate if requested
        if validate:
            try:
                validated_plan = FloorPlan(**plan_data)
                plan_data = validated_plan.to_dict()
            except Exception as e:
                raise ValueError(f"Floor plan validation failed: {e}")
        
        # Ensure plan has an ID
        if 'id' not in plan_data:
            plan_data['id'] = f"plan_{int(datetime.utcnow().timestamp())}"
        
        plan_id = plan_data['id']
        
        # Index in vector store
        self.vector_search.index_floor_plan(plan_data)
        
        # Cache the plan
        self.plan_cache[plan_id] = plan_data
        
        print(f"✓ Stored floor plan: {plan_data.get('name', plan_id)}")
        
        return plan_id
    
    def query(
        self, 
        query_text: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Query floor plans using natural language
        
        Args:
            query_text: Natural language query
            k: Number of results to return
            filters: Additional filters
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of matching floor plans with metadata
        """
        if use_hybrid:
            results = self.hybrid_search.search(query_text, k=k)
        else:
            rag_results = self.rag_engine.search(
                query_text, 
                k=k,
                return_context=False
            )
            results = rag_results['results']
        
        # Enrich results with full plan data if cached
        enriched_results = []
        for result in results:
            plan_id = result['id']
            if plan_id in self.plan_cache:
                result['floor_plan'] = self.plan_cache[plan_id]
            enriched_results.append(result)
        
        return enriched_results
    
    def answer_question(self, question: str) -> str:
        """
        Answer questions about floor plans using RAG
        
        Args:
            question: Natural language question
            
        Returns:
            Generated answer
        """
        return self.rag_engine.answer_question(question)
    
    def generate_floor_plan(
        self, 
        description: str,
        style: str = "modern"
    ) -> Dict:
        """
        Generate a new floor plan from natural language description
        
        Args:
            description: Natural language description
            style: Design style (modern, traditional, minimalist, etc.)
            
        Returns:
            Generated floor plan dictionary
        """
        result = self.copilot.process_command(description)
        floor_plan = result['floor_plan']
        
        # Store the generated plan
        self.store_floor_plan(floor_plan, validate=False)
        
        print(f"✓ Generated: {result['summary']}")
        
        return floor_plan
    
    def modify_floor_plan(
        self, 
        plan_id: str,
        command: str
    ) -> Dict:
        """
        Modify an existing floor plan using natural language
        
        Args:
            plan_id: ID of floor plan to modify
            command: Natural language modification command
            
        Returns:
            Modified floor plan dictionary
        """
        # Get current plan
        if plan_id not in self.plan_cache:
            raise ValueError(f"Floor plan {plan_id} not found in cache")
        
        current_plan = self.plan_cache[plan_id]
        
        # Apply modification
        result = self.copilot.process_command(command, current_plan)
        modified_plan = result['floor_plan']
        
        # Update storage
        self.store_floor_plan(modified_plan, validate=False)
        
        print(f"✓ Modified: {result['summary']}")
        
        return modified_plan
    
    def get_floor_plan(self, plan_id: str) -> Optional[Dict]:
        """
        Retrieve a floor plan by ID
        
        Args:
            plan_id: Floor plan ID
            
        Returns:
            Floor plan dictionary or None
        """
        return self.plan_cache.get(plan_id)
    
    def list_floor_plans(self) -> List[Dict]:
        """
        List all stored floor plans
        
        Returns:
            List of floor plan summaries
        """
        return [
            {
                'id': plan_id,
                'name': plan.get('name', 'Unnamed'),
                'total_area': plan.get('total_area', 0),
                'bedroom_count': sum(
                    1 for r in plan.get('rooms', [])
                    if r['type'] == 'bedroom'
                ),
                'bathroom_count': sum(
                    1 for r in plan.get('rooms', [])
                    if r['type'] == 'bathroom'
                )
            }
            for plan_id, plan in self.plan_cache.items()
        ]
    
    def export_floor_plan(
        self, 
        plan_id: str,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export floor plan to various formats
        
        Args:
            plan_id: Floor plan ID
            format: Export format (json, svg, dxf)
            output_path: Optional output file path
            
        Returns:
            Exported content or file path
        """
        plan = self.get_floor_plan(plan_id)
        if not plan:
            raise ValueError(f"Floor plan {plan_id} not found")
        
        if format == "json":
            content = json.dumps(plan, indent=2)
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(content)
                return output_path
            
            return content
        
        elif format == "svg":
            # Would use SVG converter here
            raise NotImplementedError("SVG export not yet implemented")
        
        elif format == "dxf":
            # Would use DXF converter here
            raise NotImplementedError("DXF export not yet implemented")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def batch_import(self, plans: List[Dict]) -> List[str]:
        """
        Import multiple floor plans at once
        
        Args:
            plans: List of floor plan dictionaries
            
        Returns:
            List of imported plan IDs
        """
        imported_ids = []
        
        for plan in plans:
            try:
                plan_id = self.store_floor_plan(plan, validate=False)
                imported_ids.append(plan_id)
            except Exception as e:
                print(f"✗ Failed to import plan: {e}")
        
        print(f"✓ Imported {len(imported_ids)} of {len(plans)} floor plans")
        
        return imported_ids
    
    def stats(self) -> Dict:
        """
        Get system statistics
        
        Returns:
            Statistics dictionary
        """
        plans = list(self.plan_cache.values())
        
        if not plans:
            return {
                'total_plans': 0,
                'total_rooms': 0,
                'avg_area': 0,
                'bedroom_distribution': {},
                'bathroom_distribution': {}
            }
        
        bedroom_counts = {}
        bathroom_counts = {}
        total_rooms = 0
        total_area = 0
        
        for plan in plans:
            rooms = plan.get('rooms', [])
            total_rooms += len(rooms)
            total_area += plan.get('total_area', 0)
            
            bedrooms = sum(1 for r in rooms if r['type'] == 'bedroom')
            bathrooms = sum(1 for r in rooms if r['type'] == 'bathroom')
            
            bedroom_counts[bedrooms] = bedroom_counts.get(bedrooms, 0) + 1
            bathroom_counts[bathrooms] = bathroom_counts.get(bathrooms, 0) + 1
        
        return {
            'total_plans': len(plans),
            'total_rooms': total_rooms,
            'avg_area': round(total_area / len(plans), 2),
            'bedroom_distribution': bedroom_counts,
            'bathroom_distribution': bathroom_counts
        }


# Example usage
if __name__ == "__main__":
    from src.schemas.floor_plan_schema import EXAMPLE_FLOOR_PLAN
    
    print("=" * 70)
    print("FLOOR PLAN MANAGEMENT SYSTEM - DEMO")
    print("=" * 70)
    
    # Initialize system
    print("\n1. Initializing system...")
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Store example floor plan
    print("\n2. Storing example floor plan...")
    plan_id = system.store_floor_plan(EXAMPLE_FLOOR_PLAN)
    
    # Query floor plans
    print("\n3. Querying floor plans...")
    queries = [
        "2 bedroom apartments",
        "apartments under 1000 sq ft",
        "plans with balcony"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = system.query(query, k=3)
        print(f"   Found {len(results)} results")
        for r in results:
            print(f"     - {r['metadata']['name']} (score: {r.get('score', 0):.3f})")
    
    # Generate new floor plan
    print("\n4. Generating new floor plan...")
    new_plan = system.generate_floor_plan(
        "Create a 3 bedroom apartment with ensuite master, open kitchen, 1200 sq ft"
    )
    
    # Modify floor plan
    print("\n5. Modifying floor plan...")
    modified_plan = system.modify_floor_plan(
        new_plan['id'],
        "Add a balcony to the living room"
    )
    
    # List all plans
    print("\n6. Listing all floor plans...")
    all_plans = system.list_floor_plans()
    for plan in all_plans:
        print(f"   - {plan['name']}: {plan['bedroom_count']}BR, "
              f"{plan['bathroom_count']}BA, {plan['total_area']:.1f} sqm")
    
    # Show statistics
    print("\n7. System statistics:")
    stats = system.stats()
    print(f"   Total plans: {stats['total_plans']}")
    print(f"   Total rooms: {stats['total_rooms']}")
    print(f"   Average area: {stats['avg_area']} sqm")
    print(f"   Bedroom distribution: {stats['bedroom_distribution']}")
    print(f"   Bathroom distribution: {stats['bathroom_distribution']}")
    
    # Answer question
    print("\n8. Answering question with RAG...")
    question = "What apartments do you have available?"
    answer = system.answer_question(question)
    print(f"   Q: {question}")
    print(f"   A: {answer}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
