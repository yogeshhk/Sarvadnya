"""
Basic usage examples for the Floor Plan AI System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_system import FloorPlanSystem
import json


def example_1_storing_and_retrieving():
    """Example 1: Store and retrieve floor plans"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Storing and Retrieving Floor Plans")
    print("="*70)
    
    # Initialize system
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Create a simple floor plan
    floor_plan = {
        "id": "apartment_101",
        "name": "Studio Apartment - Downtown",
        "total_area": 45.0,
        "level": 1,
        "metadata": {
            "version": "1.0",
            "created": "2025-01-15T10:00:00Z",
            "modified": "2025-01-15T10:00:00Z",
            "created_by": "user",
            "tags": ["studio", "downtown", "compact"]
        },
        "rooms": [
            {
                "id": "room_001",
                "type": "living_room",
                "area": 25.0,
                "dimensions": {"length": 5.0, "width": 5.0, "unit": "m"},
                "features": {
                    "windows": 2,
                    "doors": 1,
                    "balcony": False
                },
                "floor_level": 0
            },
            {
                "id": "room_002",
                "type": "kitchen",
                "area": 8.0,
                "dimensions": {"length": 4.0, "width": 2.0, "unit": "m"},
                "features": {
                    "windows": 1,
                    "doors": 1
                },
                "floor_level": 0
            },
            {
                "id": "room_003",
                "type": "bathroom",
                "area": 4.0,
                "dimensions": {"length": 2.0, "width": 2.0, "unit": "m"},
                "features": {
                    "windows": 1,
                    "doors": 1
                },
                "floor_level": 0
            }
        ],
        "adjacencies": [],
        "zones": [],
        "constraints": {
            "code_requirements": [],
            "design_notes": "Compact studio design"
        }
    }
    
    # Store the floor plan
    plan_id = system.store_floor_plan(floor_plan, validate=False)
    print(f"✓ Stored floor plan with ID: {plan_id}")
    
    # Retrieve it
    retrieved = system.get_floor_plan(plan_id)
    print(f"✓ Retrieved: {retrieved['name']}")
    print(f"  Area: {retrieved['total_area']} sqm")
    print(f"  Rooms: {len(retrieved['rooms'])}")


def example_2_natural_language_queries():
    """Example 2: Natural language queries"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Natural Language Queries")
    print("="*70)
    
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Add some sample floor plans
    sample_plans = [
        {
            "id": "plan_2br_1",
            "name": "2BR Modern Apartment",
            "total_area": 85.0,
            "level": 0,
            "metadata": {
                "version": "1.0",
                "created": "2025-01-15T10:00:00Z",
                "modified": "2025-01-15T10:00:00Z",
                "created_by": "system",
                "tags": ["2br", "modern"]
            },
            "rooms": [
                {"id": "r1", "type": "bedroom", "area": 15.0, 
                 "dimensions": {"length": 4.0, "width": 3.75, "unit": "m"},
                 "features": {"windows": 2, "doors": 1}, "floor_level": 0},
                {"id": "r2", "type": "bedroom", "area": 12.0,
                 "dimensions": {"length": 4.0, "width": 3.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r3", "type": "bathroom", "area": 5.0,
                 "dimensions": {"length": 2.5, "width": 2.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r4", "type": "kitchen", "area": 10.0,
                 "dimensions": {"length": 5.0, "width": 2.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r5", "type": "living_room", "area": 20.0,
                 "dimensions": {"length": 5.0, "width": 4.0, "unit": "m"},
                 "features": {"windows": 2, "doors": 1, "balcony": True}, "floor_level": 0},
            ],
            "adjacencies": [],
            "zones": [],
            "constraints": {"code_requirements": [], "design_notes": "Modern design"}
        },
        {
            "id": "plan_3br_1",
            "name": "3BR Family Home",
            "total_area": 120.0,
            "level": 0,
            "metadata": {
                "version": "1.0",
                "created": "2025-01-15T10:00:00Z",
                "modified": "2025-01-15T10:00:00Z",
                "created_by": "system",
                "tags": ["3br", "family"]
            },
            "rooms": [
                {"id": "r1", "type": "bedroom", "area": 18.0,
                 "dimensions": {"length": 4.5, "width": 4.0, "unit": "m"},
                 "features": {"windows": 2, "doors": 1, "ensuite": True}, "floor_level": 0},
                {"id": "r2", "type": "bedroom", "area": 12.0,
                 "dimensions": {"length": 4.0, "width": 3.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r3", "type": "bedroom", "area": 12.0,
                 "dimensions": {"length": 4.0, "width": 3.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r4", "type": "bathroom", "area": 6.0,
                 "dimensions": {"length": 3.0, "width": 2.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r5", "type": "bathroom", "area": 4.0,
                 "dimensions": {"length": 2.0, "width": 2.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r6", "type": "kitchen", "area": 12.0,
                 "dimensions": {"length": 6.0, "width": 2.0, "unit": "m"},
                 "features": {"windows": 1, "doors": 1}, "floor_level": 0},
                {"id": "r7", "type": "living_room", "area": 25.0,
                 "dimensions": {"length": 5.0, "width": 5.0, "unit": "m"},
                 "features": {"windows": 3, "doors": 1}, "floor_level": 0},
            ],
            "adjacencies": [],
            "zones": [],
            "constraints": {"code_requirements": [], "design_notes": "Family oriented"}
        }
    ]
    
    # Import plans
    system.batch_import(sample_plans)
    
    # Try various queries
    queries = [
        "Find 2 bedroom apartments",
        "Show me apartments under 1000 sq ft",
        "I need a 3 bedroom house",
        "Find plans with balconies"
    ]
    
    for query in queries:
        print(f"\n📋 Query: '{query}'")
        results = system.query(query, k=5)
        
        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                print(f"   {i}. {meta['name']}")
                print(f"      {meta['bedroom_count']}BR, {meta['bathroom_count']}BA, "
                      f"{meta['total_area']:.0f} sqm")
        else:
            print("   No results found")


def example_3_ai_generation():
    """Example 3: AI-powered floor plan generation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: AI-Powered Floor Plan Generation")
    print("="*70)
    
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Generate floor plans from descriptions
    descriptions = [
        "Create a 2 bedroom apartment with open kitchen and living room, 900 sq ft",
        "Design a 1 bedroom studio with kitchenette, about 500 sq ft",
        "Generate a 4 bedroom family home with 2.5 bathrooms, 2000 sq ft"
    ]
    
    for desc in descriptions:
        print(f"\n🏗️  Generating: '{desc}'")
        
        try:
            plan = system.generate_floor_plan(desc)
            print(f"   ✓ Created: {plan['name']}")
            print(f"     Total area: {plan['total_area']:.1f} sqm")
            print(f"     Rooms: {len(plan['rooms'])}")
            
            # List rooms
            room_summary = {}
            for room in plan['rooms']:
                room_type = room['type']
                room_summary[room_type] = room_summary.get(room_type, 0) + 1
            
            print(f"     Breakdown: ", end="")
            print(", ".join(f"{count} {rtype}" for rtype, count in room_summary.items()))
            
        except Exception as e:
            print(f"   ✗ Generation failed: {e}")


def example_4_modification():
    """Example 4: Modifying existing floor plans"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Modifying Existing Floor Plans")
    print("="*70)
    
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Generate a base floor plan
    print("\n1. Creating base floor plan...")
    base_plan = system.generate_floor_plan(
        "Create a 2 bedroom apartment with 1 bathroom, 850 sq ft"
    )
    
    plan_id = base_plan['id']
    print(f"   ✓ Created plan: {plan_id}")
    
    # Apply modifications
    modifications = [
        "Add a balcony to the living room",
        "Add another bathroom",
        "Increase the master bedroom size by 20%"
    ]
    
    print("\n2. Applying modifications...")
    for mod in modifications:
        print(f"\n   🔧 Modification: '{mod}'")
        try:
            modified = system.modify_floor_plan(plan_id, mod)
            print(f"      ✓ Applied successfully")
            print(f"      New total area: {modified['total_area']:.1f} sqm")
        except Exception as e:
            print(f"      ✗ Failed: {e}")


def example_5_rag_qa():
    """Example 5: RAG-based question answering"""
    print("\n" + "="*70)
    print("EXAMPLE 5: RAG-Based Question Answering")
    print("="*70)
    
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Add some floor plans
    sample_plans = [
        {
            "id": "luxury_1",
            "name": "Luxury Penthouse",
            "total_area": 200.0,
            "level": 10,
            "metadata": {
                "version": "1.0",
                "created": "2025-01-15T10:00:00Z",
                "modified": "2025-01-15T10:00:00Z",
                "created_by": "system",
                "tags": ["luxury", "penthouse", "3br"]
            },
            "rooms": [
                {"id": "r1", "type": "bedroom", "area": 25.0,
                 "dimensions": {"length": 5.0, "width": 5.0, "unit": "m"},
                 "features": {"windows": 3, "doors": 1, "ensuite": True, "walk_in_closet": True},
                 "floor_level": 0},
                {"id": "r2", "type": "bedroom", "area": 15.0,
                 "dimensions": {"length": 5.0, "width": 3.0, "unit": "m"},
                 "features": {"windows": 2, "doors": 1}, "floor_level": 0},
                {"id": "r3", "type": "bedroom", "area": 15.0,
                 "dimensions": {"length": 5.0, "width": 3.0, "unit": "m"},
                 "features": {"windows": 2, "doors": 1}, "floor_level": 0},
            ],
            "adjacencies": [],
            "zones": [],
            "constraints": {"code_requirements": [], "design_notes": "Luxury finishes"}
        }
    ]
    
    system.batch_import(sample_plans)
    
    # Ask questions
    questions = [
        "What luxury apartments do you have?",
        "Tell me about the penthouse",
        "Which floor plans have walk-in closets?",
        "What's the largest apartment available?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: '{question}'")
        
        try:
            answer = system.answer_question(question)
            print(f"💬 Answer: {answer}")
        except Exception as e:
            print(f"✗ Error: {e}")


def example_6_stats_and_export():
    """Example 6: Statistics and export"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Statistics and Export")
    print("="*70)
    
    system = FloorPlanSystem(vector_store_type="faiss")
    
    # Generate some plans
    print("\n1. Generating sample plans...")
    for i in range(5):
        bedrooms = (i % 3) + 1
        system.generate_floor_plan(
            f"Create a {bedrooms} bedroom apartment, 800 sq ft"
        )
    
    # Get statistics
    print("\n2. System statistics:")
    stats = system.stats()
    
    print(f"   Total plans: {stats['total_plans']}")
    print(f"   Total rooms: {stats['total_rooms']}")
    print(f"   Average area: {stats['avg_area']} sqm")
    print(f"   Bedroom distribution:")
    for br_count, plan_count in sorted(stats['bedroom_distribution'].items()):
        print(f"     {br_count} bedrooms: {plan_count} plans")
    
    # List all plans
    print("\n3. All floor plans:")
    all_plans = system.list_floor_plans()
    for i, plan in enumerate(all_plans, 1):
        print(f"   {i}. {plan['name']}")
        print(f"      {plan['bedroom_count']}BR, {plan['bathroom_count']}BA, "
              f"{plan['total_area']:.0f} sqm")
    
    # Export a plan
    if all_plans:
        plan_id = all_plans[0]['id']
        print(f"\n4. Exporting plan: {plan_id}")
        
        json_export = system.export_floor_plan(plan_id, format="json")
        print(f"   ✓ Exported to JSON ({len(json_export)} characters)")
        print(f"   Preview:\n{json_export[:200]}...")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FLOOR PLAN AI SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    examples = [
        example_1_storing_and_retrieving,
        example_2_natural_language_queries,
        example_3_ai_generation,
        example_4_modification,
        example_5_rag_qa,
        example_6_stats_and_export
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
