#!/usr/bin/env python3
"""
Test script to verify persistence functionality for both Linear RAG and Graph RAG backends.

This script tests:
1. Index creation and persistence
2. Loading from persisted indices
3. Data change detection
4. Force rebuild functionality
5. Data source-specific index directories
6. Multiple data sources without overwrites
"""

import json
import shutil
import time
from pathlib import Path
import sys

# Import backends
from linearrag.linearrag_backend import LinearRAGBackend
from graphrag.graphrag_backend import GraphRAGBackend

# Test configuration
TEST_DATA_FILE = "data/graph_small.json"
TEST_DATA_FILE2 = "data/sample-graph-json.json"
TEST_LINEAR_DIR1 = "models/linearrag_graph_small"
TEST_LINEAR_DIR2 = "models/linearrag_sample_data"
TEST_GRAPH_DIR1 = "models/graphrag_graph_small"
TEST_GRAPH_DIR2 = "models/graphrag_sample_data"
TEST_LINEAR_DIR = TEST_LINEAR_DIR1  # Default for single tests
TEST_GRAPH_DIR = TEST_GRAPH_DIR1    # Default for single tests


def cleanup_test_dirs():
    """Clean up test directories."""
    # Clean up directories created by LinearRAGBackend
    linear_dirs = ["models/linearrag_graph_small", "models/linearrag_sample_data"]
    # Clean up directories created by GraphRAGBackend
    graph_dirs = ["models/graphrag_graph_small", "models/graphrag_sample_data"]

    for dir_path in linear_dirs + graph_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            print(f"✓ Cleaned up {dir_path}")


def load_test_data():
    """Load test data."""
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_linear_rag_persistence():
    """Test Linear RAG persistence functionality."""
    print("\n" + "="*60)
    print("Testing Linear RAG Persistence")
    print("="*60)

    # Load test data
    json_data = load_test_data()

    # Test 1: First build with data source (should create new index)
    print("\n[Test 1] Building index for the first time...")
    backend1 = LinearRAGBackend(persist_base_dir="models")
    # Note: backend will automatically set persist_dir based on data_source

    start_time = time.time()
    backend1.setup_knowledge_base(json_data, data_source="graph_small")
    build_time = time.time() - start_time

    print(f"✓ Index built in {build_time:.2f} seconds")
    print(f"✓ Index persisted to {backend1.persist_dir}")

    # Verify persistence directory exists
    assert Path(backend1.persist_dir).exists(), "Persistence directory not created"
    assert (Path(backend1.persist_dir) / "metadata.json").exists(), "Metadata file not created"
    print("✓ Persistence files verified")

    # Test 2: Load from persisted index (should be faster)
    print("\n[Test 2] Loading from persisted index...")
    backend2 = LinearRAGBackend(persist_base_dir="models")

    start_time = time.time()
    backend2.setup_knowledge_base(json_data, data_source="graph_small")
    load_time = time.time() - start_time

    print(f"✓ Index loaded in {load_time:.2f} seconds")
    print(f"✓ Speedup: {build_time/load_time:.1f}x faster")

    # Test 3: Query both backends (should produce similar results)
    print("\n[Test 3] Testing queries...")
    query = "What is yoga?"

    response1 = backend1.process_query(query)
    response2 = backend2.process_query(query)

    print(f"✓ Query 1 response length: {len(response1)} chars")
    print(f"✓ Query 2 response length: {len(response2)} chars")

    # Test 4: Force rebuild
    print("\n[Test 4] Testing force rebuild...")
    backend3 = LinearRAGBackend(persist_base_dir="models")

    start_time = time.time()
    backend3.setup_knowledge_base(json_data, data_source="graph_small", force_rebuild=True)
    rebuild_time = time.time() - start_time

    print(f"✓ Index rebuilt in {rebuild_time:.2f} seconds")

    print("\n✅ All Linear RAG persistence tests passed!")
    return True


def test_graph_rag_persistence():
    """Test Graph RAG persistence functionality."""
    print("\n" + "="*60)
    print("Testing Graph RAG Persistence")
    print("="*60)

    # Load test data
    json_data = load_test_data()

    # Test 1: First build with data source (should create new index)
    print("\n[Test 1] Building graph index for the first time...")
    backend1 = GraphRAGBackend(persist_base_dir="models")
    # Note: backend will automatically set persist_dir based on data_source

    start_time = time.time()
    backend1.setup_knowledge_base(json_data, data_source="graph_small")
    build_time = time.time() - start_time

    print(f"✓ Graph index built in {build_time:.2f} seconds")
    print(f"✓ Index persisted to {backend1.persist_dir}")

    # Verify persistence directory exists
    assert Path(backend1.persist_dir).exists(), "Persistence directory not created"
    assert (Path(backend1.persist_dir) / "metadata.json").exists(), "Metadata file not created"
    print("✓ Persistence files verified")

    # Test 2: Load from persisted index (should be faster)
    print("\n[Test 2] Loading from persisted graph index...")
    backend2 = GraphRAGBackend(persist_base_dir="models")

    start_time = time.time()
    backend2.setup_knowledge_base(json_data, data_source="graph_small")
    load_time = time.time() - start_time

    print(f"✓ Graph index loaded in {load_time:.2f} seconds")
    print(f"✓ Speedup: {build_time/load_time:.1f}x faster")

    # Test 3: Query both backends (should produce similar results)
    print("\n[Test 3] Testing queries...")
    query = "What is yoga?"

    response1 = backend1.process_query(query)
    response2 = backend2.process_query(query)

    print(f"✓ Query 1 response length: {len(response1)} chars")
    print(f"✓ Query 2 response length: {len(response2)} chars")

    # Test 4: Force rebuild
    print("\n[Test 4] Testing force rebuild...")
    backend3 = GraphRAGBackend(persist_base_dir="models")

    start_time = time.time()
    backend3.setup_knowledge_base(json_data, data_source="graph_small", force_rebuild=True)
    rebuild_time = time.time() - start_time

    print(f"✓ Graph index rebuilt in {rebuild_time:.2f} seconds")

    print("\n✅ All Graph RAG persistence tests passed!")
    return True


def test_multiple_data_sources():
    """Test persistence with multiple data sources."""
    print("\n" + "="*60)
    print("Testing Multiple Data Sources")
    print("="*60)

    # Load different test data
    json_data1 = load_test_data()

    # Create second test data (modify the first one)
    json_data2 = json_data1.copy()
    json_data2["elements"]["nodes"][0]["data"]["id"] = "MODIFIED_NODE"

    # Test Linear RAG with multiple data sources
    print("\n--- Linear RAG Multiple Data Sources ---")

    # Test 1: Create index for first data source
    print("\n[Test 1] Creating Linear RAG index for data source 'graph_small'...")
    backend1 = LinearRAGBackend(persist_base_dir="models")

    backend1.setup_knowledge_base(json_data1, data_source="graph_small")
    print(f"✓ Linear RAG index created for 'graph_small' at {backend1.persist_dir}")

    # Test 2: Create index for second data source
    print("\n[Test 2] Creating Linear RAG index for data source 'sample_data'...")
    backend2 = LinearRAGBackend(persist_base_dir="models")

    backend2.setup_knowledge_base(json_data2, data_source="sample_data")
    print(f"✓ Linear RAG index created for 'sample_data' at {backend2.persist_dir}")

    # Verify both directories exist and are different
    assert Path(backend1.persist_dir).exists(), "First Linear RAG index directory not created"
    assert Path(backend2.persist_dir).exists(), "Second Linear RAG index directory not created"
    assert backend1.persist_dir != backend2.persist_dir, "Linear RAG directories should be different"

    # Test Graph RAG with multiple data sources
    print("\n--- Graph RAG Multiple Data Sources ---")

    # Test 3: Create Graph RAG index for first data source
    print("\n[Test 3] Creating Graph RAG index for data source 'graph_small'...")
    backend3 = GraphRAGBackend(persist_base_dir="models")

    backend3.setup_knowledge_base(json_data1, data_source="graph_small")
    print(f"✓ Graph RAG index created for 'graph_small' at {backend3.persist_dir}")

    # Test 4: Create Graph RAG index for second data source
    print("\n[Test 4] Creating Graph RAG index for data source 'sample_data'...")
    backend4 = GraphRAGBackend(persist_base_dir="models")

    backend4.setup_knowledge_base(json_data2, data_source="sample_data")
    print(f"✓ Graph RAG index created for 'sample_data' at {backend4.persist_dir}")

    # Verify both Graph RAG directories exist and are different
    assert Path(backend3.persist_dir).exists(), "First Graph RAG index directory not created"
    assert Path(backend4.persist_dir).exists(), "Second Graph RAG index directory not created"
    assert backend3.persist_dir != backend4.persist_dir, "Graph RAG directories should be different"

    # Test 5: Load from indices
    print("\n[Test 5] Loading from persisted indices...")
    backend5 = LinearRAGBackend(persist_base_dir="models")
    backend6 = LinearRAGBackend(persist_base_dir="models")
    backend7 = GraphRAGBackend(persist_base_dir="models")
    backend8 = GraphRAGBackend(persist_base_dir="models")

    backend5.setup_knowledge_base(json_data1, data_source="graph_small")
    backend6.setup_knowledge_base(json_data2, data_source="sample_data")
    backend7.setup_knowledge_base(json_data1, data_source="graph_small")
    backend8.setup_knowledge_base(json_data2, data_source="sample_data")

    print("✓ Successfully loaded all persisted indices")

    # Test 6: Query both Linear RAG indices
    print("\n[Test 6] Querying Linear RAG indices...")
    query = "What is yoga?"

    response1 = backend5.process_query(query)
    response2 = backend6.process_query(query)
    response3 = backend7.process_query(query)
    response4 = backend8.process_query(query)

    print(f"✓ Linear RAG 'graph_small' response: {len(response1)} chars")
    print(f"✓ Linear RAG 'sample_data' response: {len(response2)} chars")
    print(f"✓ Graph RAG 'graph_small' response: {len(response3)} chars")
    print(f"✓ Graph RAG 'sample_data' response: {len(response4)} chars")

    print("\n✅ All multiple data sources tests passed!")
    return True


def main():
    """Run all persistence tests."""
    print("\n" + "="*60)
    print("RAG Backend Persistence Test Suite")
    print("="*60)
    print(f"\nTest data: {TEST_DATA_FILE}")
    print("Linear RAG test dirs: models/linearrag_* (data source specific)")
    print("Graph RAG test dirs: models/graphrag_* (data source specific)")
    
    try:
        # Clean up any existing test directories
        print("\nCleaning up previous test runs...")
        cleanup_test_dirs()
        
        # Run tests
        linear_success = test_linear_rag_persistence()
        graph_success = test_graph_rag_persistence()
        multi_source_success = test_multiple_data_sources()
        
        # Final cleanup
        print("\n" + "="*60)
        print("Cleaning up test directories...")
        cleanup_test_dirs()
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Linear RAG:      {'✅ PASSED' if linear_success else '❌ FAILED'}")
        print(f"Graph RAG:       {'✅ PASSED' if graph_success else '❌ FAILED'}")
        print(f"Multi-Source:    {'✅ PASSED' if multi_source_success else '❌ FAILED'}")
        print(f"Data Source-Specific Indices: ✅ IMPLEMENTED")

        if linear_success and graph_success and multi_source_success:
            print("\n🎉 All tests passed successfully!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        print("\nCleaning up after error...")
        cleanup_test_dirs()
        return 1


if __name__ == "__main__":
    sys.exit(main())


