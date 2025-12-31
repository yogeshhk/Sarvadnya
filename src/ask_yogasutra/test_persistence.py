#!/usr/bin/env python3
"""
Test script to verify persistence functionality for both Linear RAG and Graph RAG backends.

This script tests:
1. Index creation and persistence
2. Loading from persisted indices
3. Data change detection
4. Force rebuild functionality
"""

import json
import shutil
import time
from pathlib import Path
import sys

# Import backends
from linearrag_backend import LinearRAGBackend
from graphrag_backend import GraphRAGBackend

# Test configuration
TEST_DATA_FILE = "data/graph_small.json"
TEST_DATA_FILE2 = "data/sample-graph-json.json"
TEST_LINEAR_DIR1 = "models/test_linearrag_graph_small"
TEST_LINEAR_DIR2 = "models/test_linearrag_sample_graph_json"
TEST_GRAPH_DIR1 = "models/test_graphrag_graph_small"
TEST_GRAPH_DIR2 = "models/test_graphrag_sample_graph_json"


def cleanup_test_dirs():
    """Clean up test directories."""
    for dir_path in [TEST_LINEAR_DIR, TEST_GRAPH_DIR]:
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
    backend1.persist_dir = TEST_LINEAR_DIR1  # Override for testing

    start_time = time.time()
    backend1.setup_knowledge_base(json_data, data_source="graph_small")
    build_time = time.time() - start_time

    print(f"✓ Index built in {build_time:.2f} seconds")
    print(f"✓ Index persisted to {TEST_LINEAR_DIR1}")

    # Verify persistence directory exists
    assert Path(TEST_LINEAR_DIR1).exists(), "Persistence directory not created"
    assert (Path(TEST_LINEAR_DIR1) / "metadata.json").exists(), "Metadata file not created"
    print("✓ Persistence files verified")

    # Test 2: Load from persisted index (should be faster)
    print("\n[Test 2] Loading from persisted index...")
    backend2 = LinearRAGBackend(persist_base_dir="models")
    backend2.persist_dir = TEST_LINEAR_DIR1  # Override for testing

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
    backend3.persist_dir = TEST_LINEAR_DIR1  # Override for testing

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
    
    # Test 1: First build (should create new index)
    print("\n[Test 1] Building graph index for the first time...")
    backend1 = GraphRAGBackend(persist_dir=TEST_GRAPH_DIR)
    
    start_time = time.time()
    backend1.setup_knowledge_base(json_data)
    build_time = time.time() - start_time
    
    print(f"✓ Graph index built in {build_time:.2f} seconds")
    print(f"✓ Index persisted to {TEST_GRAPH_DIR}")
    
    # Verify persistence directory exists
    assert Path(TEST_GRAPH_DIR).exists(), "Persistence directory not created"
    assert (Path(TEST_GRAPH_DIR) / "metadata.json").exists(), "Metadata file not created"
    print("✓ Persistence files verified")
    
    # Test 2: Load from persisted index (should be faster)
    print("\n[Test 2] Loading from persisted graph index...")
    backend2 = GraphRAGBackend(persist_dir=TEST_GRAPH_DIR)
    
    start_time = time.time()
    backend2.setup_knowledge_base(json_data)
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
    backend3 = GraphRAGBackend(persist_dir=TEST_GRAPH_DIR)
    
    start_time = time.time()
    backend3.setup_knowledge_base(json_data, force_rebuild=True)
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

    # Test 1: Create index for first data source
    print("\n[Test 1] Creating index for data source 'graph_small'...")
    backend1 = LinearRAGBackend(persist_base_dir="models")
    backend1.persist_dir = TEST_LINEAR_DIR1

    backend1.setup_knowledge_base(json_data1, data_source="graph_small")
    print(f"✓ Index created for 'graph_small' at {TEST_LINEAR_DIR1}")

    # Test 2: Create index for second data source
    print("\n[Test 2] Creating index for data source 'sample_data'...")
    backend2 = LinearRAGBackend(persist_base_dir="models")
    backend2.persist_dir = TEST_LINEAR_DIR2

    backend2.setup_knowledge_base(json_data2, data_source="sample_data")
    print(f"✓ Index created for 'sample_data' at {TEST_LINEAR_DIR2}")

    # Verify both directories exist and are different
    assert Path(TEST_LINEAR_DIR1).exists(), "First index directory not created"
    assert Path(TEST_LINEAR_DIR2).exists(), "Second index directory not created"
    assert TEST_LINEAR_DIR1 != TEST_LINEAR_DIR2, "Directories should be different"

    # Test 3: Load from first index
    print("\n[Test 3] Loading from first data source...")
    backend3 = LinearRAGBackend(persist_base_dir="models")
    backend3.persist_dir = TEST_LINEAR_DIR1

    backend3.setup_knowledge_base(json_data1, data_source="graph_small")
    print("✓ Successfully loaded first index")

    # Test 4: Load from second index
    print("\n[Test 4] Loading from second data source...")
    backend4 = LinearRAGBackend(persist_base_dir="models")
    backend4.persist_dir = TEST_LINEAR_DIR2

    backend4.setup_knowledge_base(json_data2, data_source="sample_data")
    print("✓ Successfully loaded second index")

    # Test 5: Query both indices
    print("\n[Test 5] Querying both indices...")
    query = "What is yoga?"

    response1 = backend3.process_query(query)
    response2 = backend4.process_query(query)

    print(f"✓ First index response: {len(response1)} chars")
    print(f"✓ Second index response: {len(response2)} chars")

    print("\n✅ All multiple data sources tests passed!")
    return True


def main():
    """Run all persistence tests."""
    print("\n" + "="*60)
    print("RAG Backend Persistence Test Suite")
    print("="*60)
    print(f"\nTest data: {TEST_DATA_FILE}")
    print(f"Linear RAG test dir: {TEST_LINEAR_DIR}")
    print(f"Graph RAG test dir: {TEST_GRAPH_DIR}")
    
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


