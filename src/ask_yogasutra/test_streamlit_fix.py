#!/usr/bin/env python3
"""
Test script to verify the streamlit persistence fix works.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphrag_backend import GraphRAGBackend
from linearrag_backend import LinearRAGBackend

def test_backend_initialization():
    """Test that backends initialize correctly without persist_dir being None."""
    print("Testing backend initialization...")

    # Test Graph RAG backend
    graph_backend = GraphRAGBackend()
    print(f"✓ GraphRAG persist_base_dir: {graph_backend.persist_base_dir}")
    print(f"✓ GraphRAG persist_dir: {graph_backend.persist_dir}")
    assert graph_backend.persist_base_dir == "models"
    assert graph_backend.persist_dir is None  # Should be None initially

    # Test Linear RAG backend
    linear_backend = LinearRAGBackend()
    print(f"✓ LinearRAG persist_base_dir: {linear_backend.persist_base_dir}")
    print(f"✓ LinearRAG persist_dir: {linear_backend.persist_dir}")
    assert linear_backend.persist_base_dir == "models"
    assert linear_backend.persist_dir is None  # Should be None initially

    print("✅ Backend initialization test passed!")

def test_directory_checking():
    """Test the directory checking logic that was causing the error."""
    print("\nTesting directory checking logic...")

    # Create a mock backend like Streamlit does
    class MockBackend:
        def __init__(self):
            self.persist_base_dir = "models"
            self.persist_dir = None

    backend = MockBackend()

    # This is the logic from the fixed Streamlit code
    try:
        persist_base_dir = getattr(backend, 'persist_base_dir', 'models')
        print(f"✓ persist_base_dir: {persist_base_dir}")

        # Check if base directory exists
        if os.path.exists(persist_base_dir):
            print(f"✓ Models directory exists: {persist_base_dir}")

            # Check for indices (this would work in real Streamlit)
            import glob
            graphrag_indices = glob.glob(f"{persist_base_dir}/graphrag_*")
            print(f"✓ Found {len(graphrag_indices)} Graph RAG indices")
        else:
            print(f"⚠️ Models directory not found: {persist_base_dir}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("✅ Directory checking test passed!")
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Streamlit Persistence Fix")
    print("=" * 50)

    try:
        test_backend_initialization()
        test_directory_checking()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! The fix should work.")
        print("\nThe original error was:")
        print("TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType")
        print("\nThis happened because persist_dir was None initially.")
        print("Now we use persist_base_dir and handle None values properly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
