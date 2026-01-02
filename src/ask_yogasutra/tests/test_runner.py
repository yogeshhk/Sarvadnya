"""
Unified test runner for all components.
"""
import sys
import unittest
import os

def run_all_tests():
    """Run all test suites."""
    # Check for GROQ_API_KEY
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set. RAG tests will be skipped.")
        print("Set GROQ_API_KEY in .env file to run all tests.")
        print()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Import test modules
    try:
        from graphrag.graph_builder import TestGraphBuilder
        from graphrag.graphrag_backend import TestGraphRAGBackend
        from linearrag.linearrag_backend import TestLinearRAGBackend
        
        suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphRAGBackend))
        suite.addTests(loader.loadTestsFromTestCase(TestLinearRAGBackend))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("Running all test suites for Ask Yogasutra")
    print("="*70)
    success = run_all_tests()
    print("="*70)
    if success:
        print("✓ All tests passed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)