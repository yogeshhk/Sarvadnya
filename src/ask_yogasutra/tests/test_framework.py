import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports for evaluation metrics
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    EVALUATION_DEPS_AVAILABLE = True
except ImportError:
    print("Warning: Evaluation dependencies not available (sentence-transformers, scikit-learn, numpy)")
    print("Basic functionality will work but evaluation metrics will be disabled")
    SentenceTransformer = None
    cosine_similarity = None
    np = None
    EVALUATION_DEPS_AVAILABLE = False

from graphrag.graphrag_backend import GraphRAGBackend, CONVERSATION_MODE
from linearrag.linearrag_backend import LinearRAGBackend
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import ChatMode

@dataclass
class TestConfig:
    """Configuration for a test run."""
    name: str
    description: str
    embedding_model: str
    groq_model: str
    temperature: float
    max_tokens: int
    chunk_size: int
    chunk_overlap: int
    max_triplets_per_chunk: int
    conversation_mode: bool
    chat_mode: Optional[str]
    response_mode: str
    graph_file: str
    backend_type: str = "graphrag"  # "graphrag" or "linearrag"
    use_cached_index: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestConfig':
        """Create TestConfig from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            embedding_model=data['embedding_model'],
            groq_model=data['groq_model'],
            temperature=data['temperature'],
            max_tokens=data['max_tokens'],
            chunk_size=data['chunk_size'],
            chunk_overlap=data['chunk_overlap'],
            max_triplets_per_chunk=data['max_triplets_per_chunk'],
            conversation_mode=data['conversation_mode'],
            chat_mode=data.get('chat_mode'),
            response_mode=data['response_mode'],
            graph_file=data['graph_file'],
            backend_type=data.get('backend_type', 'graphrag'),
            use_cached_index=data.get('use_cached_index', True)
        )

@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    question: str
    response: str
    response_time: float
    keyword_score: float
    sutra_reference_score: float
    semantic_score: float
    overall_score: float
    passed: bool
    actual_sutra_references: List[str]
    error: Optional[str] = None

@dataclass
class TestSummary:
    """Summary of test results for a configuration."""
    config_name: str
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_keyword_score: float
    avg_sutra_reference_score: float
    avg_semantic_score: float
    avg_response_time: float
    category_breakdown: Dict[str, Dict[str, Any]]
    detailed_results: List[TestResult]

class GraphRAGTester:
    """Main testing framework for GraphRAG benchmark evaluation."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.backend = None
        self.semantic_model = None
        self.test_data = None
        self.evaluation_available = EVALUATION_DEPS_AVAILABLE

    def initialize_backend(self) -> bool:
        """Initialize the appropriate backend based on configuration."""
        try:
            # Load graph data
            with open(self.config.graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # Initialize semantic model for evaluation (if available)
            if self.evaluation_available:
                try:
                    self.semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                except Exception as e:
                    print(f"Warning: Could not initialize semantic model: {str(e)}")
                    self.evaluation_available = False
            else:
                print("Semantic evaluation disabled (dependencies not available)")

            # Initialize appropriate backend
            if self.config.backend_type.lower() == "linearrag":
                self.backend = LinearRAGBackend()
            else:
                # Pass chat_mode to GraphRAGBackend if conversation mode is enabled
                chat_mode = self.config.chat_mode if self.config.conversation_mode else "condense_plus_context"
                self.backend = GraphRAGBackend(chat_mode=chat_mode)

            # Temporarily override backend settings
            self._override_backend_settings()

            # Setup knowledge base with cached index option
            force_rebuild = not self.config.use_cached_index
            success = self.backend.setup_knowledge_base(
                graph_data,
                force_rebuild=force_rebuild
            )
            return success

        except Exception as e:
            print(f"Error initializing backend: {str(e)}")
            return False

    def _override_backend_settings(self):
        """Override backend settings based on test configuration."""
        try:
            # Override embedding model
            if hasattr(self.backend, 'EMBEDDING_MODEL_NAME'):
                self.backend.__class__.EMBEDDING_MODEL_NAME = self.config.embedding_model

            # Override LLM model and settings
            if hasattr(self.backend, 'GROQ_MODEL_NAME'):
                self.backend.__class__.GROQ_MODEL_NAME = self.config.groq_model

            # For GraphRAG backend, we need to handle conversation mode
            if isinstance(self.backend, GraphRAGBackend):
                global CONVERSATION_MODE
                CONVERSATION_MODE = self.config.conversation_mode

        except Exception as e:
            print(f"Warning: Could not override backend settings: {str(e)}")

    def load_test_dataset(self, dataset_file: str) -> bool:
        """Load test dataset from JSON file."""
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_data = data.get('test_cases', [])
                return True
        except Exception as e:
            print(f"Error loading test dataset: {str(e)}")
            return False

    def _extract_sutra_references(self, response: str) -> List[str]:
        """Extract sutra references from response."""
        references = []

        # Look for patterns like "I.1", "I.2", "II.1", etc.
        sutra_pattern = r'\b([IVXLCD]+\.\d+)\b'
        matches = re.findall(sutra_pattern, response, re.IGNORECASE)
        references.extend(matches)

        # Also look for "sutra I.1", "Sutra I.2", etc.
        sutra_word_pattern = r'\bsutra\s+([IVXLCD]+\.\d+)\b'
        matches = re.findall(sutra_word_pattern, response, re.IGNORECASE)
        references.extend(matches)

        # Remove duplicates and normalize
        references = list(set(ref.upper() for ref in references))
        return references

    def _calculate_keyword_score(self, response: str, expected_keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        if not expected_keywords:
            return 1.0

        response_lower = response.lower()
        matched_keywords = 0

        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                matched_keywords += 1

        return matched_keywords / len(expected_keywords)

    def _calculate_sutra_reference_score(self, actual_refs: List[str], expected_refs: List[str]) -> float:
        """Calculate F1 score for sutra references."""
        if not expected_refs and not actual_refs:
            return 1.0

        if not expected_refs or not actual_refs:
            return 0.0

        # Convert to sets for comparison
        expected_set = set(ref.upper() for ref in expected_refs)
        actual_set = set(ref.upper() for ref in actual_refs)

        # Calculate precision and recall
        true_positives = len(expected_set.intersection(actual_set))
        precision = true_positives / len(actual_set) if actual_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def _calculate_semantic_score(self, response: str, expected_keywords: List[str]) -> float:
        """Calculate semantic similarity score."""
        if not expected_keywords:
            return 1.0

        if not self.evaluation_available or self.semantic_model is None:
            # Fallback: simple keyword overlap as percentage of response words
            response_words = set(response.lower().split())
            keyword_words = set(" ".join(expected_keywords).lower().split())
            overlap = len(response_words.intersection(keyword_words))
            total_keywords = len(keyword_words)
            return min(1.0, overlap / max(1, total_keywords)) if total_keywords > 0 else 0.0

        try:
            # Create keyword string
            keyword_text = " ".join(expected_keywords)

            # Encode both texts
            response_embedding = self.semantic_model.encode([response])
            keyword_embedding = self.semantic_model.encode([keyword_text])

            # Calculate cosine similarity
            similarity = cosine_similarity(response_embedding, keyword_embedding)[0][0]

            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            print(f"Error calculating semantic score: {str(e)}")
            return 0.0

    def _prepare_conversation_context(self, context: List[Dict[str, str]]) -> List[ChatMessage]:
        """Convert conversation context to ChatMessage objects."""
        messages = []
        for msg in context:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg["content"]))
        return messages

    def run_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """Run a single test case."""
        test_id = test_case['id']
        question = test_case['question']
        expected_keywords = test_case.get('expected_answer_keywords', [])
        expected_sutra_refs = test_case.get('expected_sutra_references', [])
        conversation_context = test_case.get('conversation_context', [])

        start_time = time.time()

        try:
            # Prepare conversation context if it exists
            if conversation_context and self.config.conversation_mode:
                messages = self._prepare_conversation_context(conversation_context)
                response = self.backend.process_conversation(question, messages)
            else:
                response = self.backend.process_query(question)

            response_time = time.time() - start_time

            # Extract actual sutra references
            actual_sutra_refs = self._extract_sutra_references(response)

            # Calculate scores
            keyword_score = self._calculate_keyword_score(response, expected_keywords)
            sutra_score = self._calculate_sutra_reference_score(actual_sutra_refs, expected_sutra_refs)
            semantic_score = self._calculate_semantic_score(response, expected_keywords)

            # Calculate overall score (weighted average)
            if conversation_context and self.config.conversation_mode:
                overall_score = (keyword_score + semantic_score) / 2.0
            else:
                overall_score = (keyword_score + sutra_score + semantic_score) / 3.0

            # Determine if test passed (overall score >= 0.6)
            passed = bool(overall_score >= 0.6)

            return TestResult(
                test_id=test_id,
                question=question,
                response=response,
                response_time=response_time,
                keyword_score=keyword_score,
                sutra_reference_score=sutra_score,
                semantic_score=semantic_score,
                overall_score=overall_score,
                passed=passed,
                actual_sutra_references=actual_sutra_refs
            )

        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                test_id=test_id,
                question=question,
                response="",
                response_time=response_time,
                keyword_score=0.0,
                sutra_reference_score=0.0,
                semantic_score=0.0,
                overall_score=0.0,
                passed=False,
                actual_sutra_references=[],
                error=str(e)
            )

    def run_benchmark(self, dataset_file: str) -> TestSummary:
        """Run complete benchmark test suite."""
        if not self.load_test_dataset(dataset_file):
            raise Exception("Failed to load test dataset")

        if not self.initialize_backend():
            raise Exception("Failed to initialize backend")

        print(f"Running benchmark with configuration: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Backend: {self.config.backend_type}")
        print(f"Number of test cases: {len(self.test_data)}")
        print("-" * 80)

        results = []
        category_stats = {}

        for i, test_case in enumerate(self.test_data):
            print(f"Running test {i+1}/{len(self.test_data)}: {test_case['id']}")

            result = self.run_single_test(test_case)
            results.append(result)

            # Update category statistics
            category = test_case.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'total': 0,
                    'passed': 0,
                    'keyword_scores': [],
                    'sutra_scores': [],
                    'semantic_scores': [],
                    'response_times': []
                }

            category_stats[category]['total'] += 1
            if result.passed:
                category_stats[category]['passed'] += 1

            category_stats[category]['keyword_scores'].append(result.keyword_score)
            category_stats[category]['sutra_scores'].append(result.sutra_reference_score)
            category_stats[category]['semantic_scores'].append(result.semantic_score)
            category_stats[category]['response_times'].append(result.response_time)

        # Calculate summary statistics
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests

        avg_keyword_score = float(np.mean([r.keyword_score for r in results]))
        avg_sutra_score = float(np.mean([r.sutra_reference_score for r in results]))
        avg_semantic_score = float(np.mean([r.semantic_score for r in results]))
        avg_response_time = float(np.mean([r.response_time for r in results]))

        # Process category breakdown
        category_breakdown = {}
        for category, stats in category_stats.items():
            category_breakdown[category] = {
                'total': stats['total'],
                'passed': stats['passed'],
                'pass_rate': float(stats['passed'] / stats['total']) if stats['total'] > 0 else 0.0,
                'avg_keyword': float(np.mean(stats['keyword_scores'])),
                'avg_sutra': float(np.mean(stats['sutra_scores'])),
                'avg_semantic': float(np.mean(stats['semantic_scores'])),
                'avg_time': float(np.mean(stats['response_times']))
            }

        from datetime import datetime
        timestamp = datetime.now().isoformat()

        summary = TestSummary(
            config_name=self.config.name,
            timestamp=timestamp,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            avg_keyword_score=avg_keyword_score,
            avg_sutra_reference_score=avg_sutra_score,
            avg_semantic_score=avg_semantic_score,
            avg_response_time=avg_response_time,
            category_breakdown=category_breakdown,
            detailed_results=results
        )

        return summary

def load_test_configs(config_file: str) -> Dict[str, TestConfig]:
    """Load test configurations from JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        configs = {}
        for config_name, config_data in data.get('configurations', {}).items():
            configs[config_name] = TestConfig.from_dict(config_data)

        return configs
    except Exception as e:
        raise Exception(f"Error loading test configurations: {str(e)}")

def print_results_summary(summary: TestSummary):
    """Print formatted test results summary."""
    print("\n" + "=" * 80)
    print(f"TEST RUN SUMMARY: {summary.config_name}")
    print("=" * 80)
    print(f"Timestamp: {summary.timestamp}")
    print()
    print("Overall Results:")
    print(f"  Total Tests: {summary.total_tests}")
    print(f"  Passed: {summary.passed_tests} ({summary.passed_tests/summary.total_tests*100:.1f}%)")
    print(f"  Failed: {summary.failed_tests} ({summary.failed_tests/summary.total_tests*100:.1f}%)")
    print()
    print("Average Scores:")
    print(f"  Keyword Match: {summary.avg_keyword_score:.3f}")
    print(f"  Sutra Reference: {summary.avg_sutra_reference_score:.3f}")
    print(f"  Semantic Similarity: {summary.avg_semantic_score:.3f}")
    print(f"  Response Time: {summary.avg_response_time:.3f}s")
    print()
    print("Category Breakdown:")
    for category, stats in summary.category_breakdown.items():
        print(f"  {category.upper()}:")
        print(f"    Total: {stats['total']}, Passed: {stats['passed']} ({stats['pass_rate']*100:.1f}%)")
        print(f"    Avg Keyword: {stats['avg_keyword']:.3f}, Avg Sutra: {stats['avg_sutra']:.3f}")
        print(f"    Avg Semantic: {stats['avg_semantic']:.3f}, Avg Time: {stats['avg_time']:.3f}s")
    print("=" * 80)

def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar types
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_results_to_file(summary: TestSummary, output_dir: str = "test_results"):
    """Save test results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = summary.timestamp.replace(':', '-').replace('.', '-')
    filename = f"results_{summary.config_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert results to serializable format with numpy type conversion
    results_data = {
        "config_name": summary.config_name,
        "timestamp": summary.timestamp,
        "summary": {
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "failed_tests": summary.failed_tests,
            "avg_keyword_score": _convert_numpy_types(summary.avg_keyword_score),
            "avg_sutra_reference_score": _convert_numpy_types(summary.avg_sutra_reference_score),
            "avg_semantic_score": _convert_numpy_types(summary.avg_semantic_score),
            "avg_response_time": _convert_numpy_types(summary.avg_response_time),
            "category_breakdown": _convert_numpy_types(summary.category_breakdown)
        },
        "detailed_results": [
            {
                "test_id": r.test_id,
                "question": r.question,
                "response": r.response,
                "response_time": _convert_numpy_types(r.response_time),
                "keyword_score": _convert_numpy_types(r.keyword_score),
                "sutra_reference_score": _convert_numpy_types(r.sutra_reference_score),
                "semantic_score": _convert_numpy_types(r.semantic_score),
                "overall_score": _convert_numpy_types(r.overall_score),
                "passed": _convert_numpy_types(r.passed),
                "actual_sutra_references": r.actual_sutra_references,
                "error": r.error
            }
            for r in summary.detailed_results
        ]
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath

def generate_comparison_report(results_dir: str = "test_results", output_file: str = None) -> str:
    """Generate comparison report across all test configurations."""
    if not os.path.exists(results_dir):
        raise Exception(f"Results directory not found: {results_dir}")

    # Load all result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]

    if not result_files:
        raise Exception(f"No result files found in {results_dir}")

    all_results = []
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {str(e)}")

    if not all_results:
        raise Exception("No valid result files found")

    # Generate comparison data
    comparison = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_configurations": len(all_results),
        "configurations": []
    }

    # Sort by overall performance
    all_results.sort(key=lambda x: x["summary"]["avg_keyword_score"] +
                                   x["summary"]["avg_sutra_reference_score"] +
                                   x["summary"]["avg_semantic_score"], reverse=True)

    for result in all_results:
        config_data = {
            "config_name": result["config_name"],
            "total_tests": result["summary"]["total_tests"],
            "passed_tests": result["summary"]["passed_tests"],
            "pass_rate": _convert_numpy_types(result["summary"]["passed_tests"] / result["summary"]["total_tests"]),
            "avg_keyword_score": _convert_numpy_types(result["summary"]["avg_keyword_score"]),
            "avg_sutra_reference_score": _convert_numpy_types(result["summary"]["avg_sutra_reference_score"]),
            "avg_semantic_score": _convert_numpy_types(result["summary"]["avg_semantic_score"]),
            "avg_response_time": _convert_numpy_types(result["summary"]["avg_response_time"]),
            "overall_score": _convert_numpy_types((result["summary"]["avg_keyword_score"] +
                            result["summary"]["avg_sutra_reference_score"] +
                            result["summary"]["avg_semantic_score"]) / 3.0),
            "category_breakdown": _convert_numpy_types(result["summary"]["category_breakdown"])
        }
        comparison["configurations"].append(config_data)

    # Determine best configuration per category
    category_best = {}
    for result in all_results:
        for category, stats in result["summary"]["category_breakdown"].items():
            current_score = _convert_numpy_types(stats["avg_keyword"] + stats["avg_sutra"] + stats["avg_semantic"])
            if category not in category_best:
                category_best[category] = {
                    "config": result["config_name"],
                    "score": current_score
                }
            else:
                if current_score > category_best[category]["score"]:
                    category_best[category] = {
                        "config": result["config_name"],
                        "score": current_score
                    }

    comparison["category_best_configs"] = category_best
    comparison["best_overall_config"] = comparison["configurations"][0]["config_name"]

    # Save comparison report
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"comparison_report_{timestamp}.json"

    output_path = os.path.join(results_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison report saved to: {output_path}")
    return output_path
