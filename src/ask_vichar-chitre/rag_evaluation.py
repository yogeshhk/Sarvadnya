import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import re

# NLP evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
try:
    from nltk.translate.meteor_score import meteor_score
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("⚠️ METEOR score might not work properly. Install nltk properly.")

# Embedding similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM for judge evaluation
from groq import Groq

# RAG Chatbot to evaluate
from rag_gemma import RAGChatbot


class RAGEvaluator:
    """
    Comprehensive RAG evaluation system using multiple metrics:
    - BLEU, ROUGE, METEOR scores
    - Embedding-based semantic similarity (DISABLED)
    - LLM as a judge
    - Context location matching
    """
    
    def __init__(self, 
                 csv_file: str = "evaluation_set.csv",
                 llm_model: str = "gemma2-9b-it",
                 embedding_model: str = None, # Disabled
                 groq_api_key: str = None):
        """
        Initialize the RAG evaluator
        
        Args:
            csv_file: Path to evaluation dataset CSV
            llm_model: LLM model name for Groq
            embedding_model: Sentence transformer model for embeddings (DISABLED)
            groq_api_key: Groq API key
        """
        self.csv_file = csv_file
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model 
        
        # Initialize Groq client
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("❌ GROQ_API_KEY missing. Set it in environment or pass as parameter")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Load evaluation dataset
        self.evaluation_data = self._load_evaluation_data()
        
        # Initialize embedding model
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Smoothing function for BLEU
        self.bleu_smoothing = SmoothingFunction().method4
        
        print(f"✅ RAG Evaluator initialized with {len(self.evaluation_data)} evaluation samples")
    
    def _load_evaluation_data(self) -> pd.DataFrame:
        """Load and parse the evaluation CSV file"""
        try:
            # Read CSV with proper encoding for Devanagari text
            df = pd.read_csv(self.csv_file, encoding='utf-8', sep='\t')
            
            # Rename columns for easier access
            if len(df.columns) >= 3:
                df.columns = ['question', 'correct_answer', 'source_location']
            
            print(f"📊 Loaded {len(df)} evaluation samples from {self.csv_file}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading evaluation data: {e}")
            raise
    
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score between predicted and reference text"""
        try:
            # Tokenize sentences
            reference_tokens = [reference.split()]
            predicted_tokens = predicted.split()
            
            # Calculate BLEU score
            score = sentence_bleu(reference_tokens, predicted_tokens, 
                                smoothing_function=self.bleu_smoothing)
            return score
        except:
            return 0.0
    
    def calculate_rouge_scores(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_meteor_score(self, predicted: str, reference: str) -> float:
        """Calculate METEOR score"""
        try:
            # Tokenize
            predicted_tokens = predicted.split()
            reference_tokens = reference.split()
            
            score = meteor_score([reference_tokens], predicted_tokens)
            return score
        except:
            return 0.0
    
    def calculate_semantic_similarity(self, predicted: str, reference: str) -> float:
        """Calculate semantic similarity using sentence embeddings - DISABLED"""
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode([predicted, reference])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except:
            return 0.0
    
    def llm_as_judge(self, question: str, predicted: str, reference: str) -> Dict[str, Any]:
        """Use LLM to evaluate answer quality"""
        
        judge_prompt = f"""You are expert in Mental Models and Marathi Language. You need to evaluate a RAG system's answer quality. Predict the answer only  in one line. Here is the information:

Question: {question}

Reference Answer (Ground Truth): {reference}

Predicted Answer: {predicted}

Please evaluate the predicted answer based on the following criteria:

1. **Accuracy**: Is the answer factually correct? (0-10)
2. **Completeness**: Does the answer provide complete information? (0-10)  
3. **Relevance**: Is the answer relevant to the question? (0-10)
4. **Clarity**: Is the answer clear and understandable? (0-10)

Please provide a score (0-10) and a brief comment for each criterion.
Also provide an overall score (0-10) at the end.

Respond in JSON format:
{{
    "accuracy": {{"score": X, "comment": "comment"}},
    "completeness": {{"score": X, "comment": "comment"}},
    "relevance": {{"score": X, "comment": "comment"}},
    "clarity": {{"score": X, "comment": "comment"}},
    "overall_score": X,
    "overall_comment": "overall comment"
}}"""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert evaluator who assesses RAG system answers for quality and accuracy."},
                    {"role": "user", "content": judge_prompt}
                ],
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                llm_scores = json.loads(json_str)
                return llm_scores
            else:
                return {"error": "Could not parse LLM response"}
                
        except Exception as e:
            print(f"⚠️ LLM judge evaluation failed: {e}")
            return {"error": str(e)}
    
    def calculate_context_location_match(self, predicted_context: str, correct_location: str) -> float:
        """Calculate how well the retrieved context matches the expected source location"""
        try:
            if not predicted_context or not correct_location:
                return 0.0
            
            # Extract filename from correct location
            correct_file = correct_location.split(':')[0] if ':' in correct_location else correct_location
            
            # Check if the predicted context mentions or comes from the same file
            # This is a simple text matching approach
            if correct_file in predicted_context:
                return 1.0
            
            # For more sophisticated matching, you could use fuzzy string matching
            # or check semantic similarity between context and expected content
            return 0.0
            
        except:
            return 0.0
    
    def evaluate_single_sample(self, question: str, predicted_answer: str, 
                             predicted_context: str, correct_answer: str, 
                             correct_location: str) -> Dict[str, Any]:
        """Evaluate a single prediction against ground truth"""
        
        # Calculate traditional NLP metrics
        bleu_score = self.calculate_bleu_score(predicted_answer, correct_answer)
        rouge_scores = self.calculate_rouge_scores(predicted_answer, correct_answer)
        meteor_score = self.calculate_meteor_score(predicted_answer, correct_answer)
        
        # Calculate semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(predicted_answer, correct_answer)
        
        # LLM as judge evaluation
        llm_evaluation = self.llm_as_judge(question, predicted_answer, correct_answer)
        
        # Context location matching
        context_match = self.calculate_context_location_match(predicted_context, correct_location)
        
        return {
            'question': question,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'predicted_context': predicted_context[:200] + "..." if len(predicted_context) > 200 else predicted_context,
            'correct_location': correct_location,
            'metrics': {
                'bleu_score': bleu_score,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'meteor_score': meteor_score,
                'semantic_similarity': semantic_similarity, # Will be 0
                'context_location_match': context_match
            },
            'llm_judge': llm_evaluation
        }
    
    def evaluate_rag_system(self, rag_chatbot, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the entire RAG system on the evaluation dataset
        
        Args:
            rag_chatbot: Instance of RAGChatbot to evaluate
            save_results: Whether to save detailed results to file
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        print(f"🔄 Starting evaluation of RAG system on {len(self.evaluation_data)} samples...")
        
        results = []
        all_metrics = {
            'bleu_scores': [],
            'rouge1_scores': [],
            'rouge2_scores': [],
            'rougeL_scores': [],
            'meteor_scores': [],
            'semantic_similarities': [],
            'context_matches': [],
            'llm_overall_scores': []
        }
        
        for idx, row in self.evaluation_data.iterrows():
            question = row['question']
            correct_answer = row['correct_answer']
            correct_location = row['source_location']
            
            print(f"📝 Evaluating sample {idx + 1}/{len(self.evaluation_data)}: {question[:50]}...")
            
            try:
                # Get prediction from RAG system
                rag_response = rag_chatbot.get_response(question)
                predicted_answer = rag_response["answer"]
                predicted_context = rag_response["context"]
                
                # Evaluate this sample
                sample_result = self.evaluate_single_sample(
                    question, predicted_answer, predicted_context, 
                    correct_answer, correct_location
                )
                
                results.append(sample_result)
                
                # Collect metrics
                metrics = sample_result['metrics']
                all_metrics['bleu_scores'].append(metrics['bleu_score'])
                all_metrics['rouge1_scores'].append(metrics['rouge1'])
                all_metrics['rouge2_scores'].append(metrics['rouge2'])
                all_metrics['rougeL_scores'].append(metrics['rougeL'])
                all_metrics['meteor_scores'].append(metrics['meteor_score'])
                all_metrics['semantic_similarities'].append(metrics['semantic_similarity'])
                all_metrics['context_matches'].append(metrics['context_location_match'])
                
                # LLM judge score
                llm_eval = sample_result['llm_judge']
                if 'overall_score' in llm_eval and isinstance(llm_eval['overall_score'], (int, float)):
                    all_metrics['llm_overall_scores'].append(llm_eval['overall_score'] / 10.0)  # Normalize to 0-1
                
            except Exception as e:
                print(f"❌ Error evaluating sample {idx + 1}: {e}")
                continue
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'avg_bleu_score': np.mean(all_metrics['bleu_scores']),
            'avg_rouge1_score': np.mean(all_metrics['rouge1_scores']),
            'avg_rouge2_score': np.mean(all_metrics['rouge2_scores']),
            'avg_rougeL_score': np.mean(all_metrics['rougeL_scores']),
            'avg_meteor_score': np.mean(all_metrics['meteor_scores']),
            'avg_semantic_similarity': np.mean(all_metrics['semantic_similarities']),
            'context_match_accuracy': np.mean(all_metrics['context_matches']),
            'avg_llm_judge_score': np.mean(all_metrics['llm_overall_scores']) if all_metrics['llm_overall_scores'] else 0.0
        }
        
        # Calculate composite score (weighted average of all metrics)
        composite_score = (
            aggregate_metrics['avg_bleu_score'] * 0.15 +
            aggregate_metrics['avg_rouge1_score'] * 0.15 +
            aggregate_metrics['avg_rougeL_score'] * 0.15 +
            aggregate_metrics['avg_semantic_similarity'] * 0.25 +
            aggregate_metrics['context_match_accuracy'] * 0.15 +
            aggregate_metrics['avg_llm_judge_score'] * 0.15
        )
        
        evaluation_summary = {
            'total_samples': len(results),
            'successful_evaluations': len([r for r in results if 'metrics' in r]),
            'aggregate_metrics': aggregate_metrics,
            'composite_score': "N/A - Semantic Similarity disabled",
            'detailed_results': results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results to file
        if save_results:
            results_filename = f"rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
            print(f"💾 Detailed results saved to: {results_filename}")
        
        # Print summary
        self._print_evaluation_summary(evaluation_summary)
        
        return evaluation_summary
    
    def _print_evaluation_summary(self, evaluation_summary: Dict[str, Any]):
        """Print a formatted evaluation summary"""
        print("\n" + "="*60)
        print("🎯 RAG SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        metrics = evaluation_summary['aggregate_metrics']
        
        print(f"📊 Total Samples: {evaluation_summary['total_samples']}")
        print(f"✅ Successful Evaluations: {evaluation_summary['successful_evaluations']}")
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   • BLEU Score:           {metrics['avg_bleu_score']:.4f}")
        print(f"   • ROUGE-1 Score:        {metrics['avg_rouge1_score']:.4f}")
        print(f"   • ROUGE-2 Score:        {metrics['avg_rouge2_score']:.4f}")
        print(f"   • ROUGE-L Score:        {metrics['avg_rougeL_score']:.4f}")
        print(f"   • METEOR Score:         {metrics['avg_meteor_score']:.4f}")
        print(f"   • Semantic Similarity:  {metrics['avg_semantic_similarity']:.4f} (Disabled)")
        print(f"   • Context Match Accuracy: {metrics['context_match_accuracy']:.4f}")
        print(f"   • LLM Judge Score:      {metrics['avg_llm_judge_score']:.4f}")
        
        # Composite score and interpretation are disabled
        print(f"\n🏆 COMPOSITE SCORE: {evaluation_summary['composite_score']}")
        
        # Performance interpretation
        # composite = evaluation_summary['composite_score']
        # if composite >= 0.8:
        #     performance = "🌟 Excellent"
        # elif composite >= 0.6:
        #     performance = "👍 Good"
        # elif composite >= 0.4:
        #     performance = "⚠️ Average"
        # else:
        #     performance = "❌ Needs Improvement"
        
        # print(f"📋 Overall Performance: {performance}")
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("🚀 RAG Evaluation System Test")
        
    # Initialize evaluator
    try:
        evaluator = RAGEvaluator(
            csv_file="evaluation_set.csv",
            llm_model="gemma2-9b-it"
        )
        
        print("✅ RAG Evaluator initialized successfully!")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        bot = RAGChatbot(data_directory="data", groq_api_key=groq_api_key)
        results = evaluator.evaluate_rag_system(bot)
        
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
