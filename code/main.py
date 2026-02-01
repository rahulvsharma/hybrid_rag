#!/usr/bin/env python3
"""
Main execution script for Hybrid RAG System
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for evaluation module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_collection import DataPreprocessor, create_fixed_urls_file
from rag_system import HybridRAGSystem
from evaluation.evaluation import QuestionGenerator, EvaluationPipeline
from evaluation.evaluation_pipeline import AutomatedEvaluationPipeline


def setup_data(fixed_count=200, random_count=300):
    """Setup data collection and preprocessing"""
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION & PREPROCESSING")
    print("="*70)
    
    # Check if fixed URLs exist
    fixed_urls_file = Path('fixed_urls.json')
    if fixed_urls_file.exists():
        print(f"  Loading existing fixed URLs from {fixed_urls_file}")
        with open(fixed_urls_file, 'r') as f:
            urls_data = json.load(f)
        fixed_urls = urls_data['urls']
    else:
        print(f"  Creating new fixed URL set ({fixed_count} URLs)...")
        fixed_urls = create_fixed_urls_file('fixed_urls.json', fixed_count)
    
    # Check if corpus exists
    corpus_file = Path('wikipedia_corpus.json')
    if corpus_file.exists():
        print(f"  Loading existing corpus from {corpus_file}")
    else:
        print(f"  Collecting Wikipedia corpus...")
        print(f"  - Fixed URLs: {len(fixed_urls)}")
        print(f"  - Random URLs: {random_count}")
        
        preprocessor = DataPreprocessor()
        corpus = preprocessor.collect_wikipedia_corpus(fixed_urls, random_count)
        preprocessor.save_corpus(corpus, 'wikipedia_corpus.json')
        
        print(f"  Corpus created:")
        print(f"  - Total chunks: {corpus['total_chunks']}")
        print(f"  - Total unique URLs: {corpus['total_urls']}")
    
    return fixed_urls


def test_rag_system():
    """Test RAG system with sample queries"""
    print("\n" + "="*70)
    print("STEP 2: RAG SYSTEM TEST")
    print("="*70)
    
    print("  Initializing RAG system...")
    rag_system = HybridRAGSystem('wikipedia_corpus.json')
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    print("  Running sample queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")
        result = rag_system.query(query, top_k=10, top_n=5)
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Response Time: {result['response_time']:.2f}s")
        print(f"  Top URL: {result['retrieved_chunks'][0]['url']}")
    
    print("\n  RAG system test completed successfully!")
    return rag_system


def run_evaluation(num_questions=100, output_dir='evaluation_output'):
    """Run complete evaluation pipeline"""
    print("\n" + "="*70)
    print("STEP 3: AUTOMATED EVALUATION PIPELINE")
    print("="*70)
    
    pipeline = AutomatedEvaluationPipeline(
        corpus_path='wikipedia_corpus.json',
        output_dir=output_dir
    )
    
    results = pipeline.run_complete_pipeline(num_questions)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    summary = results['summary']
    print(f"\nTotal Questions Evaluated: {summary['total_questions']}")
    print(f"Total Evaluation Time: {summary['total_evaluation_time']:.2f}s")
    print(f"Average Time per Question: {summary['avg_time_per_question']:.2f}s")
    
    print("\nKey Metrics:")
    print(f"  • MRR @ URL Level: {summary['avg_mrr_url']:.4f}")
    print(f"  • Hit Rate: {summary['avg_hit_rate']:.4f}")
    print(f"  • NDCG @ 10: {summary['avg_ndcg_at_10']:.4f}")
    print(f"  • Contextual Precision: {summary['avg_contextual_precision']:.4f}")
    print(f"  • Semantic Similarity: {summary['avg_semantic_similarity']:.4f}")
    print(f"  • Answer Faithfulness: {summary['avg_answer_faithfulness']:.4f}")
    print(f"  • ROUGE-L F1: {summary['avg_rouge_l_fmeasure']:.4f}")
    print(f"  • BERTScore F1: {summary['avg_bert_f1']:.4f}")
    print(f"  • Average Response Time: {summary['avg_response_time']:.2f}s")
    
    print(f"\nOutput Directory: {Path(output_dir).absolute()}")
    print("    evaluation_results.json")
    print("    evaluation_results.csv")
    print("    evaluation_report.pdf")
    print("    evaluation_report.html")
    print("    generated_questions.json")
    
    return results


def ablation_study():
    """Run ablation study comparing dense, sparse, and hybrid"""
    print("\n" + "="*70)
    print("STEP 4: ABLATION STUDY")
    print("="*70)
    
    print("  Ablation study would compare:")
    print("  1. Dense-only retrieval")
    print("  2. Sparse (BM25) only retrieval")
    print("  3. Hybrid (RRF) retrieval")
    print("\nNote: Implementation ready in evaluation_pipeline.py")


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid RAG System - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with 100 questions
  python main.py --full
  
  # Just test RAG system
  python main.py --test-rag
  
  # Run evaluation with custom settings
  python main.py --eval --num-questions 50 --output my_results
  
  # Setup data only
  python main.py --setup-data
        """
    )
    
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--setup-data', action='store_true', help='Only setup data')
    parser.add_argument('--test-rag', action='store_true', help='Only test RAG system')
    parser.add_argument('--eval', action='store_true', help='Only run evaluation')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--num-questions', type=int, default=100, help='Number of questions for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_output', help='Output directory')
    parser.add_argument('--fixed-urls', type=int, default=200, help='Number of fixed URLs')
    parser.add_argument('--random-urls', type=int, default=300, help='Number of random URLs')
    
    args = parser.parse_args()
    
    try:
        if args.full:
            # Run complete pipeline
            fixed_urls = setup_data(args.fixed_urls, args.random_urls)
            test_rag_system()
            run_evaluation(args.num_questions, args.output)
            ablation_study()
        
        elif args.setup_data:
            setup_data(args.fixed_urls, args.random_urls)
        
        elif args.test_rag:
            test_rag_system()
        
        elif args.eval:
            run_evaluation(args.num_questions, args.output)
        
        elif args.ablation:
            ablation_study()
        
        else:
            # Default: run full pipeline
            fixed_urls = setup_data(args.fixed_urls, args.random_urls)
            test_rag_system()
            run_evaluation(args.num_questions, args.output)
            
            print("\n" + "="*70)
            print("  PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print("\nNext Steps:")
            print("1. View results in evaluation_output/")
            print("2. Run Streamlit UI: streamlit run ui.py")
            print("3. Analyze results: Check evaluation_report.html")
            
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
