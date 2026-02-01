"""
Automated evaluation pipeline and report generation
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from rag_system import HybridRAGSystem
from evaluation import QuestionGenerator, EvaluationPipeline


class AutomatedEvaluationPipeline:
    """Automated end-to-end evaluation pipeline"""
    
    def __init__(self, corpus_path: str = 'wikipedia_corpus.json', 
                 questions_path: str = None,
                 output_dir: str = 'evaluation_output'):
        self.corpus_path = corpus_path
        self.questions_path = questions_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.rag_system = None
        self.questions = []
        self.evaluation_results = None
    
    def load_or_generate_questions(self, num_questions: int = 100) -> List[Dict]:
        """Load questions from file or generate new ones"""
        if self.questions_path and Path(self.questions_path).exists():
            print(f"Loading questions from {self.questions_path}...")
            with open(self.questions_path, 'r') as f:
                self.questions = json.load(f)
        else:
            print(f"Generating {num_questions} questions...")
            with open(self.corpus_path, 'r') as f:
                corpus = json.load(f)
            
            generator = QuestionGenerator()
            self.questions = generator.generate_questions(corpus['chunks'], num_questions)
            
            # Save questions
            questions_file = self.output_dir / 'generated_questions.json'
            with open(questions_file, 'w') as f:
                json.dump(self.questions, f, indent=2)
            print(f"Questions saved to {questions_file}")
        
        return self.questions
    
    def initialize_rag_system(self) -> HybridRAGSystem:
        """Initialize RAG system"""
        print("Initializing RAG system...")
        self.rag_system = HybridRAGSystem(self.corpus_path)
        return self.rag_system
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        if not self.rag_system:
            self.initialize_rag_system()
        
        if not self.questions:
            self.load_or_generate_questions()
        
        print(f"\nRunning evaluation on {len(self.questions)} questions...\n")
        
        # Get RAG results
        rag_results = []
        start_time = time.time()
        
        for i, question in enumerate(self.questions, 1):
            print(f"Processing question {i}/{len(self.questions)}: {question['question'][:60]}...")
            result = self.rag_system.query(question['question'], top_k=10, top_n=5)
            rag_results.append(result)
        
        total_time = time.time() - start_time
        
        # Evaluate results
        print("\nCalculating metrics...")
        pipeline = EvaluationPipeline()
        evaluation_results = pipeline.evaluate_batch(self.questions, rag_results)
        
        # Add timing info
        evaluation_results['evaluation_summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(self.questions),
            'total_evaluation_time': total_time,
            'avg_time_per_question': total_time / len(self.questions)
        }
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON"""
        output_file = self.output_dir / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    def generate_csv_report(self, results: Dict[str, Any]) -> None:
        """Generate CSV report from results"""
        df = pd.DataFrame(results['detailed_results'])
        
        output_file = self.output_dir / 'evaluation_results.csv'
        df.to_csv(output_file, index=False)
        print(f"CSV report saved to {output_file}")
    
    def generate_pdf_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive PDF report"""
        output_file = self.output_dir / 'evaluation_report.pdf'
        
        with PdfPages(output_file) as pdf:
            # Page 1: Summary
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Hybrid RAG System - Evaluation Report', fontsize=16, fontweight='bold')
            
            summary = results['summary']
            
            # Create text content
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            summary_text = f"""
EVALUATION SUMMARY
{'='*60}

Evaluation Date: {results['evaluation_summary']['timestamp']}
Total Questions Evaluated: {results['evaluation_summary']['total_questions']}
Total Evaluation Time: {results['evaluation_summary']['total_evaluation_time']:.2f}s
Average Time per Question: {results['evaluation_summary']['avg_time_per_question']:.2f}s

KEY METRICS
{'='*60}

Mandatory Metric:
• Mean Reciprocal Rank (MRR) @ URL Level: {summary['avg_mrr_url']:.4f}
  - Measures how quickly the system identifies the correct source document

Custom Metrics:
1. Hit Rate (Precision @ K): {summary['avg_hit_rate']:.4f}
   - Fraction of retrieved documents that are relevant
   
2. NDCG @ 10: {summary['avg_ndcg_at_10']:.4f}
   - Normalized Discounted Cumulative Gain accounting for ranking quality

3. Contextual Precision: {summary['avg_contextual_precision']:.4f}
   - Precision of retrieval at URL level

4. Semantic Similarity: {summary['avg_semantic_similarity']:.4f}
   - Semantic match between generated answer and context

5. Answer Faithfulness: {summary['avg_answer_faithfulness']:.4f}
   - Fraction of answer grounded in retrieved context

6. ROUGE-L F-Measure: {summary['avg_rouge_l_fmeasure']:.4f}
   - Lexical overlap between answer and context

7. BERTScore F1: {summary['avg_bert_f1']:.4f}
   - Semantic similarity using BERT embeddings

EFFICIENCY METRICS
{'='*60}

Average Response Time: {summary['avg_response_time']:.2f}s
- Measures end-to-end latency for query processing
            """
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
            # Page 2: Metrics Visualization
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('Metrics Analysis', fontsize=14, fontweight='bold')
            
            # Metric scores
            metrics = ['MRR@URL', 'Hit Rate', 'NDCG@10', 'Contextual Precision', 
                      'Semantic Sim', 'Answer Faith', 'ROUGE-L', 'BERTScore']
            scores = [summary['avg_mrr_url'], summary['avg_hit_rate'], 
                     summary['avg_ndcg_at_10'], summary['avg_contextual_precision'],
                     summary['avg_semantic_similarity'], summary['avg_answer_faithfulness'],
                     summary['avg_rouge_l_fmeasure'], summary['avg_bert_f1']]
            
            # Bar plot
            axes[0, 0].barh(metrics[:4], scores[:4], color='steelblue')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_title('Retrieval Metrics')
            axes[0, 0].set_xlim(0, 1)
            
            axes[0, 1].barh(metrics[4:], scores[4:], color='forestgreen')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_title('Generation & Quality Metrics')
            axes[0, 1].set_xlim(0, 1)
            
            # Response time histogram
            response_times = [r['response_time'] for r in results['detailed_results']]
            axes[1, 0].hist(response_times, bins=20, color='coral', edgecolor='black')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Response Time Distribution')
            
            # Question type distribution
            question_types = [r['question'][:30] for r in results['detailed_results'][:10]]
            axes[1, 1].axis('off')
            axes[1, 1].text(0.05, 0.95, f'Sample Questions (First 10):\n' + 
                           '\n'.join([f"{i+1}. {q}" for i, q in enumerate(question_types)]),
                           transform=axes[1, 1].transAxes, fontsize=8, verticalalignment='top')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Detailed Results Table
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Detailed Results Table', fontsize=14, fontweight='bold')
            
            ax = fig.add_subplot(111)
            ax.axis('tight')
            ax.axis('off')
            
            # Create table data
            df = pd.DataFrame(results['detailed_results'])
            table_data = df[['question_id', 'mrr_url', 'hit_rate', 'ndcg_at_10', 
                           'contextual_precision', 'semantic_similarity', 'response_time']].head(20)
            table_data.columns = ['Q ID', 'MRR', 'Hit Rate', 'NDCG', 'Prec', 'SemSim', 'Time(s)']
            
            # Format numeric columns
            for col in table_data.columns[1:]:
                table_data[col] = table_data[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            
            table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"PDF report saved to {output_file}")
    
    def generate_html_report(self, results: Dict[str, Any]) -> None:
        """Generate HTML report"""
        output_file = self.output_dir / 'evaluation_report.html'
        
        summary = results['summary']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid RAG System - Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; }}
        h1, h2 {{ color: #2c3e50; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; margin-top: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #667eea; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary-box {{ background-color: #ecf0f1; padding: 15px; border-left: 4px solid #667eea; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Hybrid RAG System - Evaluation Report</h1>
        
        <div class="summary-box">
            <h3>Evaluation Summary</h3>
            <p><strong>Timestamp:</strong> {results['evaluation_summary']['timestamp']}</p>
            <p><strong>Total Questions:</strong> {results['evaluation_summary']['total_questions']}</p>
            <p><strong>Total Time:</strong> {results['evaluation_summary']['total_evaluation_time']:.2f}s</p>
            <p><strong>Average Time/Question:</strong> {results['evaluation_summary']['avg_time_per_question']:.2f}s</p>
        </div>
        
        <h2>Key Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{summary['avg_mrr_url']:.4f}</div>
                <div class="metric-label">MRR @ URL</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_hit_rate']:.4f}</div>
                <div class="metric-label">Hit Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_ndcg_at_10']:.4f}</div>
                <div class="metric-label">NDCG@10</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_contextual_precision']:.4f}</div>
                <div class="metric-label">Context Precision</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{summary['avg_semantic_similarity']:.4f}</div>
                <div class="metric-label">Semantic Similarity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_answer_faithfulness']:.4f}</div>
                <div class="metric-label">Answer Faithfulness</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_rouge_l_fmeasure']:.4f}</div>
                <div class="metric-label">ROUGE-L F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['avg_bert_f1']:.4f}</div>
                <div class="metric-label">BERTScore F1</div>
            </div>
        </div>
        
        <h2>Metrics Explanation</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Description</th>
                <th>Score</th>
            </tr>
            <tr>
                <td>MRR @ URL Level</td>
                <td>Mean Reciprocal Rank - Position of first correct Wikipedia URL in results</td>
                <td>{summary['avg_mrr_url']:.4f}</td>
            </tr>
            <tr>
                <td>Hit Rate</td>
                <td>Fraction of retrieved documents that are relevant (Precision@K)</td>
                <td>{summary['avg_hit_rate']:.4f}</td>
            </tr>
            <tr>
                <td>NDCG@10</td>
                <td>Normalized Discounted Cumulative Gain accounting for ranking position</td>
                <td>{summary['avg_ndcg_at_10']:.4f}</td>
            </tr>
            <tr>
                <td>Contextual Precision</td>
                <td>Precision of URL-level retrieval results</td>
                <td>{summary['avg_contextual_precision']:.4f}</td>
            </tr>
            <tr>
                <td>Semantic Similarity</td>
                <td>Cosine similarity between answer and context using BERT embeddings</td>
                <td>{summary['avg_semantic_similarity']:.4f}</td>
            </tr>
            <tr>
                <td>Answer Faithfulness</td>
                <td>Fraction of answer tokens grounded in retrieved context</td>
                <td>{summary['avg_answer_faithfulness']:.4f}</td>
            </tr>
            <tr>
                <td>ROUGE-L F1</td>
                <td>Longest common subsequence F1 score between answer and context</td>
                <td>{summary['avg_rouge_l_fmeasure']:.4f}</td>
            </tr>
            <tr>
                <td>BERTScore F1</td>
                <td>Semantic similarity using contextual word embeddings</td>
                <td>{summary['avg_bert_f1']:.4f}</td>
            </tr>
        </table>
        
        <h2>Detailed Results</h2>
        <p>Total questions evaluated: {len(results['detailed_results'])}</p>
        <p><a href="evaluation_results.csv">Download detailed results as CSV</a></p>
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_file}")
    
    def run_complete_pipeline(self, num_questions: int = 100) -> Dict[str, Any]:
        """Run complete evaluation pipeline with all outputs"""
        print("="*70)
        print("HYBRID RAG SYSTEM - AUTOMATED EVALUATION PIPELINE")
        print("="*70)
        
        # Load/Generate questions
        self.load_or_generate_questions(num_questions)
        
        # Initialize system
        self.initialize_rag_system()
        
        # Run evaluation
        results = self.run_evaluation()
        
        # Save results
        self.save_results(results)
        self.generate_csv_report(results)
        self.generate_pdf_report(results)
        self.generate_html_report(results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Output directory: {self.output_dir.absolute()}")
        
        return results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hybrid RAG evaluation pipeline')
    parser.add_argument('--num_questions', type=int, default=100, help='Number of questions to evaluate')
    parser.add_argument('--corpus', type=str, default='wikipedia_corpus.json', help='Path to corpus JSON')
    parser.add_argument('--questions', type=str, default=None, help='Path to existing questions JSON')
    parser.add_argument('--output', type=str, default='evaluation_output', help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = AutomatedEvaluationPipeline(
        corpus_path=args.corpus,
        questions_path=args.questions,
        output_dir=args.output
    )
    
    results = pipeline.run_complete_pipeline(args.num_questions)
    
    print("\nEvaluation Summary:")
    print(json.dumps(results['summary'], indent=2))


if __name__ == "__main__":
    main()
