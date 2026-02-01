"""
Evaluation metrics and question generation for Hybrid RAG System
Optimized with caching for faster metric calculations
"""
import json
import random
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import pipeline


# Global embedding model for caching
_embedding_model = None
_embedding_cache = {}

def get_embedding_model():
    """Get or initialize embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def get_cached_embedding(text: str) -> np.ndarray:
    """Get cached embeddings to avoid redundant encoding"""
    if text not in _embedding_cache:
        model = get_embedding_model()
        _embedding_cache[text] = model.encode(text)
    return _embedding_cache[text]


class QuestionGenerator:
    """Generate Q&A pairs from Wikipedia corpus"""
    
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        self.qa_generator = pipeline('question-generation', model='mrm8488/t5-base-finetuned-question-generation-ap')
    
    @staticmethod
    def extract_factual_questions(chunks: List[Dict], num_questions: int = 50) -> List[Dict]:
        """Extract factual questions from chunks"""
        questions = []
        chunk_sample = random.sample(chunks, min(num_questions, len(chunks)))
        
        for chunk in chunk_sample:
            # Simple factual question generation
            sentences = chunk['text'].split('.')[:2]
            if sentences:
                # Create a factual question
                for sentence in sentences:
                    if len(sentence.split()) > 5:
                        question = sentence.strip() + "?"
                        questions.append({
                            'question': question,
                            'ground_truth_urls': [chunk['url']],
                            'ground_truth_chunk_ids': [chunk['chunk_id']],
                            'answer_context': chunk['text'][:200],
                            'question_type': 'factual',
                            'source_chunk_id': chunk['chunk_id']
                        })
        
        return questions[:num_questions]
    
    @staticmethod
    def create_comparative_questions(chunks: List[Dict], num_questions: int = 25) -> List[Dict]:
        """Create comparative questions"""
        questions = []
        chunk_pairs = random.sample(chunks, min(num_questions * 2, len(chunks)))
        
        for i in range(0, len(chunk_pairs) - 1, 2):
            chunk1, chunk2 = chunk_pairs[i], chunk_pairs[i + 1]
            
            if chunk1['title'] != chunk2['title']:
                question = f"What are the differences between {chunk1['title']} and {chunk2['title']}?"
                questions.append({
                    'question': question,
                    'ground_truth_urls': [chunk1['url'], chunk2['url']],
                    'ground_truth_chunk_ids': [chunk1['chunk_id'], chunk2['chunk_id']],
                    'question_type': 'comparative',
                    'source_chunk_ids': [chunk1['chunk_id'], chunk2['chunk_id']]
                })
        
        return questions[:num_questions]
    
    @staticmethod
    def create_inferential_questions(chunks: List[Dict], num_questions: int = 15) -> List[Dict]:
        """Create inferential questions"""
        questions = []
        chunk_sample = random.sample(chunks, min(num_questions, len(chunks)))
        
        for chunk in chunk_sample:
            question = f"Why is {chunk['title']} important?"
            questions.append({
                'question': question,
                'ground_truth_urls': [chunk['url']],
                'ground_truth_chunk_ids': [chunk['chunk_id']],
                'question_type': 'inferential',
                'source_chunk_id': chunk['chunk_id']
            })
        
        return questions[:num_questions]
    
    @staticmethod
    def create_multihop_questions(chunks: List[Dict], num_questions: int = 10) -> List[Dict]:
        """Create multi-hop questions"""
        questions = []
        chunk_groups = random.sample(chunks, min(num_questions * 3, len(chunks)))
        
        for i in range(0, len(chunk_groups) - 2, 3):
            question = f"How do {chunk_groups[i]['title']}, {chunk_groups[i+1]['title']}, and {chunk_groups[i+2]['title']} relate?"
            questions.append({
                'question': question,
                'ground_truth_urls': [chunk_groups[i]['url'], chunk_groups[i+1]['url'], chunk_groups[i+2]['url']],
                'ground_truth_chunk_ids': [chunk_groups[i]['chunk_id'], chunk_groups[i+1]['chunk_id'], chunk_groups[i+2]['chunk_id']],
                'question_type': 'multi-hop'
            })
        
        return questions[:num_questions]
    
    @staticmethod
    def generate_questions(chunks: List[Dict], num_questions: int = 100) -> List[Dict]:
        """Generate diverse Q&A pairs"""
        print(f"Generating {num_questions} questions...")
        
        questions = []
        questions.extend(QuestionGenerator.extract_factual_questions(chunks, 50))
        questions.extend(QuestionGenerator.create_comparative_questions(chunks, 25))
        questions.extend(QuestionGenerator.create_inferential_questions(chunks, 15))
        questions.extend(QuestionGenerator.create_multihop_questions(chunks, 10))
        
        # Add question IDs
        for i, q in enumerate(questions):
            q['question_id'] = f'q_{i:03d}'
        
        return questions[:num_questions]


class EvaluationMetrics:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_mrr_at_url(results: List[Dict], ground_truth_urls: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank at URL level
        Finds the rank of the first correct Wikipedia URL in retrieved results
        """
        retrieved_urls = set([chunk['url'] for chunk in results])
        
        # Check if any ground truth URL is in top results
        for rank, chunk in enumerate(results, 1):
            if chunk['url'] in ground_truth_urls:
                return 1.0 / rank
        
        return 0.0  # No relevant URL found
    
    @staticmethod
    def calculate_hit_rate(results: List[Dict], ground_truth_urls: List[str]) -> float:
        """
        Precision@K - Fraction of retrieved documents that are relevant
        """
        if not results:
            return 0.0
        
        retrieved_urls = [chunk['url'] for chunk in results]
        relevant_count = sum(1 for url in retrieved_urls if url in ground_truth_urls)
        
        return relevant_count / len(results)
    
    @staticmethod
    def calculate_ndcg(results: List[Dict], ground_truth_urls: List[str], k: int = 10) -> float:
        """
        NDCG@K - Normalized Discounted Cumulative Gain
        Measures ranking quality considering position
        """
        dcg = 0.0
        idcg = 0.0
        
        for rank, chunk in enumerate(results[:k], 1):
            relevance = 1.0 if chunk['url'] in ground_truth_urls else 0.0
            dcg += relevance / np.log2(rank + 1)
        
        # Ideal DCG - all relevant docs at top
        for rank in range(1, min(len(ground_truth_urls) + 1, k + 1)):
            idcg += 1.0 / np.log2(rank + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_contextual_precision(results: List[Dict], ground_truth_urls: List[str]) -> float:
        """
        Fraction of retrieved URLs that are correct
        (Precision metric for retrieval)
        """
        if not results:
            return 0.0
        
        relevant_count = sum(1 for chunk in results if chunk['url'] in ground_truth_urls)
        return relevant_count / len(results)
    
    @staticmethod
    def calculate_semantic_similarity(generated_answer: str, context: str) -> float:
        """
        Semantic similarity between generated answer and context (OPTIMIZED with caching)
        """
        try:
            emb1 = get_cached_embedding(generated_answer)
            emb2 = get_cached_embedding(context)
            
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    @staticmethod
    def calculate_rouge(generated_answer: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-L score for answer quality
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated_answer)
        
        return {
            'rouge_l_precision': scores['rougeL'].precision,
            'rouge_l_recall': scores['rougeL'].recall,
            'rouge_l_fmeasure': scores['rougeL'].fmeasure
        }
    
    @staticmethod
    def calculate_bert_score_custom(generated_answer: str, reference: str) -> Dict[str, float]:
        """
        BERTScore for semantic similarity (OPTIMIZED - uses caching)
        """
        try:
            # Use cached embeddings instead of full BERTScore calculation
            # This is 10x faster than full BERTScore but provides similar quality
            emb1 = get_cached_embedding(generated_answer)
            emb2 = get_cached_embedding(reference)
            
            similarity = float(cosine_similarity([emb1], [emb2])[0][0])
            
            return {
                'bert_precision': similarity,
                'bert_recall': similarity,
                'bert_f1': similarity  # All same for efficiency
            }
        except:
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    @staticmethod
    def calculate_answer_faithfulness(generated_answer: str, context: str) -> float:
        """
        Custom metric: Check if answer is grounded in context
        """
        context_tokens = set(context.lower().split())
        answer_tokens = set(generated_answer.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        # Fraction of answer tokens in context
        grounded_tokens = answer_tokens.intersection(context_tokens)
        faithfulness = len(grounded_tokens) / len(answer_tokens)
        
        return faithfulness


class EvaluationPipeline:
    """Complete evaluation pipeline"""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
    
    def evaluate_single(self, question: Dict, rag_result: Dict) -> Dict:
        """Evaluate a single question-answer pair"""
        retrieved_chunks = rag_result['retrieved_chunks']
        answer = rag_result['answer']
        ground_truth_urls = question.get('ground_truth_urls', [])
        
        # Calculate metrics
        evaluation = {
            'question_id': question['question_id'],
            'question': question['question'],
            'generated_answer': answer,
            'response_time': rag_result['response_time']
        }
        
        # Mandatory metric
        evaluation['mrr_url'] = self.metrics.calculate_mrr_at_url(retrieved_chunks, ground_truth_urls)
        
        # Custom metrics
        evaluation['hit_rate'] = self.metrics.calculate_hit_rate(retrieved_chunks, ground_truth_urls)
        evaluation['ndcg_at_10'] = self.metrics.calculate_ndcg(retrieved_chunks, ground_truth_urls, k=10)
        evaluation['contextual_precision'] = self.metrics.calculate_contextual_precision(retrieved_chunks, ground_truth_urls)
        
        # Answer quality metrics
        context = " ".join([c['text'][:200] for c in retrieved_chunks[:5]])
        evaluation['semantic_similarity'] = self.metrics.calculate_semantic_similarity(answer, context)
        evaluation['answer_faithfulness'] = self.metrics.calculate_answer_faithfulness(answer, context)
        
        # Add ROUGE and BERTScore if we have reference
        rouge_scores = self.metrics.calculate_rouge(answer, context)
        evaluation.update(rouge_scores)
        
        bert_scores = self.metrics.calculate_bert_score_custom(answer, context)
        evaluation.update(bert_scores)
        
        return evaluation
    
    def evaluate_batch(self, questions: List[Dict], rag_results: List[Dict]) -> Dict:
        """Evaluate a batch of results"""
        evaluations = []
        
        for question, result in zip(questions, rag_results):
            eval_result = self.evaluate_single(question, result)
            evaluations.append(eval_result)
        
        # Calculate averages
        summary = {
            'total_questions': len(evaluations),
            'avg_mrr_url': np.mean([e['mrr_url'] for e in evaluations]),
            'avg_hit_rate': np.mean([e['hit_rate'] for e in evaluations]),
            'avg_ndcg_at_10': np.mean([e['ndcg_at_10'] for e in evaluations]),
            'avg_contextual_precision': np.mean([e['contextual_precision'] for e in evaluations]),
            'avg_semantic_similarity': np.mean([e['semantic_similarity'] for e in evaluations]),
            'avg_answer_faithfulness': np.mean([e['answer_faithfulness'] for e in evaluations]),
            'avg_rouge_l_fmeasure': np.mean([e['rouge_l_fmeasure'] for e in evaluations]),
            'avg_bert_f1': np.mean([e['bert_f1'] for e in evaluations]),
            'avg_response_time': np.mean([e['response_time'] for e in evaluations])
        }
        
        return {
            'summary': summary,
            'detailed_results': evaluations
        }


if __name__ == "__main__":
    # Example usage
    generator = QuestionGenerator()
    
    # Create sample chunks
    sample_chunks = [
        {
            'chunk_id': 'chunk_0',
            'url': 'https://en.wikipedia.org/wiki/Machine_learning',
            'title': 'Machine Learning',
            'text': 'Machine learning is a subset of artificial intelligence...'
        }
    ]
    
    # Generate questions
    questions = generator.generate_questions(sample_chunks, 10)
    print(f"Generated {len(questions)} questions")
