"""
LLM-as-Judge Evaluation Component

Uses Flan-T5 to evaluate answer quality on multiple dimensions:
- Factual Accuracy: Is the answer factually correct?
- Completeness: Does the answer fully address the question?
- Relevance: Is the answer relevant to the question?
- Coherence: Is the answer coherent and well-written?
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Any
import json

class LLMAsJudge:
    """LLM-based answer evaluation"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize LLM-as-Judge"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def evaluate_answer(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate answer on multiple dimensions
        
        Args:
            question: The question asked
            answer: The generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with evaluation scores
        """
        evaluations = {}
        
        # 1. Factual Accuracy
        accuracy_prompt = f"""Given the following context, is this answer factually accurate?

Context: {context[:500]}

Question: {question}

Answer: {answer}

Rate the factual accuracy (very low, low, medium, high, very high):"""
        
        accuracy_score = self._score_response(accuracy_prompt)
        evaluations['factual_accuracy'] = accuracy_score
        
        # 2. Completeness
        completeness_prompt = f"""Does this answer completely address the question?

Question: {question}

Answer: {answer}

Rate completeness (very incomplete, incomplete, neutral, complete, very complete):"""
        
        completeness_score = self._score_response(completeness_prompt)
        evaluations['completeness'] = completeness_score
        
        # 3. Relevance
        relevance_prompt = f"""Is the answer relevant and on-topic?

Question: {question}

Answer: {answer}

Rate relevance (very low, low, medium, high, very high):"""
        
        relevance_score = self._score_response(relevance_prompt)
        evaluations['relevance'] = relevance_score
        
        # 4. Coherence
        coherence_prompt = f"""Is the answer coherent, well-written, and easy to understand?

Answer: {answer}

Rate coherence (very low, low, medium, high, very high):"""
        
        coherence_score = self._score_response(coherence_prompt)
        evaluations['coherence'] = coherence_score
        
        # 5. Overall Score
        scores = [v for k, v in evaluations.items()]
        evaluations['overall_score'] = sum(scores) / len(scores)
        
        return evaluations
    
    def _score_response(self, prompt: str) -> float:
        """
        Generate response and score it
        
        Returns score from 0-1
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50, num_beams=1)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            
            # Map response to score
            score_map = {
                'very low': 0.1,
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7,
                'very high': 0.9,
                'very incomplete': 0.1,
                'incomplete': 0.3,
                'neutral': 0.5,
                'complete': 0.7,
                'very complete': 0.9,
            }
            
            for key, score in score_map.items():
                if key in response:
                    return score
            
            return 0.5  # default to neutral
            
        except Exception as e:
            print(f"Error in scoring: {e}")
            return 0.5
    
    def evaluate_batch(self, questions: List[Dict], results: List[Dict]) -> List[Dict]:
        """Evaluate a batch of results"""
        evaluated_results = []
        
        for question, result in zip(questions, results):
            evaluation = self.evaluate_answer(
                question=question['question'],
                answer=result.get('answer', ''),
                context=result.get('context', '')
            )
            
            result['llm_judge_scores'] = evaluation
            evaluated_results.append(result)
        
        return evaluated_results


if __name__ == "__main__":
    # Example usage
    judge = LLMAsJudge()
    
    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    context = "Machine learning is a branch of AI..."
    
    scores = judge.evaluate_answer(question, answer, context)
    print(json.dumps(scores, indent=2))
