"""
Adversarial Testing Component

Tests system robustness with:
- Unanswerable questions (hallucination detection)
- Paraphrased questions (robustness)
- Negated statements (semantic understanding)
- Multi-hop reasoning challenges
- Ambiguous questions
"""

import json
from typing import List, Dict, Any
import random

class AdversarialTestSuite:
    """Create and manage adversarial test cases"""
    
    def __init__(self):
        self.adversarial_questions = []
    
    def create_unanswerable_questions(self, base_questions: List[Dict], num: int = 10) -> List[Dict]:
        """
        Create unanswerable questions to detect hallucination
        These have no relevant source document
        """
        unanswerable = []
        for _ in range(min(num, len(base_questions))):
            q = random.choice(base_questions)
            
            # Create unanswerable variant by asking about non-existent things
            unanswerable_variants = [
                f"What is the connection between {q.get('question', '')[:30]}... and fictional elements?",
                f"When did {q.get('question', '')[:30]}... occur in an alternate universe?",
                f"What hypothetical scenario explains {q.get('question', '')[:30]}...?",
                f"Who invented the non-existent concept related to {q.get('question', '')[:30]}...?",
            ]
            
            unanswerable_q = {
                'question_id': f"adv_unansw_{len(unanswerable)}",
                'question': random.choice(unanswerable_variants),
                'question_type': 'unanswerable',
                'ground_truth_urls': [],  # Empty: no correct answer exists
                'adversarial_type': 'unanswerable'
            }
            unanswerable.append(unanswerable_q)
        
        return unanswerable
    
    def create_paraphrased_questions(self, base_questions: List[Dict], num: int = 10) -> List[Dict]:
        """
        Create paraphrased versions to test robustness
        """
        paraphrased = []
        for _ in range(min(num, len(base_questions))):
            q = random.choice(base_questions)
            original_q = q['question']
            
            # Simple paraphrasing strategies
            paraphrase_templates = [
                f"In what way does {original_q.split()[1:3] if len(original_q.split()) > 2 else 'this topic'} relate to the broader context?",
                f"Explain the concept: {original_q.lower()}",
                f"Can you elaborate on {original_q.lower()}?",
                f"Describe in detail: {original_q.lower()}",
            ]
            
            paraphrased_q = {
                'question_id': f"adv_paraph_{len(paraphrased)}",
                'question': random.choice(paraphrase_templates),
                'question_type': q.get('question_type', 'factual'),
                'ground_truth_urls': q.get('ground_truth_urls', []),
                'original_question_id': q['question_id'],
                'adversarial_type': 'paraphrased'
            }
            paraphrased.append(paraphrased_q)
        
        return paraphrased
    
    def create_negated_questions(self, base_questions: List[Dict], num: int = 10) -> List[Dict]:
        """
        Create negated/opposite questions to test semantic understanding
        """
        negated = []
        negation_patterns = ['not', 'unlike', 'does NOT', 'cannot', 'is NOT']
        
        for _ in range(min(num, len(base_questions))):
            q = random.choice(base_questions)
            original_q = q['question']
            
            # Add negation
            negated_q_text = f"{random.choice(negation_patterns)}: {original_q}"
            
            negated_question = {
                'question_id': f"adv_neg_{len(negated)}",
                'question': negated_q_text,
                'question_type': q.get('question_type', 'factual'),
                'ground_truth_urls': q.get('ground_truth_urls', []),
                'original_question_id': q['question_id'],
                'adversarial_type': 'negated'
            }
            negated.append(negated_question)
        
        return negated
    
    def create_multi_hop_challenges(self, base_questions: List[Dict], num: int = 10) -> List[Dict]:
        """
        Create multi-hop reasoning challenges
        """
        multi_hop = []
        for _ in range(min(num, len(base_questions))):
            q1 = random.choice(base_questions)
            q2 = random.choice(base_questions)
            
            multi_hop_q = {
                'question_id': f"adv_multihop_{len(multi_hop)}",
                'question': f"How do {q1['question'][:40]}... and {q2['question'][:40]}... relate?",
                'question_type': 'multi-hop',
                'ground_truth_urls': list(set(q1.get('ground_truth_urls', []) + q2.get('ground_truth_urls', []))),
                'adversarial_type': 'multi-hop',
                'base_questions': [q1['question_id'], q2['question_id']]
            }
            multi_hop.append(multi_hop_q)
        
        return multi_hop
    
    def create_ambiguous_questions(self, base_questions: List[Dict], num: int = 10) -> List[Dict]:
        """
        Create ambiguous questions that could have multiple interpretations
        """
        ambiguous = []
        for _ in range(min(num, len(base_questions))):
            q = random.choice(base_questions)
            original_q = q['question']
            
            # Make question ambiguous by removing context
            ambiguous_variants = [
                f"What is it? (referring to {original_q[:30]}...)",
                f"Describe this in multiple ways: {original_q}",
                f"This could mean several things: {original_q}",
            ]
            
            ambiguous_q = {
                'question_id': f"adv_ambig_{len(ambiguous)}",
                'question': random.choice(ambiguous_variants),
                'question_type': 'ambiguous',
                'ground_truth_urls': q.get('ground_truth_urls', []),
                'adversarial_type': 'ambiguous'
            }
            ambiguous.append(ambiguous_q)
        
        return ambiguous
    
    def generate_full_suite(self, base_questions: List[Dict]) -> List[Dict]:
        """Generate complete adversarial test suite"""
        suite = []
        
        suite.extend(self.create_unanswerable_questions(base_questions, 10))
        suite.extend(self.create_paraphrased_questions(base_questions, 10))
        suite.extend(self.create_negated_questions(base_questions, 10))
        suite.extend(self.create_multi_hop_challenges(base_questions, 10))
        suite.extend(self.create_ambiguous_questions(base_questions, 10))
        
        self.adversarial_questions = suite
        return suite
    
    def save_suite(self, output_file: str):
        """Save adversarial test suite to file"""
        with open(output_file, 'w') as f:
            json.dump(self.adversarial_questions, f, indent=2)


if __name__ == "__main__":
    # Example usage
    suite = AdversarialTestSuite()
    sample_questions = [
        {'question': 'What is AI?', 'ground_truth_urls': ['url1'], 'question_type': 'factual'},
        {'question': 'How does ML work?', 'ground_truth_urls': ['url2'], 'question_type': 'factual'},
    ]
    
    adversarial_questions = suite.generate_full_suite(sample_questions)
    print(f"Generated {len(adversarial_questions)} adversarial questions")
