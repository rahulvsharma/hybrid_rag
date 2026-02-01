"""
Confidence Calibration Component

Measures correlation between model confidence and actual correctness
Helps understand when the system is reliable vs overconfident
"""

import json
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

class ConfidenceCalibration:
    """Analyze confidence-correctness correlation"""
    
    def __init__(self):
        self.confidences = []
        self.correctness = []
    
    def compute_confidence_scores(self, results: List[Dict]) -> List[float]:
        """
        Compute confidence for each answer
        Based on: MRR, semantic similarity, response consistency
        """
        confidences = []
        
        for result in results:
            # Confidence based on multiple factors
            mrr_conf = result.get('mrr', 0) * 0.4
            semantic_conf = result.get('semantic_similarity', 0) * 0.35
            bert_conf = result.get('bert_score', 0) * 0.25
            
            confidence = mrr_conf + semantic_conf + bert_conf
            confidences.append(confidence)
        
        return confidences
    
    def compute_correctness(self, results: List[Dict], threshold: float = 0.5) -> List[int]:
        """
        Determine correctness (binary)
        Answer is correct if semantic_similarity > threshold
        """
        correctness = []
        
        for result in results:
            semantic_sim = result.get('semantic_similarity', 0)
            is_correct = 1 if semantic_sim > threshold else 0
            correctness.append(is_correct)
        
        return correctness
    
    def plot_calibration_curve(self, confidences: List[float], 
                               correctness: List[int], 
                               output_file: str = "calibration_curve.png"):
        """Plot calibration curve"""
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            correctness, confidences, n_bins=10
        )
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(prob_pred, prob_true, 'bo-', label='Model Calibration')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Fraction of Positives (Correct Answers)')
        plt.title('Confidence Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def compute_ece(self, confidences: List[float], correctness: List[int], 
                    n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error
        Measures how well predicted confidence matches actual accuracy
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        total_samples = len(correctness)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Get samples in this bin
            in_bin = [
                (conf, correct) for conf, correct in zip(confidences, correctness)
                if bin_lower <= conf <= bin_upper
            ]
            
            if len(in_bin) == 0:
                continue
            
            # Compute gap
            avg_confidence = np.mean([c[0] for c in in_bin])
            avg_accuracy = np.mean([c[1] for c in in_bin])
            gap = abs(avg_confidence - avg_accuracy)
            
            # Weight by bin size
            weight = len(in_bin) / total_samples
            ece += weight * gap
        
        return ece
    
    def analyze_calibration(self, results: List[Dict]) -> Dict[str, Any]:
        """Complete calibration analysis"""
        
        confidences = self.compute_confidence_scores(results)
        correctness = self.compute_correctness(results)
        
        ece = self.compute_ece(confidences, correctness)
        
        analysis = {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_correctness': float(np.mean(correctness)),
            'expected_calibration_error': float(ece),
            'interpretation': self._interpret_calibration(ece),
            'confidence_scores': confidences,
            'correctness_labels': correctness
        }
        
        return analysis
    
    def _interpret_calibration(self, ece: float) -> str:
        """Interpret ECE score"""
        if ece < 0.05:
            return "Excellent: Model confidence well-calibrated"
        elif ece < 0.1:
            return "Good: Model confidence mostly reliable"
        elif ece < 0.15:
            return "Fair: Some miscalibration"
        elif ece < 0.2:
            return "Poor: Significant miscalibration"
        else:
            return "Very Poor: Model is overconfident or underconfident"


if __name__ == "__main__":
    # Example usage
    sample_results = [
        {'mrr': 0.8, 'semantic_similarity': 0.9, 'bert_score': 0.85},
        {'mrr': 0.3, 'semantic_similarity': 0.4, 'bert_score': 0.35},
    ]
    
    calibrator = ConfidenceCalibration()
    analysis = calibrator.analyze_calibration(sample_results)
    print(json.dumps(analysis, indent=2))
