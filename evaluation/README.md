# Evaluation Folder

Contains the evaluation code and test questions.

### CONV AI - Group 11 - ASSIGNMENT 2

1. Rahul Sharma - 2024AA05893 - 100%
2. Avantika Shukla - 2024AA05303 - 100%
3. Avishek Ghatak - 2024AA05895 - 100%
4. Mayank Upadhyaya - ⁠2024AA05165 - 100%
5. Trupti Dhoble - 2024AA05300 - 100%

## Files

### evaluation.py

Implements all 8 metrics used to test the system.

Classes:

- QuestionGenerator: Creates test questions
- EvaluationMetrics: Calculates all metrics
- EvaluationPipeline: Runs evaluation on questions

Metrics:

1. MRR@URL - Mean Reciprocal Rank at URL level
2. Hit Rate - Percentage with correct URL in top 10
3. NDCG@10 - Ranking quality metric
4. Contextual Precision - URL-level precision
5. Semantic Similarity - Answer-context similarity
6. Answer Faithfulness - Answer grounded in facts
7. ROUGE-L - Word overlap metric
8. BERTScore - Neural semantic score

### evaluation_pipeline.py

Automated pipeline for end-to-end evaluation.

Does:

1. Load or generate 100 questions
2. Initialize RAG system
3. Run evaluation on all questions
4. Calculate metrics
5. Generate reports (JSON, CSV, PDF, HTML)

Usage:

```bash
python evaluation_pipeline.py --num_questions 100 --output results
```

### generated_questions.json

100 test questions with 4 types.

Structure:

```json
[
  {
    "question_id": "q_000",
    "question": "What is AI?",
    "question_type": "factual",
    "ground_truth_urls": ["https://..."],
    "ground_truth_chunk_ids": ["chunk_0"]
  }
]
```

Types:

- Factual (50): Direct facts
- Comparative (25): Comparing concepts
- Inferential (15): Reasoning needed
- Multi-hop (10): Multiple steps

### README.md

This file.

## How Metrics Work

MRR@URL:

- Find rank of first correct Wikipedia URL
- Average across all questions
- Scale: 0 to 1

Hit Rate:

- Check if correct URL in top 10 results
- Percentage that succeed
- Scale: 0 to 1

NDCG@10:

- Scores URLs by relevance
- Discounts by position
- Compares to ideal ranking
- Scale: 0 to 1

Contextual Precision:

- How many retrieved URLs are relevant
- URL-level accuracy
- Scale: 0 to 1

Semantic Similarity:

- How well answer matches context
- Uses embeddings
- Scale: 0 to 1

Answer Faithfulness:

- Fraction of answer from context
- Overlap with retrieved text
- Scale: 0 to 1

ROUGE-L:

- Longest common subsequence
- Word-level overlap
- Scale: 0 to 1

BERTScore:

- Uses BERT embeddings
- Neural semantic similarity
- Scale: 0 to 1

## Results

Typical results on 100 questions:

- MRR: 0.42
- Hit Rate: 0.58
- NDCG: 0.55
- Contextual Precision: 0.52
- Semantic Similarity: 0.71
- Answer Faithfulness: 0.78
- ROUGE-L: 0.45
- BERTScore: 0.73
- Avg Response Time: 0.156s

## Error Analysis

Failures broken down by:

- Question type: Which types fail most?
- Error category: Retrieval vs generation issues
- Metrics: Which metrics correlate with failures

## Running Evaluation

```python
from evaluation.evaluation_pipeline import AutomatedEvaluationPipeline

pipeline = AutomatedEvaluationPipeline(
    corpus_path='data/wikipedia_corpus.json',
    output_dir='evaluation_output'
)

results = pipeline.run_complete_pipeline(num_questions=100)
```

## Output Files

Results are saved as:

- evaluation_results.json
- evaluation_results.csv
- evaluation_report.pdf
- evaluation_report.html
