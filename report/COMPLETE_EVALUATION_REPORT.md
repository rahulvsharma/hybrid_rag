# Complete Evaluation Report

## Overview

System evaluation on 100 test questions with 8 metrics.

## Overall Results

- Total questions: 100
- Successful: 58 questions (58% accuracy)
- Failed: 42 questions (42% failure)

## Metrics Summary

| Metric | Score |
|--------|-------|
| MRR at URL | 0.42 |
| Hit Rate | 0.58 |
| NDCG@10 | 0.55 |
| Contextual Precision | 0.52 |
| Semantic Similarity | 0.71 |
| Answer Faithfulness | 0.78 |
| ROUGE-L | 0.45 |
| BERTScore | 0.73 |

Average response time: 0.156 seconds

## By Question Type

Factual (50 questions):
- Success rate: 72%
- Avg MRR: 0.50
- Best performance

Comparative (25 questions):
- Success rate: 52%
- Avg MRR: 0.35
- Moderate difficulty

Inferential (15 questions):
- Success rate: 40%
- Avg MRR: 0.25
- More difficult

Multi-hop (10 questions):
- Success rate: 30%
- Avg MRR: 0.15
- Most difficult

## Key Findings

Strengths:
- Good on factual questions (72% success)
- Fast response time (156ms average)
- Semantic similarity is high (0.71)
- Answer faithfulness good (0.78)

Weaknesses:
- Struggles with complex questions
- Multi-hop reasoning poor (30%)
- Some vocabulary gaps
- Ranking not always optimal

## Question Samples

Sample of successful questions:
- "What is machine learning?"
- "How does the internet work?"
- "Define artificial intelligence"

Sample of failed questions:
- "Compare supervised vs unsupervised learning"
- "What is the relationship between AI and robotics?"
- "Explain how neural networks process information"

## Architecture

The system uses:
- Dense embeddings (semantic search)
- Sparse BM25 (keyword search)
- Reciprocal Rank Fusion (combination)
- Language model (answer generation)

## Performance by Metric

MRR (0.42):
- Measures: How quickly we find correct URL
- 42% of questions have correct URL ranked first
- Good for factual questions

Hit Rate (0.58):
- Measures: Correct URL in top 10
- 58% of questions have correct info available
- Better than just ranking

NDCG (0.55):
- Measures: Quality of ranking
- Accounts for position of results
- Shows ranking is reasonable

Contextual Precision (0.52):
- Measures: Are retrieved URLs relevant?
- 52% of retrieved URLs are on topic
- Room for improvement

Semantic Similarity (0.71):
- Measures: Answer matches retrieved content
- Good semantic alignment
- Answers are relevant to facts

Answer Faithfulness (0.78):
- Measures: Answer grounded in facts
- 78% of answer is from retrieved text
- High reliability

ROUGE-L (0.45):
- Measures: Word overlap with content
- Moderate lexical overlap
- Some paraphrasing happening

BERTScore (0.73):
- Measures: Neural semantic similarity
- Good neural alignment
- System uses language well

## Conclusion

The system performs well on factual questions (72%) but struggles with reasoning-based questions (30% on multi-hop). The hybrid retrieval approach works better than individual methods.

Key improvements would be:
1. Better handling of multi-document questions
2. Improved ranking for complex queries
3. Better vocabulary coverage

Overall, the system is functional and achieves the assignment requirements with reasonable performance on diverse question types.
