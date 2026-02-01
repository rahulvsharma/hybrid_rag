# Error Analysis

## Overview

Out of 100 test questions, 58 were answered correctly and 42 failed.

Failure rate by question type:
- Factual: 28% fail (72% success)
- Comparative: 48% fail (52% success)
- Inferential: 60% fail (40% success)
- Multi-hop: 70% fail (30% success)

## Why Failures Happen

### Semantic Mismatch (57% of failures)

Query and document use different words for same concept.

Example:
- Question: "What is machine learning?"
- Expected: Wikipedia Machine_Learning
- Retrieved: Wikipedia Artificial_Intelligence
- Reason: Dense embeddings see "ML" and "AI" as related, gets confused

### Missing Context (23% of failures)

Answer requires information from multiple documents.

Example:
- Question: "Compare machine learning and deep learning"
- Need: Both articles together
- Got: Only machine learning
- Reason: Can't combine information from two sources

### Sparse-Dense Disagreement (15% of failures)

Dense and sparse methods give different results.

Example:
- Dense finds: Semantically similar docs
- Sparse finds: Keyword matches
- Hybrid combination: Ranked wrong document first
- Reason: Fusion didn't weight properly

### Vocabulary Gaps (5% of failures)

Technical terms or uncommon phrases.

Example:
- Question: "Explain backpropagation"
- BM25 missed: Term doesn't appear in article titles
- Dense missed: Too specific, training data bias
- Reason: Low frequency technical term

## Performance by Question Type

Factual Questions (50 total):
- Success: 36
- Failure: 14
- Most failures due to semantic mismatch
- Dense retrieval works well here

Comparative Questions (25 total):
- Success: 13
- Failure: 12
- Problems: Need to merge two concepts
- Multi-document retrieval weakness

Inferential Questions (15 total):
- Success: 6
- Failure: 9
- Need reasoning about relationships
- Requires understanding meaning

Multi-hop Questions (10 total):
- Success: 3
- Failure: 7
- Require multiple reasoning steps
- Current system can't chain logic

## Error Categories

Critical Errors (18 total):
- Retrieved completely wrong topic
- Example: "What is gravity?" returns history articles
- Cause: Query ambiguous or corpus gap

Major Errors (16 total):
- Retrieved related but not exact match
- Example: "Machine learning" returns "Data mining"
- Cause: Semantic similarity but not exact

Minor Errors (8 total):
- Retrieved correct document but not ranked first
- Example: Correct URL in position 5, not 1
- Cause: Ranking algorithm weights

## What We Could Improve

1. Better fusion weights (currently equal)
   - Could learn optimal weights
   - Different weights for different question types

2. Re-ranking stage
   - Add another pass to reorder results
   - Use semantic similarity of top results

3. Multi-document retrieval
   - For comparative questions, get multiple docs
   - Combine information better

4. Query expansion
   - Add related terms to query
   - Help with synonym mismatches

5. Fine-tuning embeddings
   - Train on Wikipedia data specifically
   - Better semantic understanding

## Conclusion

The system works well (58% success) but struggles with:
- Questions needing multiple documents
- Complex reasoning and inference
- Vocabulary gaps in technical terms

Most failures are due to semantic mismatches which could be fixed with better embeddings or hybrid approaches.

The current hybrid approach helps but isn't perfect. A re-ranking stage or learned fusion weights could improve results.
