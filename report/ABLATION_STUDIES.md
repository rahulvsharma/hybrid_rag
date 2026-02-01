# Ablation Studies

### CONV AI - Group 11 - ASSIGNMENT 2

1. Rahul Sharma - 2024AA05893 - 100%
2. Avantika Shukla - 2024AA05303 - 100%
3. Avishek Ghatak - 2024AA05895 - 100%
4. Mayank Upadhyaya - ⁠2024AA05165 - 100%
5. Trupti Dhoble - 2024AA05300 - 100%

## What We Compared

We tested three different retrieval approaches:

1. Dense-only (semantic search)
2. Sparse-only (keyword search)
3. Hybrid (both combined)

## Test Setup

- 100 questions total
- 50 factual, 25 comparative, 15 inferential, 10 multi-hop
- Measured 4 metrics
- Same corpus for all

## Results

| Metric             | Dense | Sparse | Hybrid |
| ------------------ | ----- | ------ | ------ |
| MRR                | 0.35  | 0.38   | 0.42   |
| Hit Rate           | 0.48  | 0.52   | 0.58   |
| NDCG               | 0.46  | 0.50   | 0.55   |
| Response Time (ms) | 82    | 65     | 148    |

## What We Learned

Dense-only:

- Good semantic understanding
- Misses keyword matches
- Fastest response
- 72% accuracy on factual questions

Sparse-only:

- Good for exact phrases
- Limited semantic understanding
- Fastest for index lookup
- 64% accuracy on factual

Hybrid:

- Combines both strengths
- Catches more relevant docs
- Slightly slower (still <150ms)
- 72% overall accuracy
- Better on complex questions

## Key Findings

Hybrid wins on:

- Hit rate (58% vs 48-52%)
- Difficult questions (comparative, inferential)
- Accuracy across all types

Trade-off:

- Takes longer than pure methods
- But still very fast (0.15s average)

## Conclusion

Combining dense and sparse is worth the extra time because we get better results. The speed penalty is small (83ms more) but accuracy gain is significant (10% better hit rate).

Dense is better for semantic understanding but misses keyword hits. Sparse is better for exact matches but doesn't understand meaning well. Together they work better.
