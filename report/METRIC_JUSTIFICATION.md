# METRIC JUSTIFICATION DOCUMENT

I needed to choose 7 metrics to evaluate my RAG system. The assignment required at least one mandatory metric (MRR) plus custom ones. Here's why I picked each one and how I calculated them.

## 1. Mean Reciprocal Rank (MRR) - MANDATORY METRIC

### Why I Chose This

The assignment requires this one, but honestly it's a good metric anyway. For a RAG system, if you can't retrieve the correct source document, you can't generate a good answer. MRR directly measures how well you're ranking the correct document, which is the core job of the retrieval part.

### How I Calculate It

For each question:

1. I retrieve the top-K documents
2. Find which position the correct Wikipedia URL first appears
3. Record 1/rank for that position
4. Average across all 100 questions

```
MRR = (Sum of 1/rank values) / Number of questions
```

### What the Numbers Mean

- **0.9-1.0:** Excellent - Usually finding correct doc at rank 1
- **0.5-0.9:** Good - Usually finding it in top 3
- **0.1-0.5:** Fair - Finding it somewhere in top 10 (my score is here)
- **<0.1:** Poor - Rarely finding correct source

### Why Document-Level and Not Chunk-Level?

The ground truth data I have is at the document level (which Wikipedia URLs have the answer), not at the chunk level. Plus, in real usage, you usually cite which Wikipedia article you're using, not specific chunks. So document-level makes more sense.

---

## 2. Hit Rate

### Why I Chose This

This is simple and interpretable - did I retrieve the correct source in my top 10 or not? It's different from MRR because it doesn't care about ranking position, just whether I got it. Combined with MRR, it gives me two perspectives: MRR tells me "how well did I rank things" and Hit Rate tells me "did I even retrieve it".

### How I Calculate It

```
Hit Rate = (Number of questions where correct URL appears in top-10) / (Total questions) * 100
```

Basically just counting successes out of 100.

### What It Means

- **>80%:** Good - I'm retrieving the right sources most of the time
- **60-80%:** Okay - Most questions work
- **40-60%:** Fair - Mixed results (that's where I am)
- **<40%:** Bad - System struggling

### Why This Matters

In real usage, if you can't even retrieve the right document, users won't get good answers. This metric is very practical - either the source is there or it isn't.

---

## 3. NDCG (Normalized Discounted Cumulative Gain)

### Why This Metric Matters

NDCG captures ranking quality by considering both relevance and position. More relevant documents at top positions are weighted higher, providing nuanced ranking evaluation.

### Mathematical Formulation

```
DCG@K = Σ(rel_i / log2(i+1)) for i=1 to K
IDCG@K = DCG of ideal ranking
NDCG@K = DCG@K / IDCG@K
```

### Calculation Method

1. For each query, score documents by relevance (binary: relevant=1, irrelevant=0)
2. Calculate DCG: sum of (relevance / log2(position))
3. Calculate IDCG: ideal DCG with all relevant docs at top
4. Normalize: DCG / IDCG
5. Average across all queries

### Interpretation Guidelines

- **0.9-1.0**: Perfect ranking - All relevant docs at top
- **0.7-0.9**: Good ranking - Relevant docs highly ranked
- **0.5-0.7**: Fair ranking - Some relevant docs ranked well
- **<0.5**: Poor ranking - Relevant docs scattered in results

### Why This Metric?

NDCG is crucial because:

1. Accounts for ranking quality, not just presence/absence
2. Position-aware: higher positions weighted more
3. Standard in IR evaluation literature
4. Better than simple precision for understanding ranking

---

## 4. BERTScore

### Why This Metric Matters

BERTScore evaluates answer quality by measuring semantic similarity between generated and reference answers. It captures conceptual correctness beyond exact text matching.

### Mathematical Formulation

```
Precision = (1/|S_generated|) * Σ_g max_r (cos(g, r))
Recall = (1/|S_reference|) * Σ_r max_g (cos(g, r))
F1 = (2 * Precision * Recall) / (Precision + Recall)

where:
- S_generated/reference = token embeddings from BERT
- cos(g, r) = cosine similarity between embeddings
```

### Calculation Method

1. Tokenize generated and reference answers
2. Get BERT embeddings for each token
3. Compute token-level cosine similarities
4. For precision: best match for each generated token
5. For recall: best match for each reference token
6. Calculate F1 score

### Interpretation Guidelines

- **0.8-1.0**: Excellent - Generated answer aligns with reference
- **0.6-0.8**: Good - Mostly semantically similar
- **0.4-0.6**: Fair - Some semantic overlap
- **<0.4**: Poor - Little semantic similarity

### Why This Metric?

BERTScore is valuable because:

1. Captures semantic meaning, not surface text
2. Robust to paraphrasing and synonyms
3. Doesn't require reference answer
4. Better than BLEU for natural language evaluation

---

## 5. Semantic Similarity (Cosine Similarity of Sentence Embeddings)

### Why This Metric Matters

Semantic Similarity directly measures how well the generated answer aligns with the retrieved context and question. It ensures the model generates answers grounded in the retrieved information.

### Mathematical Formulation

```
Semantic_Similarity = cos(embedding_generated, embedding_reference)
= (generated · reference) / (||generated|| * ||reference||)

where:
- embeddings are from all-MiniLM-L6-v2 model
- · represents dot product
- || || represents L2 norm
```

### Calculation Method

1. Embed generated answer using sentence transformer
2. Embed reference answer (from context)
3. Compute cosine similarity: -1 to 1 (typically 0 to 1)
4. Average across all questions

### Interpretation Guidelines

- **0.9-1.0**: Highly similar - Perfect semantic alignment
- **0.7-0.9**: Very similar - Strong semantic overlap
- **0.5-0.7**: Similar - Meaningful semantic connection
- **<0.5**: Different - Low semantic similarity

### Why This Metric?

Semantic Similarity is essential because:

1. Direct measure of answer relevance to context
2. Captures meaning, not lexical overlap
3. Fast computation compared to LLM-based metrics
4. Complements other metrics for comprehensive evaluation

---

## 6. Contextual Precision

### Why This Metric Matters

Contextual Precision measures what percentage of retrieved context is actually relevant to answering the question. High precision ensures most retrieved information is useful.

### Mathematical Formulation

```
Contextual_Precision = (# relevant chunks in top-K) / K
```

### Calculation Method

1. For each question, retrieve top-K chunks
2. Manually/automatically mark chunks as relevant
3. Count relevant chunks
4. Divide by K (total retrieved)
5. Average across questions

### Interpretation Guidelines

- **>0.8**: Excellent - Most retrieved info is useful
- **0.6-0.8**: Good - Majority is relevant
- **0.4-0.6**: Fair - About half is relevant
- **<0.4**: Poor - Low useful information density

### Why This Metric?

Contextual Precision matters because:

1. Measures signal-to-noise ratio in retrieved context
2. High precision = less irrelevant information
3. Important for generation quality
4. Often more important than recall for LLMs

---

## 7. Answer Faithfulness

### Why This Metric Matters

Answer Faithfulness ensures generated answers are grounded in retrieved context and don't contain hallucinations. This is critical for trustworthy RAG systems.

### Measurement Approach

1. Use LLM-as-Judge to evaluate factual consistency
2. Check if answer statements are supported by context
3. Detect hallucinated information not in context
4. Score: % of answer claims supported by context

### Interpretation Guidelines

- **0.9-1.0**: Highly faithful - All claims grounded in context
- **0.7-0.9**: Mostly faithful - Most claims supported
- **0.5-0.7**: Partially faithful - Some claims hallucinated
- **<0.5**: Unfaithful - Many hallucinated statements

### Why This Metric?

Answer Faithfulness is crucial because:

1. Hallucination is major LLM failure mode
2. RAG systems should ground answers in retrieved docs
3. Prevents spread of false information
4. Essential for trustworthy AI systems

---

## Metric Selection Rationale

### Why These 7 Metrics?

1. **Coverage**: Retrieval (MRR, Hit Rate, NDCG, Precision), Answer Quality (BERTScore, Semantic Sim), Faithfulness (Answer Faithfulness)
2. **Diversity**: Different aspects of system performance
3. **Interpretability**: All metrics have clear meaning
4. **Practicality**: Reasonable computation time
5. **Literature**: Align with RAG evaluation best practices

### Metrics NOT Included & Why

- **BLEU**: Too lexical, poor for semantic evaluation
- **ROUGE-L**: Good but overlaps with BERTScore
- **Exact Match (EM)**: Too strict, fails on paraphrases
- **Recall**: Less important than precision for LLM context

---

## Performance Targets

Based on literature and best practices:

| Metric               | Target | Reason                               |
| -------------------- | ------ | ------------------------------------ |
| MRR                  | >0.6   | Should identify source quickly       |
| Hit Rate             | >75%   | Most questions should succeed        |
| NDCG@10              | >0.7   | Ranking should be good               |
| BERTScore F1         | >0.65  | Semantic match should be strong      |
| Semantic Sim         | >0.7   | Answer should align with context     |
| Contextual Precision | >0.7   | Retrieved context should be relevant |
| Answer Faithfulness  | >0.8   | Answers should be grounded           |

---

## Implementation Notes

All metrics are computed in `evaluation/evaluation.py`:

- **Dense Retrieval**: FAISS with all-MiniLM-L6-v2
- **Sparse Retrieval**: BM25Okapi ranking
- **Fusion**: Reciprocal Rank Fusion (k=60)
- **LLM**: Flan-T5-base for generation
- **Metric Libraries**: transformers (BERTScore), sklearn (cosine similarity)

---

**Document Version**: 1.0  
**Last Updated**: 1 February 2026  
**Metrics Reviewed**: 7/7
