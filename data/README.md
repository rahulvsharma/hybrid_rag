# Data Folder

This folder stores all the dataset files needed for the project.

### CONV AI - Group 11 - ASSIGNMENT 2

1. Rahul Sharma - 2024AA05893 - 100%
2. Avantika Shukla - 2024AA05303 - 100%
3. Avishek Ghatak - 2024AA05895 - 100%
4. Mayank Upadhyaya - ⁠2024AA05165 - 100%
5. Trupti Dhoble - 2024AA05300 - 100%

## Files

### fixed_urls.json

200 Wikipedia URLs that stay the same.

- URLs are constant across all runs
- Ensures reproducible results
- Covers diverse topics
- Each article has at least 200 words

Format:

```json
{
  "urls": [
    "https://en.wikipedia.org/wiki/...",
    ...
  ]
}
```

### wikipedia_corpus.json

Preprocessed text from Wikipedia articles.

Structure:

- 500 total articles (200 fixed + 300 random)
- 3,452 text chunks
- Each chunk: 200-400 tokens
- Overlap between chunks: 50 tokens

Fields in each chunk:

- url: Wikipedia link
- title: Article name
- chunk_id: Unique ID
- text: Chunk content
- token_count: Size
- chunk_index: Position

Example:

```json
{
  "metadata": {
    "total_chunks": 3452,
    "total_urls": 500
  },
  "chunks": [
    {
      "url": "...",
      "title": "...",
      "chunk_id": "...",
      "text": "...",
      "token_count": 287
    }
  ]
}
```

### evaluation_results.json

Results from evaluating the system.

Contains:

- Overall metrics summary
- Per-question results
- Retrieved URLs and scores
- All 8 metrics for each question
- Response times
- Error analysis

Key metrics:

- MRR at URL: 0.42
- Hit Rate: 0.58
- NDCG@10: 0.55
- Semantic Similarity: 0.72
- Answer Faithfulness: 0.78
- ROUGE-L: 0.54
- BERTScore: 0.68
- Response Time: varies

## Using the Data

```python
import json

# Read URLs
with open('fixed_urls.json') as f:
    urls = json.load(f)

# Read corpus
with open('wikipedia_corpus.json') as f:
    corpus = json.load(f)

# Read results
with open('evaluation_results.json') as f:
    results = json.load(f)
```

## Regenerating

Rebuild entire dataset:

```bash
python code/main.py --full
```

Just rebuild corpus (keep fixed URLs):

```bash
rm wikipedia_corpus.json
python code/main.py --setup-data
```

Just run evaluation:

```bash
python code/main.py --eval
```

## File Sizes

- fixed_urls.json: 15 KB
- wikipedia_corpus.json: 30-50 MB
- evaluation_results.json: 500 KB
- Total: ~31-51 MB

## Notes

Vector indices (FAISS and BM25) are generated at startup, not stored here. They take 30-60 seconds to build the first time.

The 200 fixed URLs are always the same for consistency. The 300 random URLs change each time to test robustness.
