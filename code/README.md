# Code Folder

Contains the main Python code for the RAG system.

## Files

### data_collection.py

Gets Wikipedia articles and prepares them for use.

Functions:
- create_fixed_urls_file(): Creates list of 200 fixed URLs
- DataPreprocessor: Class that collects and chunks Wikipedia articles
- WikipediaDataCollector: Gets raw Wikipedia content

Does:
- Downloads Wikipedia articles
- Splits into chunks (200-400 tokens)
- Adds metadata (title, URL, chunk ID)
- Saves to JSON

### rag_system.py

The main RAG implementation.

Classes:
- HybridRAGSystem: Main class for queries
- DenseRetriever: Uses embeddings for search
- SparseRetriever: Uses BM25 keyword search
- ResponseGenerator: Creates answers with LLM

Does:
- Loads corpus
- Builds indices for search
- Processes queries
- Retrieves documents
- Generates answers
- Tracks timing

### main.py

Main entry point script.

Options:
```bash
python code/main.py --full              # Everything
python code/main.py --setup-data        # Get data only
python code/main.py --test-rag          # Test RAG
python code/main.py --eval              # Run evaluation
python code/main.py --ablation          # Ablation study
```

## Usage

### Setup Data

```python
from code.data_collection import create_fixed_urls_file, DataPreprocessor

urls = create_fixed_urls_file('data/fixed_urls.json', 200)

with open('data/fixed_urls.json') as f:
    urls = json.load(f)['urls']

prep = DataPreprocessor()
corpus = prep.collect_wikipedia_corpus(urls, random_count=300)
prep.save_corpus(corpus, 'data/wikipedia_corpus.json')
```

### Query RAG

```python
from code.rag_system import HybridRAGSystem

rag = HybridRAGSystem('data/wikipedia_corpus.json')
result = rag.query("What is AI?")

print(result['answer'])
print(result['response_time'])
```

### Run Main Script

```bash
python code/main.py --full
```

This does:
1. Setup data
2. Test RAG system
3. Run evaluation
4. Run ablation study

## Models Used

- Dense embeddings: all-MiniLM-L6-v2
- Sparse search: BM25
- Language model: google/flan-t5-base

## Key Classes

HybridRAGSystem:
- query(question, top_k, top_n)
- Retrieves relevant chunks
- Generates answer
- Returns results with metrics

## Output

Returns dictionary with:
- question: Input question
- answer: Generated answer
- retrieved_chunks: List of chunks found
- response_time: Seconds taken
- metrics: Retrieval scores
