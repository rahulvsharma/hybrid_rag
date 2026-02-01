"""
Hybrid RAG System Implementation
Combines dense vector retrieval, sparse BM25 retrieval, and Reciprocal Rank Fusion
Optimized with caching and persistence
"""
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline
import time
import os
from functools import lru_cache
import pickle


class DenseRetriever:
    """Dense vector retrieval using sentence embeddings with caching and persistence"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = None
        self.embeddings = None
        self.index_path = 'faiss_index.bin'
        self.embeddings_path = 'embeddings_cache.pkl'
        self._embedding_cache = {}  # In-memory cache for session
    
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with LRU cache to avoid redundant encoding"""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text, convert_to_numpy=True)
        return self._embedding_cache[text]
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build FAISS index from chunks with persistence"""
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        print("Generating embeddings with caching...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype(np.float32))
        
        # Persist index to disk
        self._save_index()
        
        print(f"Built FAISS index with {len(chunks)} chunks (saved to disk)")
    
    def _save_index(self):
        """Save FAISS index to disk for faster restarts"""
        try:
            faiss.write_index(self.index, self.index_path)
            # Also save embeddings for reference
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print(f"  Index persisted to {self.index_path}")
        except Exception as e:
            print(f"  Could not persist index: {e}")
    
    def _load_index(self) -> bool:
        """Load FAISS index from disk if available"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"  Loaded persisted index from {self.index_path}")
                return True
        except Exception as e:
            print(f"  Could not load persisted index: {e}")
        return False
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve top-k chunks using dense retrieval with caching"""
        # Use cached embedding to avoid redundant encoding
        query_embedding = self._get_cached_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            results.append((chunk, similarity))
        
        return results


class SparseRetriever:
    """Sparse keyword retrieval using BM25"""
    
    def __init__(self):
        self.bm25 = None
        self.chunks = None
    
    def build_index(self, chunks: List[Dict]) -> None:
        """Build BM25 index from chunks"""
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Tokenize texts for BM25
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        print(f"Built BM25 index with {len(chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve top-k chunks using BM25"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = scores[idx]
            # Normalize score to 0-1 range
            normalized_score = min(score / 50, 1.0)  # Cap at reasonable value
            results.append((chunk, normalized_score))
        
        return results


class ReciprocalRankFusion:
    """Combine dense and sparse retrieval using RRF"""
    
    @staticmethod
    def combine_results(dense_results: List[Tuple[Dict, float]], 
                       sparse_results: List[Tuple[Dict, float]], 
                       k: int = 60, 
                       top_n: int = 10) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        # Create mapping of chunk_id to RRF scores
        rrf_scores = {}
        
        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_id = chunk['chunk_id']
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'chunk': chunk, 'score': 0, 'rank_d': rank, 'rank_s': None}
            rrf_scores[chunk_id]['score'] += 1 / (k + rank)
            rrf_scores[chunk_id]['rank_d'] = rank
        
        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_id = chunk['chunk_id']
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {'chunk': chunk, 'score': 0, 'rank_d': None, 'rank_s': rank}
            rrf_scores[chunk_id]['score'] += 1 / (k + rank)
            rrf_scores[chunk_id]['rank_s'] = rank
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Return top-n with metadata
        final_results = []
        for chunk_id, data in sorted_results[:top_n]:
            result = data['chunk'].copy()
            result['rrf_score'] = data['score']
            result['rank_dense'] = data['rank_d']
            result['rank_sparse'] = data['rank_s']
            final_results.append(result)
        
        return final_results


class ResponseGenerator:
    """Generate responses using LLM"""
    
    def __init__(self, model_name: str = 'google/flan-t5-base', device: str = 'cpu'):
        self.generator = pipeline('text2text-generation', model=model_name, device=0 if device == 'cuda' else -1)
        self.max_context_length = 512
    
    def generate(self, query: str, context_chunks: List[Dict], max_length: int = 100) -> str:
        """Generate answer based on query and context"""
        # Prepare context
        context = "\n".join([f"- {chunk['text'][:200]}" for chunk in context_chunks[:5]])
        
        # Create prompt
        prompt = f"""Context: {context}

Question: {query}

Answer concisely based on the context:"""
        
        # Generate
        try:
            result = self.generator(prompt, max_length=max_length, num_beams=2)
            return result[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class HybridRAGSystem:
    """Complete Hybrid RAG System"""
    
    def __init__(self, corpus_path: str, model_name: str = 'all-MiniLM-L6-v2', 
                 generator_model: str = 'google/flan-t5-base', device: str = 'cpu'):
        self.device = device
        
        # Load corpus
        with open(corpus_path, 'r') as f:
            corpus_data = json.load(f)
        self.chunks = corpus_data['chunks']
        
        # Initialize retrievers
        self.dense_retriever = DenseRetriever(model_name, device)
        self.sparse_retriever = SparseRetriever()
        
        # Build indices
        print("Building indices...")
        self.dense_retriever.build_index(self.chunks)
        self.sparse_retriever.build_index(self.chunks)
        
        # Initialize generator
        self.generator = ResponseGenerator(generator_model, device)
    
    def query(self, question: str, top_k: int = 10, top_n: int = 5) -> Dict[str, Any]:
        """Process a query and return results with metadata"""
        start_time = time.time()
        
        # Retrieve using both methods
        dense_results = self.dense_retriever.retrieve(question, top_k)
        sparse_results = self.sparse_retriever.retrieve(question, top_k)
        
        # Combine using RRF
        final_chunks = ReciprocalRankFusion.combine_results(dense_results, sparse_results, top_n=top_n)
        
        # Generate answer
        answer = self.generator.generate(question, final_chunks)
        
        elapsed_time = time.time() - start_time
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': final_chunks,
            'dense_results': [{'chunk_id': c[0]['chunk_id'], 'score': c[1]} for c in dense_results],
            'sparse_results': [{'chunk_id': c[0]['chunk_id'], 'score': c[1]} for c in sparse_results],
            'response_time': elapsed_time
        }


if __name__ == "__main__":
    # Example usage
    rag_system = HybridRAGSystem('wikipedia_corpus.json')
    
    # Test query
    question = "What is machine learning?"
    result = rag_system.query(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Response time: {result['response_time']:.2f}s")
    print(f"\nTop retrieved chunks:")
    for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
        print(f"{i}. {chunk['title']} (RRF Score: {chunk['rrf_score']:.4f})")
