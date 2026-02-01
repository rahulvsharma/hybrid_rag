"""
Streamlit UI for Hybrid RAG System
"""
import streamlit as st
import json
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from rag_system import HybridRAGSystem
from evaluation import QuestionGenerator, EvaluationPipeline


@st.set_page_config(page_title="Hybrid RAG System", layout="wide")

def load_rag_system():
    """Load or create RAG system"""
    if 'rag_system' not in st.session_state:
        try:
            corpus_path = Path('wikipedia_corpus.json')
            if corpus_path.exists():
                st.session_state.rag_system = HybridRAGSystem(str(corpus_path))
            else:
                st.error("Corpus not found. Please run data collection first.")
                return None
        except Exception as e:
            st.error(f"Error loading RAG system: {str(e)}")
            return None
    
    return st.session_state.rag_system


def main():
    st.title("🤖 Hybrid RAG System")
    st.markdown("**Combining Dense Retrieval, BM25, and Reciprocal Rank Fusion**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["Query Interface", "Evaluation", "Results Analysis", "About"])
    
    if page == "Query Interface":
        query_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Results Analysis":
        analysis_page()
    elif page == "About":
        about_page()


def query_page():
    """Query interface page"""
    st.header("🔍 Query Interface")
    
    rag_system = load_rag_system()
    if rag_system is None:
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your question:", placeholder="What is artificial intelligence?")
    
    with col2:
        top_k = st.slider("Top-K (retrieval)", 5, 20, 10)
        top_n = st.slider("Top-N (final)", 3, 10, 5)
    
    if st.button("Search", use_container_width=True):
        if query:
            with st.spinner("Processing query..."):
                start_time = time.time()
                result = rag_system.query(query, top_k=top_k, top_n=top_n)
                elapsed_time = time.time() - start_time
            
            # Display results
            st.success("Query processed successfully!")
            
            st.markdown("### Generated Answer")
            st.info(result['answer'])
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{elapsed_time:.2f}s")
            with col2:
                st.metric("Retrieved Chunks", len(result['retrieved_chunks']))
            with col3:
                st.metric("Top RRF Score", f"{result['retrieved_chunks'][0]['rrf_score']:.4f}" if result['retrieved_chunks'] else "N/A")
            
            st.markdown("### Retrieved Chunks (by RRF Score)")
            
            # Display retrieved chunks
            chunk_data = []
            for chunk in result['retrieved_chunks']:
                chunk_data.append({
                    'Rank': len(chunk_data) + 1,
                    'Title': chunk['title'],
                    'URL': chunk['url'],
                    'RRF Score': f"{chunk['rrf_score']:.4f}",
                    'Dense Rank': chunk['rank_dense'] or '-',
                    'Sparse Rank': chunk['rank_sparse'] or '-',
                    'Preview': chunk['text'][:150] + "..."
                })
            
            df_chunks = pd.DataFrame(chunk_data)
            st.dataframe(df_chunks, use_container_width=True)
            
            st.markdown("### Score Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dense vs Sparse vs RRF scores
                if result['dense_results'] and result['sparse_results']:
                    scores_data = {
                        'Method': (['Dense'] * len(result['dense_results']) + 
                                 ['Sparse'] * len(result['sparse_results'])),
                        'Score': ([r['score'] for r in result['dense_results']] + 
                                [r['score'] for r in result['sparse_results']])
                    }
                    
                    fig = px.box(pd.DataFrame(scores_data), x='Method', y='Score', 
                               title='Score Distribution by Retrieval Method')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RRF scores
                rrf_scores = [c['rrf_score'] for c in result['retrieved_chunks']]
                fig = go.Figure()
                fig.add_trace(go.Bar(y=rrf_scores, name='RRF Score'))
                fig.update_layout(title='RRF Scores by Chunk', xaxis_title='Chunk Rank', yaxis_title='Score')
                st.plotly_chart(fig, use_container_width=True)


def evaluation_page():
    """Evaluation page"""
    st.header("📊 Evaluation")
    
    rag_system = load_rag_system()
    if rag_system is None:
        return
    
    st.markdown("""
    ### Evaluation Configuration
    Generate and evaluate questions on the RAG system.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_questions = st.slider("Number of Questions", 10, 100, 50, step=10)
    
    with col2:
        evaluation_type = st.selectbox("Evaluation Focus", 
                                      ["Comprehensive", "Retrieval Only", "Generation Only"])
    
    if st.button("Run Evaluation", use_container_width=True):
        # Load corpus
        try:
            with open('wikipedia_corpus.json', 'r') as f:
                corpus = json.load(f)
            chunks = corpus['chunks']
        except:
            st.error("Corpus not found")
            return
        
        with st.spinner("Generating questions..."):
            generator = QuestionGenerator()
            questions = generator.generate_questions(chunks, num_questions)
        
        with st.spinner("Running evaluations..."):
            pipeline = EvaluationPipeline()
            rag_results = []
            progress_bar = st.progress(0)
            
            for i, question in enumerate(questions):
                result = rag_system.query(question['question'], top_k=10, top_n=5)
                rag_results.append(result)
                progress_bar.progress((i + 1) / len(questions))
            
            evaluation_results = pipeline.evaluate_batch(questions, rag_results)
        
        # Display results
        st.success("Evaluation completed!")
        
        summary = evaluation_results['summary']
        
        st.markdown("### Summary Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg MRR@URL", f"{summary['avg_mrr_url']:.4f}")
        with col2:
            st.metric("Avg Hit Rate", f"{summary['avg_hit_rate']:.4f}")
        with col3:
            st.metric("Avg NDCG@10", f"{summary['avg_ndcg_at_10']:.4f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Contextual Precision", f"{summary['avg_contextual_precision']:.4f}")
        with col2:
            st.metric("Semantic Similarity", f"{summary['avg_semantic_similarity']:.4f}")
        with col3:
            st.metric("Answer Faithfulness", f"{summary['avg_answer_faithfulness']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Response Time", f"{summary['avg_response_time']:.2f}s")
        with col2:
            st.metric("Avg ROUGE-L F1", f"{summary['avg_rouge_l_fmeasure']:.4f}")
        
        # Detailed results
        st.markdown("### Detailed Results")
        
        results_df = pd.DataFrame(evaluation_results['detailed_results'])
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )
        
        # Save to JSON
        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        st.info("  Results saved to evaluation_results.json")


def analysis_page():
    """Results analysis page"""
    st.header("📈 Results Analysis")
    
    # Try to load previous evaluation results
    try:
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
    except:
        st.info("No evaluation results found. Run evaluation first.")
        return
    
    summary = results['summary']
    detailed = results['detailed_results']
    
    # Metrics comparison
    st.markdown("### Metrics Comparison")
    
    metrics_data = {
        'Metric': ['MRR@URL', 'Hit Rate', 'NDCG@10', 'Contextual Precision', 'Semantic Similarity', 'Answer Faithfulness'],
        'Score': [
            summary['avg_mrr_url'],
            summary['avg_hit_rate'],
            summary['avg_ndcg_at_10'],
            summary['avg_contextual_precision'],
            summary['avg_semantic_similarity'],
            summary['avg_answer_faithfulness']
        ]
    }
    
    fig = px.bar(pd.DataFrame(metrics_data), x='Metric', y='Score', 
                title='Evaluation Metrics Summary')
    st.plotly_chart(fig, use_container_width=True)
    
    # Response time analysis
    st.markdown("### Response Time Analysis")
    
    response_times = [r['response_time'] for r in detailed]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=response_times, nbinsx=20, name='Response Time'))
    fig.update_layout(title='Response Time Distribution', xaxis_title='Time (seconds)', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)
    
    # Error analysis
    st.markdown("### Performance by Question Type")
    
    question_types = {}
    for result in detailed:
        # Group by performance (to be enhanced)
        pass


def about_page():
    """About page"""
    st.header("About Hybrid RAG System")
    
    st.markdown("""
    ## System Overview
    
    This is a Hybrid Retrieval-Augmented Generation (RAG) system that combines:
    
    1. **Dense Vector Retrieval**: Using sentence embeddings (all-MiniLM-L6-v2)
    2. **Sparse Keyword Retrieval**: Using BM25 algorithm
    3. **Reciprocal Rank Fusion**: Combining both retrieval methods
    4. **Response Generation**: Using fine-tuned T5 models
    
    ## Key Features
    
    - 🔍 Hybrid retrieval combining dense and sparse methods
    - 🎯 Reciprocal Rank Fusion (RRF) for result combination
    - 📊 Comprehensive evaluation metrics
    - 🎨 Interactive Streamlit UI
    - 📈 Advanced analysis and visualization
    
    ## Dataset
    
    - 200 fixed Wikipedia URLs (consistent across runs)
    - 300 random Wikipedia URLs (regenerated per run)
    - Text chunking: 200-400 tokens with 50-token overlap
    
    ## Evaluation Metrics
    
    ### Mandatory
    - **MRR@URL**: Mean Reciprocal Rank at URL level
    
    ### Custom
    - **Hit Rate**: Fraction of relevant documents in top-K
    - **NDCG@10**: Normalized Discounted Cumulative Gain
    - **Contextual Precision**: Precision of retrieval
    - **Semantic Similarity**: Answer-context semantic matching
    - **Answer Faithfulness**: Grounding in retrieved context
    - **ROUGE-L**: Lexical overlap with context
    - **BERTScore**: Semantic similarity using BERT
    
    ### Innovative
    - Ablation studies (dense-only, sparse-only, hybrid)
    - Error analysis by question type
    - Confidence calibration
    - Response time analysis
    
    ## Installation
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## Usage
    
    ```bash
    streamlit run ui.py
    ```
    
    """)
    
    st.markdown("### System Architecture")
    
    with st.expander("View Architecture Diagram"):
        st.image("https://via.placeholder.com/800x400?text=RAG+Architecture", 
                caption="Hybrid RAG System Architecture")


if __name__ == "__main__":
    main()
