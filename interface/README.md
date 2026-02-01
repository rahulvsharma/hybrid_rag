# Interface Folder

The web interface for testing the RAG system.

## Files

### app.py

Main Streamlit application.

Has 4 pages:
1. Query - Ask questions and get answers
2. Evaluation Results - Browse test results
3. Analytics - View charts and metrics
4. About - System information

Features:
- Real-time Q&A
- Shows retrieved URLs and scores
- Displays response time
- Error handling
- Session history

### ui.py

UI components and styling.

Includes:
- Custom CSS styling (ChatGPT-style green theme)
- Page components
- Reusable functions
- Styling utilities

### README.md

This file.

## Running the App

```bash
streamlit run app.py
```

Access at http://localhost:8501

## Pages

Query Page:
- Text input for questions
- Real-time processing
- Shows answer, URLs, and timing
- Copy answer button

Results Page:
- Table of all 100 test questions
- Shows metrics for each
- Filter and search
- Sort by different columns

Analytics Page:
- Metric comparison charts
- Score distributions
- Question type breakdown
- Response time analysis
- Retrieval method comparison

About Page:
- How the system works
- What each metric means
- System architecture
- Key features
- Performance summary

## Configuration

Custom CSS in ui.py:
- Green theme matching ChatGPT
- Professional appearance
- Responsive design
- Works on desktop and mobile

## Dependencies

From requirements.txt:
- streamlit
- transformers
- sentence-transformers
- faiss-cpu
- rank-bm25
- plotly
- pandas

## Session Management

App maintains:
- Query history
- Recent questions
- Cached embeddings
- Session state

## Error Handling

Handles:
- Missing files
- Invalid queries
- Network issues
- Graceful degradation

## Design

Based on ChatGPT interface:
- Clean, minimal design
- Green accent color
- Easy navigation
- Fast response

## Customization

To change styling, edit ui.py:
- Colors in CSS
- Layout in components
- Page structure

## Performance

Typical response:
- First load: 30-60s (downloads models)
- Subsequent queries: 0.2-0.5s
- Charts: Instant

First time takes longer because it downloads embeddings and LLM models. After that it's fast.
