# Report

### CONV AI - Group 11 - ASSIGNMENT 2

1. Rahul Sharma - 2024AA05893 - 100%
2. Avantika Shukla - 2024AA05303 - 100%
3. Avishek Ghatak - 2024AA05895 - 100%
4. Mayank Upadhyaya - ⁠2024AA05165 - 100%
5. Trupti Dhoble - 2024AA05300 - 100%

Contains analysis and results documentation.

## Files

### COMPLETE_EVALUATION_REPORT.md

Full evaluation results with all metrics and analysis.

Contains:

- Executive summary
- Metrics for all 100 questions
- Per-question breakdowns
- Error analysis by question type
- Performance charts
- System insights

### ABLATION_STUDIES.md

Compares different retrieval methods.

Tests:

- Dense-only retrieval
- Sparse-only (BM25) retrieval
- Hybrid (both combined)

Shows:

- Performance differences
- Trade-offs
- Why hybrid is better
- Recommendations

### ERROR_ANALYSIS.md

Detailed breakdown of failures.

Analyzes:

- Which question types fail most
- Common error patterns
- Why errors happen
- Ideas for improvement

### INNOVATIVE_APPROACHES.md

Describes innovative features of the project.

Covers:

- Hybrid retrieval combination
- Error categorization
- Response time analysis
- Performance visualization
- Custom evaluation metrics

### README.md

This file.

### Screenshots

System interface screenshots showing:

- Query page in action
- Results dashboard
- Analytics charts
- About page

## Structure

Reports are organized as:

- Summary at top
- Detailed metrics below
- Charts and visualizations
- Analysis and insights
- Conclusions

## Metrics Reported

For each question:

- Question text and type
- Ground truth URL
- Retrieved URLs and scores
- Answer generated
- All 8 metrics
- Response time
- Success/failure

Aggregated:

- Average of each metric
- Distribution by question type
- Correlation analysis
- Trends

## Usage

Reports are generated automatically by evaluation pipeline:

```bash
python evaluation_pipeline.py --num_questions 100
```

Output:

- JSON results
- CSV export
- PDF report
- HTML dashboard

## Viewing Results

Open COMPLETE_EVALUATION_REPORT.md in any markdown viewer.

View interactive dashboard at http://localhost:8501 (Analytics page).

## Analysis Files

Error analysis: ERROR_ANALYSIS.md

- Which questions fail
- Why they fail
- Patterns in errors

Ablation study: ABLATION_STUDIES.md

- Performance of each method
- Hybrid benefits
- Trade-offs

Innovations: INNOVATIVE_APPROACHES.md

- Novel techniques
- Custom metrics
- Analysis methods

## Notes

Reports include actual numbers from evaluation.

Screenshots show the actual system interface.

Everything is documented for understanding results.
