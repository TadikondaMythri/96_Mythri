### Metrics

**Classical ML Model**
- Accuracy: 72%
- Precision: 66%
- Recall: 68%
- F1-score (macro & weighted): 67%
- Confusion Matrix (full test set): [[79, 29, 13],
				    				[36, 462, 77],
 									[25, 92, 155]]

**Zero-Shot LLM (FinBERT)**
- Accuracy on balanced sample: 95%
- Confusion Matrix (class-wise performance): [[15, 0, 0],
                                             [0, 14, 1],
                                             [1, 0, 14]]

---

### Tests Performed

1. **Train–Test Split Validation**
   - Stratified split to preserve class distribution

2. **Classical ML Baseline**
   - TF-IDF features
   - Supervised classifier
   - Evaluated on full test set

3. **Zero-Shot LLM Evaluation**
   - Used pretrained FinBERT (no fine-tuning)
   - Evaluated on a balanced subset to reduce class imbalance bias

4. **Cross-Model Comparison**
   - Compared ML predictions vs LLM predictions
   - Identified agreement and disagreement cases

---

### Guardrails Applied

- Restricted sentiment labels to:
  - Positive
  - Neutral
  - Negative
- Invalid or unexpected outputs defaulted safely
- Balanced sampling used for LLM evaluation
- Classical ML and LLM evaluations kept **separate** to avoid leakage

---

### Limitations

- Dataset is heavily skewed toward Neutral sentiment
- Zero-shot LLM evaluation is performed on a subset for efficiency
- FinBERT may under-detect subtle negative sentiment
- No real-time streaming or UI included (time constraints)

---

### Expandability

- Replace FinBERT with larger financial LLMs
- Add RAG with earnings reports or SEC filings
- Deploy ML model via FastAPI endpoint
- Add temporal sentiment trend analysis

---

## Summary 

A reproducible financial news sentiment classifier combining classical ML and zero-shot LLM evaluation using a domain-specific transformer.
