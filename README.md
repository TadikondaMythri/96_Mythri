# Financial News Sentiment Classifier

## Problem Statement

Classify financial news headlines or sentences as **Positive**, **Neutral**, or **Negative** to provide market context.

---

## Description

Financial markets are highly sensitive to news. Headlines with overly positive or negative sentiment can influence investor decisions, sometimes leading to biased or misleading market behavior.

The objective of this project is to classify financial news headlines into **Positive**, **Neutral**, or **Negative** sentiment, enabling better market context understanding and supporting informed decision-making.

This problem is inspired by large-scale **trust and safety systems** used in industry to detect sentiment bias and low-quality or manipulative content.

---

## Dataset

- **Name:** Sentiment Analysis for Financial News  
- **Source:** Public dataset from Kaggle  
- **Link:** https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news  

---

## Assumptions

- **Correct Labeling Assumption**  
  The dataset labels (positive, neutral, negative) are assumed to be accurate and correctly represent the sentiment of each financial news headline.

- **Headline-Only Context Assumption**  
  Sentiment classification is performed using only the news headline, without access to the full article body or external financial context.

- **Static Market Context Assumption**  
  The model does not consider real-time market conditions or stock price movements; sentiment is inferred purely from textual content.

- **Balanced Interpretation Assumption**  
  Neutral sentiment is treated as informational content without explicit positive or negative market impact.

- **LLM Consistency**
  LLM zero-shot outputs are assumed to be reasonably consistent for identical prompts, despite inherent randomness in token generation.

---

## Solution Overview

The solution follows a **two-track evaluation approach**:

### 1. Classical Machine Learning Model
- Text preprocessing (cleaning, tokenization, vectorization)
- Feature extraction using **TF-IDF**
- Sentiment classification using **Logistic Regression**
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

### 2. LLM Zero-Shot Comparison
- Use a Large Language Model (LLM) in **zero-shot mode**
- Prompt the LLM to classify sentiment without task-specific training
- Compare LLM predictions with classical ML outputs

---

## Tech Stack

- **Machine Learning:** Python, scikit-learn (TF-IDF, Logistic Regression)
- **Generative AI:** HuggingFace Transformers (FinBERT)
- **Evaluation:** Confusion Matrix, Precision, Recall, F1-score, Error Analysis

---

## System Architecture

| Step | Component | Description |
|-----|----------|-------------|
| 1 | Financial News Headline | Input financial news text |
| 2 | Text Cleaning & Preprocessing | Tokenization, lowercasing, stopword removal |
| 3 | Feature Extraction | TF-IDF vectorization |
| 4 | ML Model | Logistic Regression classifier |
| 5 | Sentiment Prediction | Positive / Neutral / Negative |
| 6 | Model Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 7 | Error Analysis | Misclassified and ambiguous headlines |



## Parallel LLM-Based Evaluation Path

| Step | Component | Description |
|-----:|----------|-------------|
| 1 | Financial News Headline | Input financial news headline used for LLM evaluation |
| 2 | Prompt Engineering | Designing effective prompts to guide the LLM for sentiment analysis |
| 3 | LLM Zero-Shot Inference | LLM predicts sentiment without task-specific training |
| 4 | LLM Sentiment Output | Sentiment predicted as Positive, Neutral, or Negative |
| 5 | ML vs LLM Comparison | Comparison between classical ML and LLM sentiment predictions |

---

## Outcome

- A working financial news sentiment classifier.
- Comparative study between classical ML and LLM zero-shot inference.
- Clear evaluation using confusion matrix and metrics.
