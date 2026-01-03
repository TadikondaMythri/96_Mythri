#Financial News Sentiment Classifier

##Problem Statement

Classify news headlines/sentences as positive/negative/neutral for market context.

---

##Description

Financial markets are highly sensitive to news. Headlines with overly positive or negative sentiment can influence investor decisions, sometimes leading to biased or misleading market behavior.
The objective of this project is to classify financial news headlines into Positive, Neutral, or Negative sentiment, enabling better market context understanding and supporting informed decision-making.
This problem is inspired by large-scale trust and safety systems used in industry to detect sentiment bias and low-quality or manipulative content.

---

##Dataset

Name: Sentiment Analysis for Financial News
Source: Public dataset from Kaggle
Link: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

---

##Assumptions

- Correct Labeling Assumption
The dataset labels (positive, neutral, negative) are assumed to be accurate and correctly represent the sentiment of each financial news headline.

- Headline-Only Context Assumption
Sentiment classification is performed using only the news headline, without access to the full article body or external financial context.

- Static Market Context Assumption
The model does not consider real-time market conditions or stock price movements; sentiment is inferred purely from textual content.

- Balanced Interpretation Assumption
Neutral sentiment is treated as informational content without explicit positive or negative market impact.

- LLM Consistency Assumption
LLM zero-shot outputs are assumed to be reasonably consistent for identical prompts, despite inherent randomness in token generation.

---

##Solution Overview

The solution follows a two-track evaluation approach:

1️. Classical Machine Learning Model

Text preprocessing (cleaning, tokenization, vectorization)

Feature extraction using TF-IDF

Sentiment classification using Logistic Regression

Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

2️. LLM Zero-Shot Comparison

Use a Large Language Model (LLM) in zero-shot mode

Prompt the LLM to classify sentiment without task-specific training

Compare LLM predictions with classical ML outputs

---

##Tech Stack

ML: Python + scikit-learn (TF-IDF, LogisticRegression)
GenAI: HuggingFace Transformers (FinBERT) + LangChain prompts
Eval: Confusion matrix, precision/recall, error analysis

---

##System Architecture

Financial News Headline
        ↓
Text Cleaning & Preprocessing
(Tokenization, Duplicate Removal)
        ↓
Feature Extraction
(TF-IDF Vectorization)
        ↓
Classical Machine Learning Model
(Logistic Regression)
        ↓
Sentiment Prediction
(Positive / Neutral / Negative)
        ↓
Model Evaluation
(Accuracy, Precision, Recall, F1, Confusion Matrix)
        ↓
Error Analysis
(Misclassified & Ambiguous Headlines)


##Parallel LLM-Based Evaluation Path

Financial News Headline
        ↓
Prompt Engineering
(Explicit Sentiment Classification Instruction)
        ↓
Large Language Model (Zero-Shot Inference)
        ↓
LLM Sentiment Output
(Positive / Neutral / Negative)
        ↓
Comparison with Classical ML Output
(Agreement / Disagreement Analysis)

---

##Outcome

A working financial news sentiment classifier.
Comparative study between classical ML and LLM zero-shot inference.
Clear evaluation using confusion matrix and metrics.
