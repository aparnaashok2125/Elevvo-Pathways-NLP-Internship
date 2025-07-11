# Natural Language Processing Tasks – Elevvo Pathways Internship

Welcome to the **Natural Language Processing (NLP) Projects Repository** created as part of my internship at **Elevvo Pathways**.

## Internship Overview

As part of the **Elevvo Internship Program**, I completed a series of NLP-focused projects that demonstrate foundational to intermediate-level natural language processing techniques. Each task involved working with real-world datasets, applying essential preprocessing, implementing machine learning models, and evaluating results using standard metrics.

This repository contains **4 NLP tasks** organized across two levels — covering topics from sentiment analysis to named entity recognition.

---

## Repository Structure
├── Task_1_Sentiment_Analysis/
├── Task_2_News_Category_Classification/
├── Task_3_Fake_News_Detection/
├── Task_4_NER_News_Articles/
└── README.md

---

## Task Details

### Level 1 Tasks

---

### **Task 1: Sentiment Analysis on Product Reviews**

- **Dataset:** IMDb Reviews / Amazon Product Reviews (from Kaggle)
- **Objective:** Classify reviews as **positive** or **negative**
- **Steps:**
  - Text preprocessing (lowercasing, stopword removal)
  - Text vectorization (TF-IDF or CountVectorizer)
  - Train binary classifiers like Logistic Regression or Naive Bayes
- **Tools Used:** Python, Pandas, NLTK, Scikit-learn
- **Bonus Implemented:**
  - Accuracy comparison between Logistic Regression and Naive Bayes
  - Visualization of most frequent positive/negative words

---

### **Task 2: News Category Classification**

- **Dataset:** AG News Dataset (from Kaggle)
- **Objective:** Classify news into categories such as **Sports, Business, Politics, Technology**
- **Steps:**
  - Text preprocessing (tokenization, lemmatization)
  - Feature engineering with TF-IDF
  - Trained multiclass classifiers (e.g., Logistic Regression, SVM)
- **Tools Used:** Python, Pandas, NLTK, Scikit-learn
- **Bonus Implemented:**
  - Word cloud visualizations for each category
  - Feedforward neural network using Keras (optional)

---

### Level 2 Tasks

---

### **Task 3: Fake News Detection**

- **Dataset:** Fake and Real News Dataset (from Kaggle)
- **Objective:** Identify whether a news article is **real or fake**
- **Steps:**
  - Preprocessing (title + content)
  - TF-IDF vectorization
  - Binary classification using Logistic Regression and SVM
- **Evaluation:** Accuracy and F1-Score
- **Tools Used:** Python, Pandas, NLTK, Scikit-learn
- **Bonus Implemented:**
  - Word cloud visualization comparing real and fake news

---

### **Task 4: Named Entity Recognition (NER) from News Articles**

- **Dataset:** CoNLL-2003 (from Kaggle)
- **Objective:** Extract named entities (people, organizations, locations)
- **Steps:**
  - Applied both rule-based and model-based NER using SpaCy
  - Highlighted and categorized entities in news articles
- **Tools Used:** Python, Pandas, SpaCy
- **Bonus Implemented:**
  - Visualized entities with `displacy`
  - Compared results using two different SpaCy models

---
