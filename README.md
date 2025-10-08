# Spam Email Detection Project

A comprehensive machine learning project that uses Natural Language Processing (NLP) with both **Classification** and **Clustering** methods to detect spam emails with high accuracy.

## 📊 Project Overview

This project implements a complete spam email detection pipeline including:
- Data preprocessing and cleaning
- Text preprocessing with NLTK
- Feature extraction using TF-IDF vectorization
- **Classification**: Multinomial Naive Bayes and Logistic Regression
- **Clustering**: K-Means clustering analysis
- Performance evaluation and visualization

## 🎯 Results

### Classification Results
- **Multinomial Naive Bayes**: High accuracy for spam detection
- **Logistic Regression**: Excellent precision and recall
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Detailed classification analysis

### Clustering Results (K-Means)
- **Clusters**: 2 clusters grouping similar emails
- **Visualization**: PCA-reduced plots showing cluster separation
- **Metrics**: Adjusted Rand Index and Normalized Mutual Information
- **Analysis**: Comparison of clusters with actual spam labels

## 📁 Project Structure

```
├── Spam1_2_Final.ipynb          # Main Jupyter notebook with classification & clustering
├── dataset/
│   ├── cleaned_dataset.csv      # cleaned and preprocessed dataset
│   ├── emails.csv               # Email dataset 1
│   └── spam.csv                 # Email dataset 2
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🗃️ Dataset

- **Source1: emails.csv** Basic_dataset.zip under the Assignment2 Instruction.
- **Source2: spam.csv** Kaggle.com
- **Size:** 10,864 emails (merged from 2 datasets)
- **Features:** 2 columns including text content and spam labels
- **Columns:**
  - `text`: Email content
  - `spam`: Binary label (1 = spam, 0 = not spam)

## 🛠️ Technologies Used

- **Python 3.13.3**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, wordcloud
- **NLP:** NLTK (Natural Language Toolkit)
- **Machine Learning:** scikit-learn
- **Classification:** Logistic Regression, Multinomial Naive Bayes
- **Clustering:** K-Means
- **Dimensionality Reduction:** PCA
- **Text Processing:** TF-IDF vectorization

## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Computing_Tech_Proj_Assignment
   ```

2. **Switch to the main-project branch:**
   ```bash
   git checkout main-project
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open and run the notebook:**
   - Open `Spam1_2_Final.ipynb` in Jupyter Notebook or VS Code
   - Run all cells to execute the complete pipeline

## 📈 Key Features

### Data Preprocessing
- Text normalization (lowercase conversion)
- Removal of email headers (subject, re, fw, news)
- Tokenization using NLTK
- Duplicate removal
- Stop word removal
- Punctuation removal
- Stemming using Porter Stemmer
- Missing value handling
- Feature extraction (character count, word count, sentence count)

### Text Processing
- Tokenization using NLTK
- Stop word removal
- Punctuation removal
- Stemming using Porter Stemmer

### Machine Learning Pipeline
- **Classification**: Multinomial Naive Bayes and Logistic Regression with TF-IDF vectorization
- **Clustering**: K-Means clustering with PCA visualization
- Train-test split (75-25)
- Performance evaluation with multiple metrics
- Cluster analysis and comparison with actual labels

## 📊 Visualizations

The notebook includes various visualizations:
- **Classification**: Performance metrics and confusion matrices
- **Clustering**: PCA-reduced scatter plots showing cluster separation
- **Comparison**: Side-by-side visualization of clusters vs actual labels
- **Analysis**: Statistical comparison of clustering results

## 🔍 Model Performance

### Classification (Multinomial Naive Bayes + Logistic Regression)
- **Multinomial Naive Bayes**: High accuracy for spam detection
- **Logistic Regression**: Excellent precision and recall
- **Performance Comparison**: Both algorithms evaluated with multiple metrics
- **Confusion Matrix**: Detailed analysis of classification results

### Clustering (K-Means)
- **2 clusters** grouping similar emails
- **Adjusted Rand Index** measuring cluster quality
- **Normalized Mutual Information** for cluster evaluation
- **PCA visualization** showing cluster separation

## 📝 Usage Example

```python
# Classification: Predict spam using Multinomial Naive Bayes or Logistic Regression
new_email = "Your email content here..."
processed_email = text_transform(new_email)
email_vector = tfidf.transform([processed_email])

# Using Multinomial Naive Bayes
prediction_mnb = mnb.predict(email_vector)
print(f"MultinomialNB prediction: {'SPAM' if prediction_mnb[0] == 1 else 'NOT SPAM'}")

# Using Logistic Regression
prediction_lr = clf.predict(email_vector)
print(f"Logistic Regression prediction: {'SPAM' if prediction_lr[0] == 1 else 'NOT SPAM'}")

# Clustering: Group similar emails
cluster_labels = kmeans.predict(email_vector)
print(f"Email belongs to cluster: {cluster_labels[0]}")
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the preprocessing pipeline
- Experimenting with different classification algorithms
- Adding more clustering methods (DBSCAN, Hierarchical)
- Adding regression analysis
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as part of Computing Technology Project Assignment.

---

**Note:** This project combines both **Classification** (Logistic Regression) and **Clustering** (K-Means) methods to provide comprehensive spam email detection. The notebook has been adapted for local execution - make sure all dependencies are installed before running.
