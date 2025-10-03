# Spam Email Detection Project

A comprehensive machine learning project that uses Natural Language Processing (NLP) with both **Classification** and **Clustering** methods to detect spam emails with high accuracy.

## 📊 Project Overview

This project implements a complete spam email detection pipeline including:
- Data preprocessing and cleaning
- Text preprocessing with NLTK
- Feature extraction using TF-IDF vectorization
- **Classification**: Logistic Regression
- **Clustering**: K-Means clustering analysis
- Performance evaluation and visualization

## 🎯 Results

### Classification Results (Logistic Regression)
- **Accuracy**: High performance on spam detection
- **Precision**: Excellent precision for spam classification
- **Recall**: Good recall for identifying spam emails
- **F1-Score**: Balanced precision and recall

### Clustering Results (K-Means)
- **Clusters**: 2 clusters grouping similar emails
- **Visualization**: PCA-reduced plots showing cluster separation
- **Metrics**: Adjusted Rand Index and Normalized Mutual Information
- **Analysis**: Comparison of clusters with actual spam labels

## 📁 Project Structure

```
├── SPAM_FINAL.ipynb              # Main Jupyter notebook with classification & clustering
├── dataset/
│   ├── cleaned_dataset.csv        # Preprocessed dataset
│   └── preprocessed_dataset.csv   # Final processed dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🗃️ Dataset

- **Source:** Spam email detection dataset
- **Size:** 10,598 emails
- **Features:** 6 columns including text content, spam labels, and derived features
- **Columns:**
  - `text`: Original email content
  - `spam`: Binary label (1 = spam, 0 = not spam)
  - `num_chars`: Character count
  - `num_words`: Word count
  - `num_sen10`: Sentence count
  - `preprocessed_text`: Cleaned and processed text

## 🛠️ Technologies Used

- **Python 3.13.3**
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **NLP:** NLTK (Natural Language Toolkit)
- **Machine Learning:** scikit-learn
- **Classification:** Logistic Regression
- **Clustering:** K-Means
- **Dimensionality Reduction:** PCA
- **Text Processing:** TF-IDF vectorization

## 📋 Requirements

Install the required packages:

```bash
pip install pandas matplotlib seaborn nltk scikit-learn wordcloud
```

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Computing_Tech_Proj_Assignment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open and run the notebook:**
   - Open `SPAM_FINAL.ipynb` in Jupyter Notebook or VS Code
   - Run all cells to execute the complete pipeline

## 📈 Key Features

### Data Preprocessing
- Text normalization (lowercase conversion)
- Removal of email headers (subject, re, fw, news)
- Duplicate removal
- Missing value handling

### Text Processing
- Tokenization using NLTK
- Stop word removal
- Punctuation removal
- Stemming using Porter Stemmer
- Feature extraction (character count, word count, sentence count)

### Exploratory Data Analysis
- Spam vs non-spam distribution visualization
- Statistical analysis of text features
- Correlation analysis
- Word frequency analysis
- Word cloud generation

### Machine Learning Pipeline
- **Classification**: Logistic Regression with TF-IDF vectorization
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

### Classification (Logistic Regression)
- **High accuracy** for spam detection
- **Excellent precision** for spam classification
- **Good recall** for identifying spam emails
- **Balanced F1-score** combining precision and recall

### Clustering (K-Means)
- **2 clusters** grouping similar emails
- **Adjusted Rand Index** measuring cluster quality
- **Normalized Mutual Information** for cluster evaluation
- **PCA visualization** showing cluster separation

## 📝 Usage Example

```python
# Classification: Predict spam using Logistic Regression
new_email = "Your email content here..."
processed_email = text_transform(new_email)
email_vector = tfidf.transform([processed_email])
prediction = clf.predict(email_vector)

if prediction == 1:
    print("This is SPAM")
else:
    print("This is NOT SPAM")

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