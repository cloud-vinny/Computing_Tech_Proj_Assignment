# Spam Email Detection Project

A machine learning project that uses Natural Language Processing (NLP) and various Naive Bayes classifiers to detect spam emails with high accuracy.

## 📊 Project Overview

This project implements a complete spam email detection pipeline including:
- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- Text preprocessing with NLTK
- Feature extraction using CountVectorizer
- Machine learning model training and evaluation
- Performance comparison of different Naive Bayes algorithms

## 🎯 Results

The project achieves excellent performance across different algorithms:

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| **Bernoulli Naive Bayes** | **98.9%** | **98.6%** |
| Gaussian Naive Bayes | 94.9% | 95.0% |
| Multinomial Naive Bayes | 88.1% | 98.7% |

**Best Model:** Bernoulli Naive Bayes with 98.9% accuracy

## 📁 Project Structure

```
├── Spam1.ipynb                    # Main Jupyter notebook
├── dataset/
│   ├── cleaned_dataset.csv        # Preprocessed dataset
│   └── preprocessed_dataset.csv   # Final processed dataset
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
- **Text Processing:** wordcloud

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
   - Open `Spam1.ipynb` in Jupyter Notebook or VS Code
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
- Text vectorization using CountVectorizer
- Train-test split (80-20)
- Multiple algorithm comparison
- Performance evaluation with accuracy, precision, and confusion matrix

## 📊 Visualizations

The notebook includes various visualizations:
- Pie chart showing spam distribution
- Histograms comparing text features between spam and non-spam
- Correlation heatmap
- Word clouds for spam and non-spam emails
- Bar charts showing most frequent words

## 🔍 Model Performance

The Bernoulli Naive Bayes model shows the best performance:
- **Accuracy:** 98.9%
- **Precision:** 98.6%
- **Confusion Matrix:**
  ```
  [[845   4]
   [  8 282]]
  ```

## 📝 Usage Example

```python
# Load the trained model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# Predict on new email
new_email = "Your email content here..."
processed_email = text_transform(new_email)
email_vector = cv.transform([processed_email])
prediction = bnb.predict(email_vector)

if prediction == 1:
    print("This is SPAM")
else:
    print("This is NOT SPAM")
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the preprocessing pipeline
- Experimenting with different algorithms
- Adding new features
- Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as part of Computing Technology Project Assignment.

---

**Note:** This project was originally developed in Google Colab and has been adapted for local execution. Make sure all dependencies are installed before running the notebook.