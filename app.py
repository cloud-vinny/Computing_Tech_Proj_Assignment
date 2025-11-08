from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import uvicorn
from nltk import word_tokenize
from typing import Dict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Startup event using lifespan (modern FastAPI approach)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Spam Detection API...")
    try:
        load_and_train_models()
        print("API startup completed successfully!")
    except Exception as e:
        print(f"Startup failed: {e}")
        # Don't crash the app, just log the error
        print("API will start without pre-trained models")
    yield
    # Shutdown (if needed)
    print("Shutting down Spam Detection API...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Spam Detection API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize stemmer
ps = PorterStemmer()
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

# Global variables for models
tfidf = None
clf = None
mnb = None
kmeans = None
kmeans_label_map: Dict[int, int] = {}

# Pydantic models for request/response
class SpamRequest(BaseModel):
    text: str
    model: str = "logistic"  # Default to logistic regression

class SpamResponse(BaseModel):
    is_spam: bool
    confidence: float
    model_used: str
    message: str

# Text preprocessing function (same as in your notebook)
def text_transform(text):
    text = text.lower()  # lowercase
    tokens = word_tokenize(text)  # tokenize

    filtered_tokens = [
        ps.stem(token)
        for token in tokens
        if token.isalnum() and token not in STOP_WORDS and token not in PUNCTUATION
    ]

    return " ".join(filtered_tokens)

# Load and train models
def load_and_train_models():
    global tfidf, clf, mnb, kmeans, kmeans_label_map
    
    try:
        print("Loading dataset...")
        # Load dataset with smaller sample for faster startup
        df = pd.read_csv("dataset/cleaned_dataset_small.csv")
        df.dropna(subset=['preprocessed_text'], inplace=True)
        
        # Use only first 2000 rows for faster training
        df = df.head(2000)
        print(f"Using {len(df)} samples for training")
        
        # Prepare data
        ros = RandomOverSampler(random_state=2)
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))  # Reduced features
        X = tfidf.fit_transform(df['preprocessed_text']).toarray()
        y = df['spam'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
        
        # Train models with reduced complexity
        print("Training Logistic Regression...")
        clf = LogisticRegression(max_iter=100, class_weight='balanced', n_jobs=1)  # Reduced iterations
        clf.fit(X_train_bal, y_train_bal)
        
        print("Training Naive Bayes...")
        mnb = MultinomialNB(alpha=0.1)
        mnb.fit(X_train_bal, y_train_bal)
        
        print("Training K-Means...")
        kmeans = KMeans(n_clusters=2, random_state=2, n_init=10)  # Reduced n_init
        kmeans.fit(X)

        # Build a cluster -> class map so we can interpret predictions deterministically
        cluster_assignments = kmeans.predict(X)
        kmeans_label_map = {}
        for cluster_id in range(kmeans.n_clusters):
            indices = np.where(cluster_assignments == cluster_id)[0]
            if len(indices) == 0:
                kmeans_label_map[cluster_id] = 0
                continue

            spam_votes = int(y[indices].sum())
            ham_votes = len(indices) - spam_votes
            kmeans_label_map[cluster_id] = 1 if spam_votes >= ham_votes else 0
        
        print("All models trained successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Spam Detection API is running!", "status": "healthy"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "tfidf": tfidf is not None,
            "logistic_regression": clf is not None,
            "naive_bayes": mnb is not None,
            "kmeans": kmeans is not None
        }
    }

# Spam detection endpoint
@app.post("/detect", response_model=SpamResponse)
async def detect_spam(request: SpamRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        model_key = request.model.lower()
        if tfidf is None:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please try again soon.")
        if model_key == "logistic" and clf is None:
            raise HTTPException(status_code=503, detail="Logistic Regression model is unavailable. Please try again later.")
        if model_key == "naive_bayes" and mnb is None:
            raise HTTPException(status_code=503, detail="Naive Bayes model is unavailable. Please try again later.")
        if model_key == "kmeans" and kmeans is None:
            raise HTTPException(status_code=503, detail="K-Means model is unavailable. Please try again later.")

        # Preprocess text
        cleaned_text = text_transform(request.text)
        text_vector_sparse = tfidf.transform([cleaned_text])
        text_vector_dense = text_vector_sparse.toarray()
        
        # Select model and make prediction
        if model_key == "logistic":
            prediction = clf.predict(text_vector_dense)[0]
            confidence = clf.predict_proba(text_vector_dense)[0][1]  # Probability of being spam
            model_used = "Logistic Regression"
            is_spam = bool(prediction)
        elif model_key == "naive_bayes":
            prediction = mnb.predict(text_vector_dense)[0]
            confidence = mnb.predict_proba(text_vector_dense)[0][1]  # Probability of being spam
            model_used = "Naive Bayes"
            is_spam = bool(prediction)
        elif model_key == "kmeans":
            cluster_id = int(kmeans.predict(text_vector_dense)[0])
            mapped_label = kmeans_label_map.get(cluster_id, 0)
            is_spam = bool(mapped_label)
            # For K-Means, we'll use distance to cluster centers as confidence
            distances = kmeans.transform(text_vector_dense)[0]
            max_distance = float(distances.max())
            min_distance = float(distances.min())
            confidence = 1.0 if max_distance == 0 else 1 - (min_distance / max_distance)
            confidence = max(0.0, min(confidence, 1.0))
            model_used = "K-Means"
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Use 'logistic', 'naive_bayes', or 'kmeans'")
        
        # Determine result message
        if is_spam:
            message = "This message appears to be spam."
        else:
            message = "This message appears to be legitimate."
        
        return SpamResponse(
            is_spam=is_spam,
            confidence=float(confidence),
            model_used=model_used,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Get available models
@app.get("/models")
async def get_models():
    return {
        "available_models": [
            {"name": "logistic", "description": "Logistic Regression - Best overall performance"},
            {"name": "naive_bayes", "description": "Naive Bayes - Good for text classification"},
            {"name": "kmeans", "description": "K-Means Clustering - Unsupervised learning"}
        ]
    }

# Dataset analysis endpoints
def analyze_dataset():
    """Load and analyze dataset for visualization"""
    # Try to load the dataset (use small for faster loading)
    dataset_path = "dataset/cleaned_dataset_small.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/cleaned_dataset.csv"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file not found")
    
    df = pd.read_csv(dataset_path)
    df.dropna(subset=['preprocessed_text'], inplace=True)
    
    return df

@app.get("/dataset/stats")
async def get_dataset_stats():
    """Get dataset statistics for visualization"""
    try:
        df = analyze_dataset()
        
        # Calculate statistics
        total_samples = len(df)
        spam_count = int(df['spam'].sum())
        ham_count = total_samples - spam_count
        spam_percentage = (spam_count / total_samples * 100) if total_samples > 0 else 0
        ham_percentage = (ham_count / total_samples * 100) if total_samples > 0 else 0
        balance_ratio = (spam_count / ham_count) if ham_count > 0 else 0
        
        # Text length statistics
        text_lengths = df['preprocessed_text'].str.len()
        avg_text_length = float(text_lengths.mean())
        min_text_length = int(text_lengths.min())
        max_text_length = int(text_lengths.max())
        
        # Word count statistics
        word_counts = df['preprocessed_text'].str.split().str.len()
        avg_word_count = float(word_counts.mean())
        
        return {
            "total_samples": total_samples,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": round(spam_percentage, 2),
            "ham_percentage": round(ham_percentage, 2),
            "balance_ratio": round(balance_ratio, 3),
            "average_text_length": round(avg_text_length, 2),
            "min_text_length": min_text_length,
            "max_text_length": max_text_length,
            "average_word_count": round(avg_word_count, 2),
            "training_samples_used": 2000  # From app.py optimization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating dataset stats: {str(e)}")

@app.get("/dataset/distribution")
async def get_dataset_distribution():
    """Get dataset distribution data for charts"""
    try:
        df = analyze_dataset()
        
        # Text length distribution bins
        text_lengths = df['preprocessed_text'].str.len()
        length_bins = [0, 50, 100, 200, 300, 500, 1000, float('inf')]
        length_labels = ['0-50', '50-100', '100-200', '200-300', '300-500', '500-1000', '1000+']
        
        df['length_bin'] = pd.cut(text_lengths, bins=length_bins, labels=length_labels, right=False)
        
        distribution_data = []
        for label in length_labels:
            bin_data = df[df['length_bin'] == label]
            spam_count = int(bin_data['spam'].sum())
            ham_count = len(bin_data) - spam_count
            
            distribution_data.append({
                "range": label,
                "spam": spam_count,
                "ham": ham_count,
                "total": len(bin_data)
            })
        
        # Word count distribution
        word_counts = df['preprocessed_text'].str.split().str.len()
        word_bins = [0, 10, 20, 30, 50, 100, float('inf')]
        word_labels = ['0-10', '10-20', '20-30', '30-50', '50-100', '100+']
        
        df['word_bin'] = pd.cut(word_counts, bins=word_bins, labels=word_labels, right=False)
        
        word_distribution = []
        for label in word_labels:
            bin_data = df[df['word_bin'] == label]
            spam_count = int(bin_data['spam'].sum())
            ham_count = len(bin_data) - spam_count
            
            word_distribution.append({
                "range": label,
                "spam": spam_count,
                "ham": ham_count,
                "total": len(bin_data)
            })
        
        return {
            "text_length_distribution": distribution_data,
            "word_count_distribution": word_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating distribution: {str(e)}")

@app.get("/dataset/features")
async def get_dataset_features():
    """Get top words and features from dataset"""
    try:
        df = analyze_dataset()
        
        # Separate spam and ham messages
        spam_messages = df[df['spam'] == 1]['preprocessed_text']
        ham_messages = df[df['spam'] == 0]['preprocessed_text']
        
        def get_top_words(messages, top_n=15):
            """Extract top words from messages"""
            all_words = []
            for text in messages:
                if pd.notna(text):
                    words = str(text).split()
                    all_words.extend(words)
            
            from collections import Counter
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(top_n)
            
            return [{"word": word, "count": count} for word, count in top_words]
        
        top_spam_words = get_top_words(spam_messages, 15)
        top_ham_words = get_top_words(ham_messages, 15)
        
        return {
            "top_spam_words": top_spam_words,
            "top_ham_words": top_ham_words
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing features: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
