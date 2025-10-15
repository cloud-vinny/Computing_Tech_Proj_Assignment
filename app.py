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
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import uvicorn

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI(title="Spam Detection API", version="1.0.0")

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

# Global variables for models
tfidf = None
clf = None
mnb = None
kmeans = None

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
    text = nltk.word_tokenize(text)  # tokenize
    
    # remove special chars & stopwords & punctuation & stemming
    b = []
    for a in text:
        if a.isalnum() and a not in stopwords.words('english') and a not in string.punctuation:
            b.append(ps.stem(a))
    
    return " ".join(b)

# Load and train models
def load_and_train_models():
    global tfidf, clf, mnb, kmeans
    
    try:
        # Load dataset
        df = pd.read_csv("dataset/cleaned_dataset_small.csv")
        df.dropna(subset=['preprocessed_text'], inplace=True)
        
        # Prepare data
        ros = RandomOverSampler(random_state=2)
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        X = tfidf.fit_transform(df['preprocessed_text']).toarray()
        y = df['spam'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
        
        # Train models
        print("Training Logistic Regression...")
        clf = LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=None)
        clf.fit(X_train_bal, y_train_bal)
        
        print("Training Naive Bayes...")
        mnb = MultinomialNB(alpha=0.1)
        mnb.fit(X_train_bal, y_train_bal)
        
        print("Training K-Means...")
        kmeans = KMeans(n_clusters=2, random_state=2)
        kmeans.fit(X)
        
        print("All models trained successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

# Startup event
@app.on_event("startup")
async def startup_event():
    print("Starting up Spam Detection API...")
    load_and_train_models()

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
        
        # Preprocess text
        cleaned_text = text_transform(request.text)
        text_vector = tfidf.transform([cleaned_text])
        
        # Select model and make prediction
        if request.model.lower() == "logistic":
            prediction = clf.predict(text_vector)[0]
            confidence = clf.predict_proba(text_vector)[0][1]  # Probability of being spam
            model_used = "Logistic Regression"
        elif request.model.lower() == "naive_bayes":
            prediction = mnb.predict(text_vector)[0]
            confidence = mnb.predict_proba(text_vector)[0][1]  # Probability of being spam
            model_used = "Naive Bayes"
        elif request.model.lower() == "kmeans":
            prediction = kmeans.predict(text_vector)[0]
            # For K-Means, we'll use distance to cluster centers as confidence
            distances = kmeans.transform(text_vector)
            confidence = 1 - (distances.min() / distances.max())  # Normalized confidence
            model_used = "K-Means"
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Use 'logistic', 'naive_bayes', or 'kmeans'")
        
        # Determine result message
        if prediction == 1:
            message = "This message appears to be spam."
        else:
            message = "This message appears to be legitimate."
        
        return SpamResponse(
            is_spam=bool(prediction),
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
