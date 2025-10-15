# Spam Detection Full-Stack Application

A complete full-stack web application for AI-powered spam detection using Next.js, TypeScript, FastAPI, and machine learning.

## 🚀 Tech Stack

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **CSS** - Custom styling (no frameworks)
- **Axios** - HTTP client for API calls

### Backend
- **FastAPI** - Modern Python web framework
- **Scikit-learn** - Machine learning models
- **NLTK** - Natural language processing
- **Pandas** - Data processing

### AI Models
- **Logistic Regression** - Best overall performance (96.5% accuracy)
- **Naive Bayes** - Text classification
- **K-Means** - Clustering analysis

## 📁 Project Structure

```
├── app/                          # Next.js App Router
│   ├── globals.css               # Global styles
│   ├── layout.tsx               # Root layout
│   └── page.tsx                 # Home page
├── components/                   # React components
│   ├── Header.tsx               # Navigation header
│   ├── Footer.tsx               # Footer component
│   └── SpamDetectionForm.tsx    # Main detection form
├── lib/                         # Utility functions
│   └── api.ts                   # API client with TypeScript
├── dataset/                     # ML datasets
├── app.py                       # FastAPI backend
├── requirements_fastapi.txt     # Python dependencies
├── package.json                 # Node.js dependencies
└── tsconfig.json               # TypeScript configuration
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### 1. Install Python Dependencies
```bash
pip install -r requirements_fastapi.txt
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🚀 Running the Application

### Start FastAPI Backend (Terminal 1)
```bash
python app.py
```
Backend will be available at: http://localhost:8000

### Start Next.js Frontend (Terminal 2)
```bash
npm run dev
```
Frontend will be available at: http://localhost:3000

## 🔧 API Endpoints

### FastAPI Backend (http://localhost:8000)

#### POST `/detect`
Analyze text for spam detection.

**Request:**
```json
{
  "text": "Your email content here...",
  "model": "logistic"
}
```

**Response:**
```json
{
  "is_spam": true,
  "confidence": 0.85,
  "model_used": "Logistic Regression",
  "message": "This message appears to be spam."
}
```

#### GET `/health`
Check API health status.

#### GET `/models`
Get available AI models.

## 🎯 Features

### Frontend Features
- **Modern UI** - Clean, responsive design with custom CSS
- **TypeScript** - Full type safety and better development experience
- **Real-time Analysis** - Instant spam detection with confidence scores
- **Model Selection** - Choose between different AI models
- **Error Handling** - User-friendly error messages
- **Loading States** - Visual feedback during analysis

### Backend Features
- **RESTful API** - Clean, documented endpoints
- **Multiple Models** - Logistic Regression, Naive Bayes, K-Means
- **Text Preprocessing** - NLTK-based text cleaning and tokenization
- **CORS Support** - Cross-origin requests enabled
- **Health Checks** - API status monitoring
- **Error Handling** - Comprehensive error responses

### AI Model Features
- **High Accuracy** - 96.5% accuracy with Logistic Regression
- **Confidence Scores** - Probability-based predictions
- **Multiple Algorithms** - Different approaches for comparison
- **Real-time Processing** - Fast inference for user input

## 📊 Model Performance

### Logistic Regression (Recommended)
- **Accuracy**: 96.5%
- **Precision**: 88.7%
- **Recall**: 94.6%
- **F1-Score**: 91.6%
- **ROC-AUC**: 99.2%

### Available Models
1. **Logistic Regression** - Best overall performance
2. **Naive Bayes** - Good for text classification
3. **K-Means** - Unsupervised clustering

## 🧪 Testing

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Spam detection
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Buy one get one free!", "model": "logistic"}'
```

### Test the Frontend
1. Open http://localhost:3000
2. Enter sample text
3. Select AI model
4. Click "Analyze for Spam"

## 🚀 Deployment

### Frontend (Vercel)
1. Connect GitHub repository to Vercel
2. Set environment variables
3. Deploy automatically

### Backend (Railway/Heroku)
1. Add `Procfile` with: `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
2. Deploy with Python buildpack
3. Set environment variables

## 📝 Development Notes

### Code Structure
- **TypeScript** - Full type safety with interfaces
- **Component-based** - Reusable React components
- **API Client** - Centralized API communication
- **Error Boundaries** - Graceful error handling

### Best Practices
- **Type Safety** - All API calls are typed
- **Error Handling** - Comprehensive error management
- **Loading States** - User feedback during operations
- **Responsive Design** - Mobile-first approach

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Computing Technology Project Assignment.

---

**Built with ❤️ using Next.js, TypeScript, FastAPI, and Machine Learning**