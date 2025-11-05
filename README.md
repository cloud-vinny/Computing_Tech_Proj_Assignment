🛡️ Spam Detection App

A full-stack web application that uses machine learning to detect spam emails and messages. Built with Next.js, TypeScript, FastAPI, and advanced AI models including Logistic Regression, Naive Bayes, and K-Means clustering.

## 🌟 Features

- **🤖 Multiple AI Models**: Choose between Logistic Regression, Naive Bayes, or K-Means clustering
- **📊 Confidence Scores**: Get detailed confidence percentages for each prediction
- **🌓 Dark Mode**: Toggle between light and dark themes with persistent user preference
- **🎨 Modern UI**: Clean, responsive design with scalable CSS design system using custom properties
- **⚡ Real-time Analysis**: Instant spam detection with loading states
- **🔒 Secure**: No data storage - your messages are analyzed and forgotten
- **📱 Fully Responsive**: Optimized for all device sizes (mobile, tablet, desktop) with breakpoint-based layouts
- **🎯 Input Validation**: Minimum 5 words requirement for accurate spam detection (coming in next update)

## 🚀 Live Demo

**🌐 Frontend:** [View Live Application on Vercel](https://spamdetector-mauve.vercel.app/)  
**🔧 Backend API:** [Railway Backend](https://computingtechprojassignment-production.up.railway.app/health)  
**✅ Status:** Backend integrated and ready!


## 🛠️ Tech Stack

### Frontend
- **Next.js 15.5.5** - React framework with App Router
- **TypeScript** - Type-safe development
- **React Context API** - Global state management for theme switching
- **Custom CSS Design System** - Scalable architecture with CSS custom properties (variables)
- **Axios** - HTTP client for API calls

### Backend
- **FastAPI** - High-performance Python web framework
- **Scikit-learn** - Machine learning library
- **Pandas & NumPy** - Data processing
- **NLTK** - Natural language processing

### AI Models
- **Logistic Regression** - Best overall performance
- **Naive Bayes** - Text classification specialist
- **K-Means Clustering** - Pattern recognition

## 📋 Prerequisites

Before running this project, make sure you have:

- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **Git** (for cloning the repository)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cloud-vinny/Computing_Tech_Proj_Assignment.git
cd Computing_Tech_Proj_Assignment
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The project uses `requirements.txt` with optimized dependency versions for Railway deployment.

### 4. Run the Application

#### Start the Backend (Terminal 1)
```bash
py app.py
```
The FastAPI server will start on `http://localhost:8000`

#### Start the Frontend (Terminal 2)
```bash
npm run dev
```
The Next.js app will start on `http://localhost:3000`

### 5. Open Your Browser

Visit `http://localhost:3000` to use the spam detection app!

## 📁 Project Structure

```
Computing_Tech_Proj_Assignment/
├── app/                          # Next.js app directory
│   ├── globals.css               # Global styles with CSS design system
│   ├── layout.tsx                # Root layout with ThemeProvider
│   └── page.tsx                  # Home page
├── components/                   # React components
│   ├── SpamDetectionForm.tsx    # Main detection form
│   ├── Header.tsx               # Header component
│   ├── Footer.tsx               # Footer component
│   └── ThemeToggle.tsx          # Dark/light mode toggle button
├── contexts/                     # React Context providers
│   └── ThemeContext.tsx         # Theme state management
├── lib/                         # Utility libraries
│   └── api.ts                   # API client
├── dataset/                     # Training data
│   ├── cleaned_dataset.csv      # Full processed dataset
│   ├── cleaned_dataset_small.csv # Optimized dataset for training
│   └── cleaned_dataset_full_backup.csv # Backup dataset
├── app.py                       # FastAPI backend server
├── Best_Model.ipynb            # AI model training notebook
├── Spam1_2_Final.ipynb         # Data preprocessing notebook
├── requirements.txt             # Python dependencies
├── railway.json                 # Railway deployment configuration
├── runtime.txt                  # Python version specification
├── package.json                 # Node.js dependencies
└── README.md                    # This file
```

## 🔧 Development

### Available Scripts

```bash
# Frontend
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint

# Backend
py app.py            # Start FastAPI server
```

### Environment Variables

Create a `.env.local` file in the root directory:

```env
# Optional: Customize API endpoints
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🚀 Deployment

### Vercel Deployment (Recommended)

1. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Sign in with your GitHub account
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Build Settings**:
   - Framework Preset: **Next.js**
   - Root Directory: `./` (default)
   - Build Command: `npm run build`
   - Output Directory: `.next` (default)

3. **Deploy**:
   - Click "Deploy"
   - Wait for deployment to complete
   - Your app will be live at `https://your-app.vercel.app`

### Backend Deployment

For the FastAPI backend, consider deploying to:
- **Railway** - Easy Python deployment
- **Render** - Free tier available
- **Heroku** - Popular platform
- **DigitalOcean** - VPS deployment

## 📊 API Endpoints

### FastAPI Backend (`http://localhost:8000`)

- `GET /health` - Health check
- `GET /models` - Available AI models
- `POST /detect` - Spam detection endpoint

**Request Body:**
```json
{
  "text": "Your message here (minimum 5 words required)",
  "model": "logistic" // Options: "logistic", "naive_bayes", "kmeans"
}
```

**Note:** Input validation requires a minimum of 5 words for accurate spam detection. This will be enforced in the frontend in the next update.

#### Example API Usage

   ```bash
# Health check
curl http://localhost:8000/health

# Detect spam
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Win money now!", "model": "logistic"}'
```

## 🤖 AI Models Explained

### 1. Logistic Regression (Recommended)
- **Best overall performance**
- Uses TF-IDF vectorization
- Trained on balanced dataset
- High accuracy for spam detection

### 2. Naive Bayes
- **Text classification specialist**
- Probabilistic approach
- Fast inference
- Good for text-based features

### 3. K-Means Clustering
- **Pattern recognition**
- Unsupervised learning
- Groups similar messages
- Identifies spam clusters

## 🎨 Design System

This project uses a scalable CSS architecture based on custom properties (CSS variables) for:

- **Theme Management**: Light and dark themes with seamless switching
- **Responsive Breakpoints**: Mobile-first design with breakpoints at 640px (sm), 768px (md), 1024px (lg), 1280px (xl)
- **Color System**: Centralized color tokens for consistent theming
- **Spacing System**: Standardized spacing units for layout consistency
- **Component Styling**: Reusable design tokens for buttons, cards, inputs, and results

The theme system uses React Context API to provide global theme state across all components, with localStorage persistence for user preferences.

## 📈 Performance

- **Build Time**: ~18 seconds
- **Bundle Size**: 124 kB (First Load JS)
- **Lighthouse Score**: 95+ (Performance)
- **Model Accuracy**: 95%+ (Logistic Regression)
- **Theme Switching**: Instant with zero layout shift

## 🐛 Troubleshooting

### Common Issues

1. **Port 3000/8000 already in use**:
   ```bash
   # Kill processes using the ports
   npx kill-port 3000 8000
   ```

2. **Python dependencies issues**:
   ```bash
   # Install with specific versions
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Build failures**:
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules .next
   npm install
   npm run build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Cloud Vinny**
- GitHub: [@cloud-vinny](https://github.com/cloud-vinny)
- Project: [Computing_Tech_Proj_Assignment](https://github.com/cloud-vinny/Computing_Tech_Proj_Assignment)

## 🙏 Acknowledgments

- **Scikit-learn** for machine learning algorithms
- **Next.js** team for the amazing React framework
- **FastAPI** for the high-performance Python backend
- **Vercel** for seamless deployment platform

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an [Issue](https://github.com/cloud-vinny/Computing_Tech_Proj_Assignment/issues)
3. Contact the maintainer

**Happy Spam Detection! 🛡️✨**
