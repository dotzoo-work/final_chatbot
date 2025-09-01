# Dr. Meenakshi Tomar Dental Chatbot - Separated Architecture

## 🏗️ Architecture Overview

This project is now separated into **frontend** and **backend** components for better deployment and scalability:

- **Backend**: FastAPI server with AI agents and chat logic
- **Frontend**: Static HTML/CSS/JS that can be hosted anywhere

## 🚀 Quick Start

### Backend (API Server)
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python app.py
```

### Frontend (Web Interface)
```bash
cd frontend
# Update config.js with your backend URL
# Open index.html in browser or deploy to static hosting
```

## 📁 Project Structure

```
├── backend/              # FastAPI API server
│   ├── app.py           # Main server
│   ├── requirements.txt # Python dependencies
│   ├── Dockerfile       # Container config
│   └── [AI modules]     # All Python logic
├── frontend/            # Static web app
│   ├── index.html       # Main interface
│   ├── app.js          # Frontend logic
│   ├── styles.css      # Styling
│   └── config.js       # API configuration
└── DEPLOYMENT_GUIDE.md  # Detailed deployment instructions
```

## 🌐 Deployment Options

### Backend Deployment
- **Heroku**: `git push heroku main`
- **Railway**: `railway up`
- **Render**: Connect GitHub repo
- **Docker**: `docker build -t chatbot .`

### Frontend Deployment
- **Netlify**: Drag & drop frontend folder
- **Vercel**: Connect GitHub repo
- **GitHub Pages**: Enable in repo settings
- **Any static host**: Upload files

## ⚙️ Configuration

1. **Deploy backend first** and get the API URL
2. **Update frontend config**: Edit `frontend/config.js` with your backend URL
3. **Deploy frontend** to any static hosting service

## 🔧 Environment Variables (Backend)

```bash
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
ENVIRONMENT=production
PORT=8000
```

## 🎯 Benefits of Separation

- **Independent scaling**: Scale frontend and backend separately
- **Multiple deployment options**: Choose the best platform for each component
- **Cost optimization**: Use free static hosting for frontend
- **Better security**: API keys only on backend server
- **Easier maintenance**: Update components independently

## 📖 Full Documentation

See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions and platform-specific configurations.

## 🔗 API Endpoints

- `POST /api/chat` - Main chat endpoint
- `GET /api/health` - Health check

## 🛠️ Development

The frontend automatically detects localhost and uses the development API URL. For production, update the configuration in `frontend/config.js`.