# 🌦️ ClimaCast - AI-Powered Climate Impact Predictor

**ClimaCast** is an advanced climate forecasting and analysis platform that combines real-time weather data, machine learning predictions, and AI-powered insights to help users understand climate patterns and make informed decisions.

![ClimaCast Logo](https://img.shields.io/badge/ClimaCast-AI%20Climate%20Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning)
- Internet connection (for API data)

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/climacast.git
   cd climacast
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   streamlit run climacast_app.py
   ```

4. **Access the Application**
   - Open your browser and go to: `http://localhost:8501`

## 📋 Complete Setup Guide

### Method 1: Automatic Setup (Recommended)

**For Windows Users:**
```cmd
# Clone the repository
git clone https://github.com/yourusername/climacast.git
cd climacast

# Run the setup script
bash setup.sh

# Start the application
start.bat
```

**For Linux/Mac Users:**
```bash
# Clone the repository
git clone https://github.com/yourusername/climacast.git
cd climacast

# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Start the application
./start.sh
```

### Method 2: Manual Setup

1. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create Required Directories**
   ```bash
   mkdir data exports logs
   ```

4. **Initialize Database**
   ```bash
   python -c "
   import sqlite3
   import os
   
   if not os.path.exists('climacast.db'):
       conn = sqlite3.connect('climacast.db')
       cursor = conn.cursor()
       
       # Create regions table
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS regions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               country TEXT NOT NULL,
               latitude REAL NOT NULL,
               longitude REAL NOT NULL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       
       # Create climate_data table
       cursor.execute('''
           CREATE TABLE IF NOT EXISTS climate_data (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               region_id INTEGER,
               date TIMESTAMP NOT NULL,
               temperature REAL,
               rainfall REAL,
               co2 REAL,
               humidity REAL,
               wind_speed REAL,
               pressure REAL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               FOREIGN KEY (region_id) REFERENCES regions (id)
           )
       ''')
       
       conn.commit()
       conn.close()
       print('Database initialized successfully')
   else:
       print('Database already exists')
   "
   ```

5. **Create Environment File**
   ```bash
   # Create .env file
   echo "DATABASE_URL=sqlite:///climacast.db" > .env
   echo "STREAMLIT_PORT=8501" >> .env
   ```

6. **Run the Application**
   ```bash
   streamlit run climacast_app.py
   ```

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop installed
- Docker Compose installed

### Quick Docker Setup

1. **Build and Run**
   ```bash
   docker-compose up --build
   ```

2. **Access Application**
   - Application: `http://localhost:8501`
   - With Nginx: `http://localhost:80`

3. **Stop Application**
   ```bash
   docker-compose down
   ```

### Individual Docker Commands

```bash
# Build Docker image
docker build -t climacast .

# Run container
docker run -p 8501:8501 climacast

# Run with volume mounts
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  -v $(pwd)/logs:/app/logs \
  climacast
```

## 📱 Application Features

### 🌐 Web Interface
- **Modern Streamlit Dashboard**: Clean, responsive interface
- **Interactive Visualizations**: Dynamic charts with Plotly
- **Real-time Updates**: Live data refresh capabilities
- **Mobile Responsive**: Works on all devices

### 📈 Forecasting Capabilities
- **Multi-Variable Predictions**: Temperature, rainfall, humidity, CO2
- **Machine Learning Models**: LSTM and Prophet models
- **Multiple Regions**: Support for major cities worldwide
- **Confidence Intervals**: Statistical uncertainty measures

### 🤖 AI-Powered Features
- **Climate AI Advisor**: Intelligent analysis and recommendations
- **Audience-Specific Insights**: Tailored for farmers, policymakers, public health
- **Interactive Chatbot**: Real-time climate Q&A system
- **Fallback Analysis**: Basic insights when AI services unavailable

### 📊 Advanced Analytics
- **Historical Data Integration**: NOAA API integration
- **Trend Analysis**: Statistical analysis of climate patterns
- **Anomaly Detection**: Identification of unusual climate events
- **Export Capabilities**: PDF and image export for reports

## 🛠️ Technology Stack

### Backend & Core
- **Python 3.8+**: Main programming language
- **Streamlit**: Web application framework
- **FastAPI**: API endpoints for data services
- **SQLAlchemy**: Database ORM
- **SQLite**: Local database storage

### Data & Analytics
- **Pandas & NumPy**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **Prophet**: Time series forecasting
- **Scikit-learn**: Machine learning models
- **Statsmodels**: Statistical analysis

### AI & Machine Learning
- **Ollama Integration**: Local LLM support
- **LSTM Models**: Deep learning for time series
- **Prophet Models**: Facebook's forecasting tool
- **Custom AI Advisor**: Climate-specific insights

### External APIs
- **NOAA API**: Real-time weather data
- **Ollama**: Local AI model hosting

## 📁 Project Structure

```
ClimaCast/
├── 📱 Main Application
│   ├── climacast_app.py          # Main Streamlit application
│   ├── climacast_api.py          # FastAPI backend services
│   └── climate_data_fetcher.py  # NOAA API integration
│
├── 🤖 AI Components
│   ├── ai_advisor_new.py         # AI climate advisor
│   └── ai_chatbot.py            # Interactive AI chatbot
│
├── 🧠 Machine Learning
│   ├── ml/
│   │   ├── enhanced_models.py   # ML model implementations
│   │   ├── preprocess_data.py   # Data preprocessing
│   │   ├── train_prophet.py     # Prophet model training
│   │   └── models/              # Trained model files
│   │       ├── *.h5             # LSTM models
│   │       ├── *.pkl             # Scaler files
│   │       └── *.json            # Prophet models
│
├── 📊 Data & Reports
│   ├── report_exporter.py       # Export functionality
│   ├── climacast.db            # SQLite database
│   ├── data/                   # Data storage
│   ├── exports/                 # Exported reports
│   └── logs/                   # Application logs
│
├── ⚙️ Configuration
│   ├── requirements.txt         # Python dependencies
│   ├── setup.sh               # Setup script
│   ├── start.bat              # Windows startup script
│   ├── stop.bat               # Windows stop script
│   ├── .env                   # Environment variables
│   ├── .gitignore             # Git ignore rules
│   └── README.md              # This file
│
├── 🐳 Docker Files
│   ├── Dockerfile             # Docker configuration
│   ├── docker-compose.yml     # Multi-container setup
│   ├── .dockerignore          # Docker ignore rules
│   ├── nginx.conf             # Reverse proxy config
│   └── DOCKER_DEPLOYMENT.md   # Docker deployment guide
```

## 🚀 Running Procedures

### Development Mode

1. **Start Development Server**
   ```bash
   # Windows
   start.bat
   
   # Linux/Mac
   streamlit run climacast_app.py
   ```

2. **Access Application**
   - Open browser: `http://localhost:8501`

3. **Stop Development Server**
   ```bash
   # Windows
   stop.bat
   
   # Linux/Mac
   Ctrl+C in terminal
   ```

### Production Mode

1. **Using Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

2. **Using Docker**
   ```bash
   docker build -t climacast .
   docker run -d -p 8501:8501 climacast
   ```

3. **Access Application**
   - Application: `http://localhost:8501`
   - With Nginx: `http://localhost:80`

### Cloud Deployment

1. **AWS ECS**
   ```bash
   # Push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   docker tag climacast:latest your-account.dkr.ecr.us-east-1.amazonaws.com/climacast:latest
   docker push your-account.dkr.ecr.us-east-1.amazonaws.com/climacast:latest
   ```

2. **Google Cloud Run**
   ```bash
   # Build and deploy
   gcloud run deploy climacast --source . --platform managed --region us-central1 --allow-unauthenticated
   ```

3. **Azure Container Instances**
   ```bash
   # Build and deploy
   az container create --resource-group myResourceGroup --name climacast --image climacast:latest --ports 8501 --dns-name-label climacast-app
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=sqlite:///climacast.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501

# Model Configuration
DEFAULT_MODEL=prophet
FORECAST_DAYS=365

# External API Keys (Optional)
OPENWEATHER_API_KEY=your_openweather_api_key_here
NASA_API_KEY=your_nasa_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/climacast.log
```

### API Configuration
- **NOAA API**: Get free token from [NOAA API](https://www.ncdc.noaa.gov/cdo-web/token)
- **Ollama**: Default localhost:11434
- **Database**: SQLite (no configuration needed)

## 🧪 Machine Learning Models

### Available Models
- **LSTM**: Deep learning for time series forecasting
- **Prophet**: Facebook's forecasting tool
- **Statistical Models**: ARIMA, seasonal decomposition

### Model Training
```bash
# Train Prophet models
python ml/train_prophet.py

# Preprocess data
python ml/preprocess_data.py

# Enhanced models
python ml/enhanced_models.py
```

### Pre-trained Models
- **Temperature Models**: T2M_lstm.h5 & T2M_prophet.json
- **Precipitation Models**: PRECTOTCORR_lstm.h5 & PRECTOTCORR_prophet.json
- **Humidity Models**: RH2M_lstm.h5 & RH2M_prophet.json
- **Region-Specific Models**: Custom models for major cities

## 📊 API Endpoints

### FastAPI Backend (`climacast_api.py`)
- `GET /api/forecast`: Get climate forecasts
- `GET /api/analysis`: Statistical analysis
- `POST /api/insights`: AI insights generation
- `GET /api/health`: Health check

## 🚀 Deployment Options

### ✅ Recommended (smoothest): Render + Docker

This is the most reliable and fastest way to get a public URL for interviews.

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render from your repo.
3. Render auto-builds from `Dockerfile`.
4. Set environment variables in Render:
   - `NOAA_API_TOKEN`
   - `OPENWEATHER_API_KEY`
   - `PORT` (auto provided by Render)
   - Optional: `OLLAMA_URL` (if using external hosted Ollama)
5. Deploy and open the public URL.

Notes:
- Dockerfile already supports dynamic cloud port via `PORT`.
- If Ollama is not hosted in cloud, app still runs with safe AI fallback behavior.

### Local Development
```bash
streamlit run climacast_app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud
streamlit run climacast_app.py --server.port 8501

# Using Docker
docker build -t climacast .
docker run -p 8501:8501 climacast

# Using Docker Compose
docker-compose up -d --build
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions
- Write unit tests for new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **"AI insights not available"**
   - Ensure Ollama is running
   - Check model is pulled (`ollama pull qwen:0.5b`)
   - Verify Ollama URL configuration

2. **"NOAA API error"**
   - Check API token validity
   - Verify internet connection
   - Check API rate limits

3. **"Database connection error"**
   - Ensure SQLite file permissions
   - Check disk space
   - Verify database file integrity

4. **"Port already in use"**
   - Change port in configuration
   - Kill existing processes
   - Use different port number

5. **"Dependencies not found"**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility
   - Use virtual environment

### Getting Help
- Check the documentation
- Review error messages
- Test with sample data
- Check logs in `logs/` directory

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time alerts and notifications
- [ ] Mobile app development
- [ ] Advanced ML model integration
- [ ] Multi-language support
- [ ] Cloud deployment options
- [ ] API rate limiting and caching
- [ ] User authentication and profiles
- [ ] Custom dashboard creation

### Technical Improvements
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] Test coverage expansion
- [ ] Documentation updates
- [ ] Security enhancements


## 🎉 Ready to Go!

Your ClimaCast application is now ready for:
- ✅ **Local Development**
- ✅ **Production Deployment**
- ✅ **Cloud Deployment**
- ✅ **Docker Containerization**
- ✅ **Hackathon Presentation**

**Start exploring climate data with AI-powered insights! 🌦️**

---

*Built with ❤️ for climate awareness and sustainable future*