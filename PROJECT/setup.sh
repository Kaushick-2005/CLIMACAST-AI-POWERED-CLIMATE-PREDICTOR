#!/bin/bash

# ClimaCast Setup Script
# This script helps set up the ClimaCast Climate Impact Predictor

set -e

echo "🌦️  ClimaCast - Climate Impact Predictor Setup"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python version $python_version is not compatible. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip is installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Requirements installed successfully"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p models
mkdir -p exports
mkdir -p logs

echo "✅ Directories created"

# Initialize database
echo "🗄️  Initializing database..."
python3 -c "
import sqlite3
import os

# Create database if it doesn't exist
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
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region_id INTEGER,
            variable TEXT NOT NULL,
            model_type TEXT NOT NULL,
            target_date TIMESTAMP NOT NULL,
            predicted_value REAL NOT NULL,
            confidence_lower REAL,
            confidence_upper REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (region_id) REFERENCES regions (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print('✅ Database initialized successfully')
else:
    print('✅ Database already exists'
"

# Create sample data
echo "📊 Creating sample data..."
python3 -c "
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('climacast.db')
cursor = conn.cursor()

# Add sample regions
regions = [
    ('New York', 'USA', 40.7128, -74.0060),
    ('London', 'UK', 51.5074, -0.1278),
    ('Tokyo', 'Japan', 35.6762, 139.6503),
    ('Sydney', 'Australia', -33.8688, 151.2093),
    ('Mumbai', 'India', 19.0760, 72.8777),
    ('São Paulo', 'Brazil', -23.5505, -46.6333)
]

cursor.executemany(
    'INSERT OR IGNORE INTO regions (name, country, latitude, longitude) VALUES (?, ?, ?, ?)',
    regions
)

# Get region IDs
cursor.execute('SELECT id FROM regions')
region_ids = [row[0] for row in cursor.fetchall()]

# Generate sample climate data for each region
for region_id in region_ids:
    # Generate 5 years of daily data
    base_date = datetime.now() - timedelta(days=5*365)
    dates = [base_date + timedelta(days=i) for i in range(5*365)]
    
    climate_data = []
    for date in dates:
        # Generate realistic climate data with trends
        day_of_year = date.timetuple().tm_yday
        year_progress = (date.year - 2019) / 4
        
        # Temperature with seasonal variation and warming trend
        temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + year_progress * 2 + np.random.normal(0, 2)
        
        # Rainfall with seasonal variation and declining trend
        rainfall = max(0, 50 + 30 * np.sin(2 * np.pi * day_of_year / 365) - year_progress * 10 + np.random.normal(0, 10))
        
        # CO2 with increasing trend
        co2 = 400 + year_progress * 10 + np.random.normal(0, 1)
        
        # Other weather parameters
        humidity = np.random.uniform(40, 80)
        wind_speed = np.random.uniform(5, 25)
        pressure = np.random.uniform(990, 1020)
        
        climate_data.append((
            region_id, date, temperature, rainfall, co2, humidity, wind_speed, pressure
        ))
    
    # Insert data in batches
    cursor.executemany(
        '''INSERT INTO climate_data 
           (region_id, date, temperature, rainfall, co2, humidity, wind_speed, pressure) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        climate_data
    )

conn.commit()
conn.close()

print('✅ Sample data created successfully')
"

# Create environment file
echo "⚙️  Creating environment file..."
cat > .env << EOF
# ClimaCast Environment Configuration

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

# External API Keys (Optional - Add your own keys)
# OPENWEATHER_API_KEY=your_openweather_api_key_here
# NASA_API_KEY=your_nasa_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/climacast.log
EOF

echo "✅ Environment file created"

# Create startup script
echo "🚀 Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash

# ClimaCast Startup Script

echo "🌦️  Starting ClimaCast Climate Impact Predictor..."

# Activate virtual environment
source venv/bin/activate

# Check if database exists
if [ ! -f "climacast.db" ]; then
    echo "🗄️  Database not found. Please run setup.sh first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start API server in background
echo "🚀 Starting API server..."
python3 climacast_api.py > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server started successfully (PID: $API_PID)"
else
    echo "❌ Failed to start API server"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start Streamlit app
echo "🌐 Starting Streamlit app..."
streamlit run climacast_app.py --server.port=8501 --server.address=0.0.0.0 > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

echo "✅ Streamlit app started successfully (PID: $STREAMLIT_PID)"
echo ""
echo "🎉 ClimaCast is now running!"
echo "   - API: http://localhost:8000"
echo "   - Dashboard: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping ClimaCast services..."
    kill $API_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script termination
trap cleanup SIGINT SIGTERM

# Wait for services
wait
EOF

chmod +x start.sh
echo "✅ Startup script created"

# Create stop script
echo "🛑 Creating stop script..."
cat > stop.sh << 'EOF'
#!/bin/bash

# ClimaCast Stop Script

echo "🛑 Stopping ClimaCast services..."

# Find and kill API server
API_PID=$(pgrep -f "climacast_api.py" || true)
if [ ! -z "$API_PID" ]; then
    kill $API_PID
    echo "✅ API server stopped"
else
    echo "ℹ️  API server not running"
fi

# Find and kill Streamlit app
STREAMLIT_PID=$(pgrep -f "streamlit run climacast_app.py" || true)
if [ ! -z "$STREAMLIT_PID" ]; then
    kill $STREAMLIT_PID
    echo "✅ Streamlit app stopped"
else
    echo "ℹ️  Streamlit app not running"
fi

echo "✅ All services stopped"
EOF

chmod +x stop.sh
echo "✅ Stop script created"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Start the application: ./start.sh"
echo "2. Open your browser and go to: http://localhost:8501"
echo "3. View API documentation: http://localhost:8000/docs"
echo "4. Stop the application: ./stop.sh"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "🌦️ Happy climate analyzing!"