from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
from pathlib import Path

app = FastAPI(title="ClimaCast API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DB_PATH = Path("climacast.db")

def init_db():
    """Initialize the database"""
    conn = sqlite3.connect(DB_PATH)
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

# Initialize database on startup
init_db()

# Pydantic models
class ForecastRequest(BaseModel):
    region: str
    variable: str
    start_date: str
    end_date: str
    model_type: str = "prophet"

class AnalysisRequest(BaseModel):
    region: str
    variable: str
    start_year: int
    end_year: int

class ForecastResponse(BaseModel):
    dates: List[str]
    values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    insights: Dict[str, Any]

class AnalysisResponse(BaseModel):
    dates: List[str]
    values: List[float]
    statistics: Dict[str, float]
    insights: List[str]

# Sample data initialization
def add_sample_data():
    """Add sample regions and climate data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Add sample regions
    regions = [
        ("New York", "USA", 40.7128, -74.0060),
        ("London", "UK", 51.5074, -0.1278),
        ("Tokyo", "Japan", 35.6762, 139.6503),
        ("Sydney", "Australia", -33.8688, 151.2093)
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO regions (name, country, latitude, longitude) VALUES (?, ?, ?, ?)",
        regions
    )
    
    # Add sample climate data
    cursor.execute("SELECT id FROM regions")
    region_ids = cursor.fetchall()
    
    for region_id in region_ids:
        region_id = region_id[0]
        
        # Generate sample data for the last 5 years
        base_date = datetime.now() - timedelta(days=5*365)
        dates = [base_date + timedelta(days=i) for i in range(5*365)]
        
        climate_data = []
        for date in dates:
            # Generate realistic climate data with trends
            day_of_year = date.timetuple().tm_yday
            year_progress = (date.year - 2019) / 4
            
            temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + year_progress * 2 + np.random.normal(0, 2)
            rainfall = max(0, 50 + 30 * np.sin(2 * np.pi * day_of_year / 365) - year_progress * 10 + np.random.normal(0, 10))
            co2 = 400 + year_progress * 10 + np.random.normal(0, 1)
            
            climate_data.append((
                region_id, date, temperature, rainfall, co2,
                np.random.uniform(40, 80),  # humidity
                np.random.uniform(5, 25),   # wind_speed
                np.random.uniform(990, 1020) # pressure
            ))
        
        cursor.executemany(
            """INSERT INTO climate_data 
               (region_id, date, temperature, rainfall, co2, humidity, wind_speed, pressure) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            climate_data
        )
    
    conn.commit()
    conn.close()

# Add sample data on startup
add_sample_data()

# ML Models
class ClimateForecaster:
    def __init__(self):
        self.models = {}
    
    def train_prophet_model(self, data: pd.DataFrame, variable: str):
        """Train Prophet model for forecasting"""
        df = data[['date', variable]].rename(columns={'date': 'ds', variable: 'y'})
        df = df.dropna()
        
        if len(df) < 10:
            raise ValueError("Insufficient data for training")
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive'
        )
        
        model.fit(df)
        return model
    
    def train_arima_model(self, data: pd.DataFrame, variable: str):
        """Train ARIMA model for forecasting"""
        series = data[variable].dropna()
        
        if len(series) < 20:
            raise ValueError("Insufficient data for ARIMA training")
        
        # Simple ARIMA(1,1,1) model
        model = ARIMA(series, order=(1, 1, 1))
        fitted_model = model.fit()
        return fitted_model
    
    def forecast_prophet(self, model, periods: int):
        """Generate forecast using Prophet model"""
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast
    
    def forecast_arima(self, model, periods: int):
        """Generate forecast using ARIMA model"""
        forecast = model.forecast(steps=periods)
        return forecast

forecaster = ClimateForecaster()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ClimaCast API - Climate Impact Predictor", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    """Generate climate forecast"""
    try:
        # Get region ID
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM regions WHERE name = ?", (request.region,))
        region_result = cursor.fetchone()
        
        if not region_result:
            raise HTTPException(status_code=404, detail="Region not found")
        
        region_id = region_result[0]
        
        # Get historical data
        cursor.execute(f"""
            SELECT date, {request.variable} 
            FROM climate_data 
            WHERE region_id = ? AND {request.variable} IS NOT NULL
            ORDER BY date
        """, (region_id,))
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < 10:
            raise HTTPException(status_code=400, detail="Insufficient historical data for forecasting")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['date', request.variable])
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate forecast period
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        periods = (end_date - start_date).days + 1
        
        if periods <= 0:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        # Train model and generate forecast
        if request.model_type == "prophet":
            model = forecaster.train_prophet_model(df, request.variable)
            forecast = forecaster.forecast_prophet(model, periods)
            
            # Extract forecast values
            forecast_values = forecast['yhat'].tail(periods).tolist()
            confidence_lower = forecast['yhat_lower'].tail(periods).tolist()
            confidence_upper = forecast['yhat_upper'].tail(periods).tolist()
            
        elif request.model_type == "arima":
            model = forecaster.train_arima_model(df, request.variable)
            forecast_values = forecaster.forecast_arima(model, periods).tolist()
            
            # Simple confidence intervals for ARIMA
            confidence_lower = [v * 0.95 for v in forecast_values]
            confidence_upper = [v * 1.05 for v in forecast_values]
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # Generate dates
        forecast_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(periods)]
        
        # Calculate insights
        avg_value = np.mean(forecast_values)
        trend = (forecast_values[-1] - forecast_values[0]) / len(forecast_values) if len(forecast_values) > 1 else 0
        
        insights = {
            "average_value": float(avg_value),
            "trend_per_day": float(trend),
            "trend_per_year": float(trend * 365),
            "data_points": len(forecast_values),
            "model_used": request.model_type
        }
        
        return ForecastResponse(
            dates=forecast_dates,
            values=forecast_values,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            insights=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis", response_model=AnalysisResponse)
async def analysis(request: AnalysisRequest):
    """Analyze historical climate data"""
    try:
        # Get region ID
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM regions WHERE name = ?", (request.region,))
        region_result = cursor.fetchone()
        
        if not region_result:
            raise HTTPException(status_code=404, detail="Region not found")
        
        region_id = region_result[0]
        
        # Get historical data for the specified period
        cursor.execute(f"""
            SELECT date, {request.variable} 
            FROM climate_data 
            WHERE region_id = ? 
              AND {request.variable} IS NOT NULL
              AND date >= ? 
              AND date <= ?
            ORDER BY date
        """, (region_id, f"{request.start_year}-01-01", f"{request.end_year}-12-31"))
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) == 0:
            raise HTTPException(status_code=404, detail="No data found for the specified period")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['date', request.variable])
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate statistics
        values = df[request.variable].values
        statistics = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "data_points": len(values)
        }
        
        # Calculate trend
        if len(values) > 1:
            # Simple linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            annual_trend = slope * 365  # Convert to annual trend
        else:
            annual_trend = 0
        
        statistics["annual_trend"] = float(annual_trend)
        
        # Generate insights
        insights = []
        
        if request.variable == "temperature":
            if annual_trend > 0.5:
                insights.append("Temperature is rising at an alarming rate")
            elif annual_trend > 0:
                insights.append("Temperature shows an increasing trend")
            elif annual_trend < -0.5:
                insights.append("Temperature is decreasing significantly")
            else:
                insights.append("Temperature remains relatively stable")
                
        elif request.variable == "rainfall":
            if annual_trend < -20:
                insights.append("Rainfall is decreasing at a concerning rate")
            elif annual_trend < -5:
                insights.append("Rainfall shows a declining trend")
            elif annual_trend > 10:
                insights.append("Rainfall is increasing significantly")
            else:
                insights.append("Rainfall patterns are relatively stable")
                
        elif request.variable == "co2":
            if annual_trend > 2:
                insights.append("CO2 levels are rising rapidly")
            elif annual_trend > 0.5:
                insights.append("CO2 levels show an increasing trend")
            elif annual_trend < 0:
                insights.append("CO2 levels are decreasing")
            else:
                insights.append("CO2 levels remain relatively stable")
        
        return AnalysisResponse(
            dates=df['date'].dt.strftime("%Y-%m-%d").tolist(),
            values=values.tolist(),
            statistics=statistics,
            insights=insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regions")
async def get_regions():
    """Get list of available regions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, country, latitude, longitude FROM regions")
    regions = cursor.fetchall()
    conn.close()
    
    return [
        {
            "name": row[0],
            "country": row[1],
            "latitude": row[2],
            "longitude": row[3]
        }
        for row in regions
    ]

@app.post("/upload-data")
async def upload_climate_data(
    file: UploadFile = File(...),
    region: str = None,
    variable: str = None
):
    """Upload climate data from CSV file"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['date', variable or 'value']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSV must contain 'date' and variable columns")
        
        # Get region ID
        if region:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM regions WHERE name = ?", (region,))
            region_result = cursor.fetchone()
            
            if not region_result:
                raise HTTPException(status_code=404, detail="Region not found")
            
            region_id = region_result[0]
            
            # Insert data into database
            for _, row in df.iterrows():
                cursor.execute(f"""
                    INSERT INTO climate_data (region_id, date, {variable})
                    VALUES (?, ?, ?)
                """, (region_id, row['date'], row[variable]))
            
            conn.commit()
            conn.close()
        
        return {"message": f"Successfully uploaded {len(df)} data points"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export-report")
async def export_report(region: str, variable: str, start_date: str, end_date: str):
    """Export climate report as JSON"""
    try:
        # Get both forecast and analysis data
        forecast_request = ForecastRequest(
            region=region,
            variable=variable,
            start_date=start_date,
            end_date=end_date
        )
        
        forecast_data = await predict(forecast_request)
        
        # Create report
        report = {
            "region": region,
            "variable": variable,
            "period": {
                "start": start_date,
                "end": end_date
            },
            "forecast": forecast_data.dict(),
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "source": "ClimaCast API",
                "version": "1.0.0"
            }
        }
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)