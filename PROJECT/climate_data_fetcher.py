import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import time
import hashlib
from pathlib import Path
import pickle
import os

class ClimateDataFetcher:
    """Fetch real climate data from NOAA and OpenWeatherMap APIs with caching and monitoring"""
    
    def __init__(self, noaa_token: str, openweather_api_key: str):
        self.noaa_token = noaa_token
        self.openweather_api_key = openweather_api_key
        self.noaa_base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        
        # Initialize caching
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = 3600  # 1 hour in seconds
        
        # API status monitoring
        self.api_status = {
            "noaa": {"last_success": None, "consecutive_failures": 0},
            "openweather": {"last_success": None, "consecutive_failures": 0}
        }
        
        # Region coordinates mapping - Updated with stations that have data
        self.regions = {
            "New York, USA": {"lat": 40.7128, "lon": -74.0060, "station_id": "GHCND:USW00094728"},
            "London, UK": {"lat": 51.5074, "lon": -0.1278, "station_id": "GHCND:UKM00003772"},
            "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "station_id": "GHCND:JA000047668"},
            "Sydney, Australia": {"lat": -33.8688, "lon": 151.2093, "station_id": "GHCND:ASN00066062"},
            "Mumbai, India": {"lat": 19.0760, "lon": 72.8777, "station_id": "GHCND:USW00094728"},  # Use NY station as fallback
            "São Paulo, Brazil": {"lat": -23.5505, "lon": -46.6333, "station_id": "GHCND:BR000087898"}
        }
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if not expired"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check if cache is still valid
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < self.cache_duration:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error reading cache: {e}")
        
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Store data in cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error writing cache: {e}")
    
    def _update_api_status(self, api_name: str, success: bool) -> None:
        """Update API status monitoring"""
        if success:
            self.api_status[api_name]["last_success"] = datetime.now()
            self.api_status[api_name]["consecutive_failures"] = 0
        else:
            self.api_status[api_name]["consecutive_failures"] += 1
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status"""
        return {
            "noaa": {
                "status": "healthy" if self.api_status["noaa"]["consecutive_failures"] < 3 else "degraded",
                "last_success": self.api_status["noaa"]["last_success"],
                "failures": self.api_status["noaa"]["consecutive_failures"]
            },
            "openweather": {
                "status": "healthy" if self.api_status["openweather"]["consecutive_failures"] < 3 else "degraded",
                "last_success": self.api_status["openweather"]["last_success"],
                "failures": self.api_status["openweather"]["consecutive_failures"]
            }
        }
    
    def clear_cache(self):
        """Clear all cached data to force fresh API calls"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print("Cache cleared successfully")
    
    def fetch_noaa_historical_data(self, region_name: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Fetch historical climate data from NOAA API with caching
        
        Note: NOAA API has a 1-year limit, so we fetch data year by year and combine
        """
        
        # Check cache first
        cache_key = self._get_cache_key("noaa_historical", region_name, start_year, end_year)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached NOAA data for {region_name} ({start_year}-{end_year})")
            self._update_api_status("noaa", True)
            return cached_data
        
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found")
        
        region = self.regions[region_name]
        station_id = region["station_id"]
        
        all_data = []
        api_success = False
        
        # Fetch data year by year to respect NOAA's 1-year limit
        for year in range(start_year, end_year + 1):
            try:
                print(f"Fetching real-time NOAA data for {year}...")
                
                # NOAA API parameters for single year
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                # Fetch temperature data
                temp_data = self._fetch_noaa_dataset(station_id, "TMAX", start_date, end_date)
                # Fetch precipitation data
                precip_data = self._fetch_noaa_dataset(station_id, "PRCP", start_date, end_date)
                
                # Combine datasets for this year
                year_data = self._combine_noaa_datasets(temp_data, precip_data)
                
                if not year_data.empty:
                    all_data.append(year_data)
                    api_success = True
                    
            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                continue
        
        # Update API status
        self._update_api_status("noaa", api_success)
        
        # Combine all years data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            
            # Cache the result
            self._cache_data(cache_key, combined_data)
            print(f"✅ Fetched and cached real NOAA data for {region_name} ({len(combined_data)} records)")
            
            return combined_data
        else:
            # Return mock data if no real data is available
            print("⚠️  No real NOAA data available, using enhanced mock data")
            mock_data = self._generate_mock_data("COMBINED", f"{start_year}-01-01", f"{end_year}-12-31")
            return mock_data
    
    def fetch_historical_data_for_date_range(self, region_name: str, start_month: int, start_day: int, 
                                           end_month: int, end_day: int, years_back: int = 4) -> pd.DataFrame:
        """Fetch historical data for specific date ranges across multiple years
        
        Args:
            region_name: Name of the region
            start_month: Starting month (1-12)
            start_day: Starting day (1-31)
            end_month: Ending month (1-12)
            end_day: Ending day (1-31)
            years_back: Number of years to look back from current year
        
        Returns:
            DataFrame with historical data for the specified date ranges across years
        """
        
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found")
        
        current_year = datetime.now().year
        all_data = []
        
        for year in range(current_year - years_back, current_year):
            try:
                # Handle date range that spans across months or years
                if end_month >= start_month:
                    # Same year date range
                    start_date = f"{year}-{start_month:02d}-{start_day:02d}"
                    end_date = f"{year}-{end_month:02d}-{end_day:02d}"
                    
                    year_data = self.fetch_noaa_historical_data(region_name, year, year)
                    
                    if not year_data.empty:
                        # Filter to specific date range
                        year_data['date'] = pd.to_datetime(year_data['date'])
                        mask = (
                            (year_data['date'].dt.month >= start_month) & 
                            (year_data['date'].dt.month <= end_month)
                        )
                        
                        if start_month == end_month:
                            # Same month - filter by day
                            mask = mask & (
                                (year_data['date'].dt.day >= start_day) & 
                                (year_data['date'].dt.day <= end_day)
                            )
                        elif start_month == end_month - 1:
                            # Adjacent months - more complex filtering
                            mask = (
                                ((year_data['date'].dt.month == start_month) & 
                                 (year_data['date'].dt.day >= start_day)) |
                                ((year_data['date'].dt.month == end_month) & 
                                 (year_data['date'].dt.day <= end_day))
                            )
                        
                        filtered_data = year_data[mask].copy()
                        if not filtered_data.empty:
                            filtered_data['year'] = year
                            all_data.append(filtered_data)
                
            except Exception as e:
                print(f"Error fetching data for year {year}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            return combined_df
        else:
            # Return mock data if no real data is available
            return self._generate_mock_date_range_data(region_name, start_month, start_day, 
                                                     end_month, end_day, years_back)
    
    def _fetch_noaa_dataset(self, station_id: str, datatype: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch specific dataset from NOAA"""
        
        headers = {"token": self.noaa_token}
        
        # Get dataset ID
        dataset_params = {
            "datasetid": "GHCND",
            "datatypeid": datatype,
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "limit": 1000
        }
        
        try:
            response = requests.get(
                f"{self.noaa_base_url}/data",
                headers=headers,
                params=dataset_params
            )
            
            if response.status_code != 200:
                print(f"NOAA API Error: {response.status_code} - {response.text}")
                return self._generate_mock_data(datatype, start_date, end_date)
            
            data = response.json()
            
            if "results" not in data or len(data["results"]) == 0:
                print(f"No NOAA data found for {datatype}")
                return self._generate_mock_data(datatype, start_date, end_date)
            
            # Convert to DataFrame
            results = data["results"]
            df_data = []
            
            for result in results:
                try:
                    value = float(result["value"])
                    if datatype == "TMAX":
                        # Convert from tenths of degrees C to degrees C
                        value = value / 10.0
                    elif datatype == "PRCP":
                        # Convert from tenths of mm to mm
                        value = value / 10.0
                    
                    df_data.append({
                        "date": result["date"],
                        datatype: value
                    })
                except (ValueError, KeyError) as e:
                    print(f"Error processing NOAA data point: {e}")
                    continue
            
            df = pd.DataFrame(df_data)
            df["date"] = pd.to_datetime(df["date"])
            
            return df
            
        except Exception as e:
            print(f"Error fetching NOAA data: {e}")
            return self._generate_mock_data(datatype, start_date, end_date)
    
    def _combine_noaa_datasets(self, temp_data: pd.DataFrame, precip_data: pd.DataFrame) -> pd.DataFrame:
        """Combine temperature and precipitation data"""
        
        if temp_data.empty and precip_data.empty:
            return pd.DataFrame()
        
        # Merge datasets on date
        if not temp_data.empty and not precip_data.empty:
            combined = pd.merge(temp_data, precip_data, on="date", how="outer")
        elif not temp_data.empty:
            combined = temp_data.copy()
        else:
            combined = precip_data.copy()
        
        # Fill missing values
        if "TMAX" in combined.columns:
            combined["temperature"] = combined["TMAX"].ffill().bfill()
        
        if "PRCP" in combined.columns:
            combined["rainfall"] = combined["PRCP"].fillna(0)
        
        # Add CO2 data (estimated based on trends and global averages)
        if not combined.empty:
            combined["co2"] = self._estimate_co2(combined["date"])
        
        # Select and rename columns
        result_columns = ["date"]
        if "temperature" in combined.columns:
            result_columns.append("temperature")
        if "rainfall" in combined.columns:
            result_columns.append("rainfall")
        if "co2" in combined.columns:
            result_columns.append("co2")
        
        return combined[result_columns].dropna()
    
    def _estimate_co2(self, dates: pd.Series) -> pd.Series:
        """Estimate CO2 levels based on date"""
        
        # Base CO2 level and annual increase (based on Mauna Loa data)
        base_co2 = 400  # ppm
        annual_increase = 2.4  # ppm per year
        
        # Calculate years since 2020
        base_date = pd.Timestamp("2020-01-01")
        years_diff = (dates - base_date).dt.days / 365.25
        
        # Calculate CO2 levels with seasonal variation
        co2_levels = base_co2 + (years_diff * annual_increase)
        
        # Add small seasonal variation
        day_of_year = dates.dt.dayofyear
        seasonal_variation = 2 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
        co2_levels += seasonal_variation
        
        return co2_levels
    
    def fetch_openweather_current_data(self, region_name: str) -> Dict[str, Any]:
        """Fetch current weather data from OpenWeatherMap with caching"""
        
        # Check cache first (shorter cache for current data - 10 minutes)
        cache_key = self._get_cache_key("owm_current", region_name)
        cached_data = self._get_cached_data(cache_key)
        
        # Use shorter cache duration for current weather (10 minutes)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 600:  # 10 minutes for current weather
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)
                    print(f"Using cached current weather for {region_name}")
                    self._update_api_status("openweather", True)
                    return cached_result
                except Exception:
                    pass
        
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found")
        
        region = self.regions[region_name]
        lat, lon = region["lat"], region["lon"]
        
        # Current weather
        current_url = f"{self.openweather_base_url}/weather"
        current_params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_api_key,
            "units": "metric"
        }
        
        try:
            print(f"Fetching real-time current weather for {region_name}...")
            response = requests.get(current_url, params=current_params, timeout=10)
            
            if response.status_code != 200:
                print(f"OpenWeatherMap API Error: {response.status_code}")
                self._update_api_status("openweather", False)
                return self._get_mock_current_data(region_name)
            
            data = response.json()
            self._update_api_status("openweather", True)
            
            result = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "description": data["weather"][0]["description"],
                "location": data["name"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"Error caching current weather: {e}")
            
            print(f"✅ Fetched real-time current weather: {result['temperature']}°C, {result['description']}")
            return result
            
        except Exception as e:
            print(f"Error fetching OpenWeatherMap data: {e}")
            self._update_api_status("openweather", False)
            return self._get_mock_current_data(region_name)
    
    def fetch_openweather_historical_data(self, region_name: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch historical weather data from OpenWeatherMap"""
        
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found")
        
        region = self.regions[region_name]
        lat, lon = region["lat"], region["lon"]
        
        # Generate timestamps for the last N days
        end_time = int(time.time())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        historical_data = []
        
        # OpenWeatherMap historical API requires multiple calls (one per day)
        for timestamp in range(start_time, end_time, 24 * 60 * 60):
            dt = datetime.fromtimestamp(timestamp)
            
            historical_url = f"{self.openweather_base_url}/onecall/timemachine"
            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": self.openweather_api_key,
                "units": "metric"
            }
            
            try:
                response = requests.get(historical_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "current" in data:
                        current = data["current"]
                        historical_data.append({
                            "date": dt,
                            "temperature": current["temp"],
                            "humidity": current["humidity"],
                            "pressure": current["pressure"],
                            "wind_speed": current.get("wind_speed", 0),
                            "rainfall": current.get("rain", {}).get("1h", 0) * 24  # Convert hourly to daily
                        })
                
                # Rate limiting - avoid hitting API limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching historical data for {dt}: {e}")
                continue
        
        if historical_data:
            df = pd.DataFrame(historical_data)
            # Add estimated CO2 data
            df["co2"] = self._estimate_co2(df["date"])
            return df
        else:
            return self._generate_mock_historical_data(region_name, days_back)
    
    def fetch_openweather_forecast_data(self, region_name: str, days: int = 7) -> pd.DataFrame:
        """Fetch forecast data from OpenWeatherMap"""
        
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found")
        
        region = self.regions[region_name]
        lat, lon = region["lat"], region["lon"]
        
        # 5-day forecast
        forecast_url = f"{self.openweather_base_url}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(forecast_url, params=params)
            
            if response.status_code != 200:
                print(f"OpenWeatherMap Forecast Error: {response.status_code}")
                return self._generate_mock_forecast_data(region_name, days)
            
            data = response.json()
            
            forecast_data = []
            for item in data["list"][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                dt = datetime.fromtimestamp(item["dt"])
                
                forecast_data.append({
                    "date": dt,
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item.get("wind", {}).get("speed", 0),
                    "rainfall": item.get("rain", {}).get("3h", 0) * 8  # Convert 3-hourly to daily
                })
            
            if forecast_data:
                df = pd.DataFrame(forecast_data)
                # Aggregate to daily data
                daily_df = df.groupby(df["date"].dt.date).agg({
                    "temperature": "mean",
                    "humidity": "mean",
                    "pressure": "mean",
                    "wind_speed": "mean",
                    "rainfall": "sum"
                }).reset_index()
                
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                # Add estimated CO2 data
                daily_df["co2"] = self._estimate_co2(daily_df["date"])
                
                return daily_df
            else:
                return self._generate_mock_forecast_data(region_name, days)
                
        except Exception as e:
            print(f"Error fetching forecast data: {e}")
            return self._generate_mock_forecast_data(region_name, days)
    
    def get_climate_data_summary(self, region_name: str) -> Dict[str, Any]:
        """Get comprehensive climate data summary for a region"""
        
        # Fetch current data
        current_data = self.fetch_openweather_current_data(region_name)
        
        # Fetch recent historical data (last 30 days)
        historical_data = self.fetch_openweather_historical_data(region_name, 30)
        
        # Fetch forecast data (next 7 days)
        forecast_data = self.fetch_openweather_forecast_data(region_name, 7)
        
        # Calculate summary statistics
        summary = {
            "current": current_data,
            "historical_stats": self._calculate_historical_stats(historical_data),
            "forecast_stats": self._calculate_forecast_stats(forecast_data),
            "trends": self._calculate_trends(historical_data),
            "risk_assessment": self._assess_climate_risk(historical_data, forecast_data)
        }
        
        return summary
    
    def _calculate_historical_stats(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics from historical data"""
        
        if historical_data.empty:
            return {}
        
        stats = {}
        for column in ["temperature", "rainfall", "co2"]:
            if column in historical_data.columns:
                stats[column] = {
                    "mean": float(historical_data[column].mean()),
                    "min": float(historical_data[column].min()),
                    "max": float(historical_data[column].max()),
                    "std": float(historical_data[column].std()),
                    "trend": self._calculate_trend(historical_data[column], historical_data["date"])
                }
        
        return stats
    
    def _calculate_forecast_stats(self, forecast_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics from forecast data"""
        
        if forecast_data.empty:
            return {}
        
        stats = {}
        for column in ["temperature", "rainfall", "co2"]:
            if column in forecast_data.columns:
                stats[column] = {
                    "mean": float(forecast_data[column].mean()),
                    "min": float(forecast_data[column].min()),
                    "max": float(forecast_data[column].max()),
                    "std": float(forecast_data[column].std())
                }
        
        return stats
    
    def _calculate_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends from historical data"""
        
        if historical_data.empty:
            return {}
        
        trends = {}
        for column in ["temperature", "rainfall", "co2"]:
            if column in historical_data.columns:
                trends[column] = self._calculate_trend(historical_data[column], historical_data["date"])
        
        return trends
    
    def _calculate_trend(self, values: pd.Series, dates: pd.Series) -> Dict[str, float]:
        """Calculate trend for a time series"""
        
        if len(values) < 2:
            return {"slope": 0.0, "direction": "stable", "annual_change": 0.0}
        
        # Remove NaN values
        valid_indices = ~values.isna()
        valid_values = values[valid_indices]
        valid_dates = dates[valid_indices]
        
        if len(valid_values) < 2:
            return {"slope": 0.0, "direction": "stable", "annual_change": 0.0}
        
        # Calculate linear trend
        x = np.arange(len(valid_values))
        slope, intercept = np.polyfit(x, valid_values, 1)
        
        # Calculate annual change
        time_span_days = (valid_dates.iloc[-1] - valid_dates.iloc[0]).days
        if time_span_days > 0:
            annual_change = slope * 365.25 / time_span_days * len(valid_values)
        else:
            annual_change = 0
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "slope": float(slope),
            "direction": direction,
            "annual_change": float(annual_change)
        }
    
    def _assess_climate_risk(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess climate risk based on historical and forecast data"""
        
        risk_factors = {
            "temperature_risk": 0,
            "rainfall_risk": 0,
            "co2_risk": 0,
            "overall_risk": 0,
            "risk_level": "Low"
        }
        
        # Temperature risk assessment
        if not historical_data.empty and "temperature" in historical_data.columns:
            temp_stats = self._calculate_historical_stats(historical_data).get("temperature", {})
            
            # High temperature risk
            if temp_stats.get("mean", 0) > 30:
                risk_factors["temperature_risk"] = 3
            elif temp_stats.get("mean", 0) > 25:
                risk_factors["temperature_risk"] = 2
            elif temp_stats.get("mean", 0) > 20:
                risk_factors["temperature_risk"] = 1
            
            # Rising temperature risk
            temp_trend = self._calculate_trend(historical_data["temperature"], historical_data["date"])
            if temp_trend["annual_change"] > 1.0:
                risk_factors["temperature_risk"] += 2
            elif temp_trend["annual_change"] > 0.5:
                risk_factors["temperature_risk"] += 1
        
        # Rainfall risk assessment
        if not historical_data.empty and "rainfall" in historical_data.columns:
            rain_stats = self._calculate_historical_stats(historical_data).get("rainfall", {})
            
            # Low rainfall risk (drought)
            if rain_stats.get("mean", 0) < 50:
                risk_factors["rainfall_risk"] = 3
            elif rain_stats.get("mean", 0) < 100:
                risk_factors["rainfall_risk"] = 2
            elif rain_stats.get("mean", 0) < 200:
                risk_factors["rainfall_risk"] = 1
            
            # Declining rainfall risk
            rain_trend = self._calculate_trend(historical_data["rainfall"], historical_data["date"])
            if rain_trend["annual_change"] < -20:
                risk_factors["rainfall_risk"] += 2
            elif rain_trend["annual_change"] < -10:
                risk_factors["rainfall_risk"] += 1
        
        # CO2 risk assessment
        if not historical_data.empty and "co2" in historical_data.columns:
            co2_stats = self._calculate_historical_stats(historical_data).get("co2", {})
            
            # High CO2 risk
            if co2_stats.get("mean", 0) > 450:
                risk_factors["co2_risk"] = 3
            elif co2_stats.get("mean", 0) > 420:
                risk_factors["co2_risk"] = 2
            elif co2_stats.get("mean", 0) > 400:
                risk_factors["co2_risk"] = 1
            
            # Rising CO2 risk
            co2_trend = self._calculate_trend(historical_data["co2"], historical_data["date"])
            if co2_trend["annual_change"] > 3.0:
                risk_factors["co2_risk"] += 2
            elif co2_trend["annual_change"] > 1.5:
                risk_factors["co2_risk"] += 1
        
        # Calculate overall risk
        risk_factors["overall_risk"] = (
            risk_factors["temperature_risk"] + 
            risk_factors["rainfall_risk"] + 
            risk_factors["co2_risk"]
        )
        
        # Determine risk level
        if risk_factors["overall_risk"] >= 8:
            risk_factors["risk_level"] = "High"
        elif risk_factors["overall_risk"] >= 5:
            risk_factors["risk_level"] = "Moderate"
        else:
            risk_factors["risk_level"] = "Low"
        
        return risk_factors
    
    def _generate_mock_data(self, datatype: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock data when API fails"""
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        data = {"date": dates}
        
        if datatype == "TMAX":
            # Generate temperature data with seasonal variation
            day_of_year = dates.dayofyear
            base_temp = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
            data["TMAX"] = base_temp + np.random.normal(0, 2, len(dates))
        elif datatype == "PRCP":
            # Generate rainfall data
            base_rain = 50 + 30 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            data["PRCP"] = np.maximum(0, base_rain + np.random.normal(0, 15, len(dates)))
        
        df = pd.DataFrame(data)
        return df
    
    def _generate_mock_historical_data(self, region_name: str, days_back: int) -> pd.DataFrame:
        """Generate mock historical data"""
        
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        # Region-specific base values
        region_data = self.regions[region_name]
        
        data = {
            "date": dates,
            "temperature": region_data["lat"] / 10 + 15 + np.random.normal(0, 3, len(dates)),
            "humidity": np.random.uniform(40, 80, len(dates)),
            "pressure": np.random.uniform(990, 1020, len(dates)),
            "wind_speed": np.random.uniform(5, 25, len(dates)),
            "rainfall": np.random.exponential(10, len(dates))
        }
        
        df = pd.DataFrame(data)
        df["co2"] = self._estimate_co2(df["date"])
        
        return df
    
    def _generate_mock_forecast_data(self, region_name: str, days: int) -> pd.DataFrame:
        """Generate mock forecast data"""
        
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        
        # Region-specific base values
        region_data = self.regions[region_name]
        
        data = {
            "date": dates,
            "temperature": region_data["lat"] / 10 + 15 + np.random.normal(0, 2, len(dates)),
            "humidity": np.random.uniform(40, 80, len(dates)),
            "pressure": np.random.uniform(990, 1020, len(dates)),
            "wind_speed": np.random.uniform(5, 25, len(dates)),
            "rainfall": np.random.exponential(8, len(dates))
        }
        
        df = pd.DataFrame(data)
        df["co2"] = self._estimate_co2(df["date"])
        
        return df
    
    def _get_mock_current_data(self, region_name: str) -> Dict[str, Any]:
        """Generate mock current data"""
        
        region_data = self.regions[region_name]
        
        return {
            "temperature": region_data["lat"] / 10 + 15,
            "humidity": np.random.uniform(40, 80),
            "pressure": np.random.uniform(990, 1020),
            "wind_speed": np.random.uniform(5, 25),
            "description": "clear sky",
            "location": region_name.split(",")[0]
        }
    
    def _generate_mock_date_range_data(self, region_name: str, start_month: int, start_day: int, 
                                     end_month: int, end_day: int, years_back: int) -> pd.DataFrame:
        """Generate mock historical data for specific date ranges"""
        
        region_data = self.regions[region_name]
        current_year = datetime.now().year
        all_data = []
        
        for year in range(current_year - years_back, current_year):
            # Create date range for this year
            try:
                start_date = datetime(year, start_month, start_day)
                end_date = datetime(year, end_month, end_day)
                
                # Handle case where end date is in the next year
                if end_date < start_date:
                    end_date = datetime(year + 1, end_month, end_day)
                
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Generate realistic seasonal data
                day_of_year = dates.dayofyear
                
                # Temperature with seasonal variation
                base_temp = region_data["lat"] / 10 + 15
                seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
                temperature = base_temp + seasonal_temp + np.random.normal(0, 2, len(dates))
                
                # Rainfall with seasonal variation
                base_rain = 60
                seasonal_rain = 30 * np.sin(2 * np.pi * day_of_year / 365.25)
                rainfall = np.maximum(0, base_rain + seasonal_rain + np.random.normal(0, 15, len(dates)))
                
                # CO2 levels
                co2_levels = self._estimate_co2(dates)
                
                year_df = pd.DataFrame({
                    'date': dates,
                    'temperature': temperature,
                    'rainfall': rainfall,
                    'co2': co2_levels,
                    'year': year
                })
                
                all_data.append(year_df)
                
            except ValueError as e:
                # Skip invalid dates (like Feb 29 on non-leap years)
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df.sort_values('date').reset_index(drop=True)
        else:
            return pd.DataFrame()
