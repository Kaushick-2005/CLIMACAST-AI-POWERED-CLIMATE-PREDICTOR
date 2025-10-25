import requests
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re

class ClimateAIChatbot:
    """AI Chatbot for climate forecasting and recommendations using Llama 3.2:3b model."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._base_url = None
        self.conversation_history = []
    
    def _get_base_url(self) -> str:
        """Check for a working base URL to accommodate different server versions."""
        if self._base_url:
            return self._base_url

        possible_urls = [self.ollama_url, f"{self.ollama_url}/api"]

        for url in possible_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200 and "ollama" in response.text.lower():
                    self._base_url = url
                    return url
            except requests.RequestException:
                continue

        return self.ollama_url

    def _try_endpoints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Try different API endpoints to find a working one."""
        endpoints = [
            "api/generate",
            "v1/chat/completions",
            "api/chat",
            "generate",
            "chat",
            "complete",
            "api/complete",
        ]

        base_url_to_try = self._get_base_url()
        headers = {"Content-Type": "application/json"}

        last_exc = None
        for ep in endpoints:
            url = f"{base_url_to_try}/{ep.lstrip('/')}"
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code in [405, 404]:
                    last_exc = requests.exceptions.HTTPError(f"{resp.status_code} for url: {url}")
                    continue
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"raw": resp.text}
            except Exception as e:
                last_exc = e
                continue

        raise last_exc if last_exc is not None else Exception("No compatible endpoint responded")

    def _generate_chat_prompt(self, user_question: str, forecast_data: Dict[str, Any], 
                            selected_date: str, region: str) -> str:
        """Generate a dynamic prompt based on user question and forecast data."""
        
        # Extract forecast information for the selected date
        date_forecast = self._extract_date_forecast(forecast_data, selected_date)
        
        prompt = f"""You are a climate AI assistant. A user is asking about climate conditions for a specific date and region.

USER QUESTION: "{user_question}"

SELECTED DATE: {selected_date}
REGION: {region}

FORECAST DATA FOR {selected_date}:
{self._format_forecast_data(date_forecast)}

HISTORICAL CONTEXT:
- Region: {region}
- Date: {selected_date}
- Season: {self._get_season(selected_date)}

INSTRUCTIONS:
1. Answer the user's question directly and specifically
2. Use ONLY the actual forecast data provided above
3. Base your recommendations on the real values, not assumptions
4. Be conversational and helpful
5. If data is missing, say so and explain what you can infer
6. Provide specific, actionable advice based on the actual forecast
7. Do NOT use any predefined responses or templates
8. Make your response unique to this specific date and data

Respond naturally as if you're having a conversation with the user."""

        return prompt

    def _extract_date_forecast(self, forecast_data: Dict[str, Any], selected_date: str) -> Dict[str, Any]:
        """Extract forecast data for a specific date."""
        try:
            target_date = pd.to_datetime(selected_date)
            
            # If forecast_data is a DataFrame
            if isinstance(forecast_data, pd.DataFrame) and 'date' in forecast_data.columns:
                date_data = forecast_data[forecast_data['date'] == target_date]
                if not date_data.empty:
                    return date_data.iloc[0].to_dict()
            
            # If forecast_data is a dictionary with date-based structure
            if isinstance(forecast_data, dict):
                # Look for the closest date in the forecast
                if 'dates' in forecast_data and 'values' in forecast_data:
                    dates = pd.to_datetime(forecast_data['dates'])
                    closest_idx = (dates - target_date).abs().idxmin()
                    return {
                        'date': forecast_data['dates'][closest_idx],
                        'temperature': forecast_data.get('values', [])[closest_idx] if 'temperature' in str(forecast_data.get('variables', [])).lower() else None,
                        'rainfall': forecast_data.get('values', [])[closest_idx] if 'rainfall' in str(forecast_data.get('variables', [])).lower() else None,
                        'co2': forecast_data.get('values', [])[closest_idx] if 'co2' in str(forecast_data.get('variables', [])).lower() else None
                    }
            
            # Fallback: return basic structure
            return {
                'date': selected_date,
                'temperature': 'Data not available',
                'rainfall': 'Data not available', 
                'co2': 'Data not available'
            }
            
        except Exception as e:
            return {
                'date': selected_date,
                'temperature': f'Error extracting data: {str(e)}',
                'rainfall': 'Data not available',
                'co2': 'Data not available'
            }

    def _format_forecast_data(self, date_forecast: Dict[str, Any]) -> str:
        """Format forecast data for the prompt."""
        formatted = f"Date: {date_forecast.get('date', 'Unknown')}\n"
        
        for key, value in date_forecast.items():
            if key != 'date' and value is not None:
                if isinstance(value, (int, float)):
                    formatted += f"- {key.title()}: {value:.2f}\n"
                else:
                    formatted += f"- {key.title()}: {value}\n"
        
        return formatted

    def _get_season(self, date_str: str) -> str:
        """Get season based on date."""
        try:
            date_obj = pd.to_datetime(date_str)
            month = date_obj.month
            
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"
        except:
            return "Unknown"

    def get_chat_response(self, user_question: str, forecast_data: Dict[str, Any], 
                         selected_date: str, region: str) -> Dict[str, Any]:
        """Get AI response for user question."""
        
        prompt = self._generate_chat_prompt(user_question, forecast_data, selected_date, region)
        
        payload_variants = [
            {"model": self.model, "prompt": prompt, "stream": False},
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        ]

        last_err = None
        for payload in payload_variants:
            try:
                result = self._try_endpoints(payload)
                text = None
                
                if isinstance(result, dict):
                    if "response" in result:
                        text = result.get("response")
                    elif "message" in result and isinstance(result["message"], dict):
                        text = result["message"].get("content")
                    elif "choices" in result and isinstance(result["choices"], list) and result["choices"]:
                        c = result["choices"][0]
                        text = c.get("text") or c.get("message", {}).get("content")
                    elif "raw" in result:
                        text = result.get("raw")
                    else:
                        text = json.dumps(result)
                else:
                    text = str(result)

                if text and len(text.strip()) > 10:
                    # Store in conversation history
                    self.conversation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "user_question": user_question,
                        "selected_date": selected_date,
                        "ai_response": text.strip(),
                        "region": region
                    })
                    
                    return {
                        "success": True,
                        "response": text.strip(),
                        "timestamp": datetime.now().isoformat()
                    }

            except Exception as e:
                last_err = e
                continue

        return {
            "success": False,
            "error": f"AI chatbot unavailable. Error: {str(last_err)}",
            "response": "I'm sorry, I'm having trouble connecting to the AI service. Please check if Ollama is running and the llama3.2:3b model is available."
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
