import requests
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re

class ClimateAIChatbot:
    """AI Chatbot for climate forecasting and recommendations using Llama 3.2:3b model."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen:0.5b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._base_url = None
        self.conversation_history = []

    def _build_generation_prompt(self, user_question: str, forecast_data: Dict[str, Any], selected_date: str, region: str) -> str:
        """Single grounded prompt: generate free-form answer, but only for weather/climate topics."""
        date_forecast = self._extract_date_forecast(forecast_data, selected_date)
        forecast_text = self._format_forecast_data(date_forecast)

        recent_context = ""
        if self.conversation_history:
            last_items = self.conversation_history[-3:]
            context_lines = []
            for item in last_items:
                context_lines.append(f"User: {item.get('user_question', '')}")
                context_lines.append(f"Assistant: {item.get('ai_response', '')}")
            recent_context = "\n".join(context_lines)

        prompt = f"""
You are ClimaCast AI assistant.

You must generate a natural, human-like response (not template bullets) based on forecast values.
You are ONLY allowed to answer weather/climate/forecast related questions.

Rules:
1) If user question is weather/climate/forecast related, answer using the provided forecast data for the selected date and region.
2) Be specific and practical. Mention predicted values when useful.
3) Do not use phrases like "As an AI language model".
4) Do not ask user to check external tools unless data is truly missing.
5) If question is NOT related to weather/climate, reply with one short refusal:
   "I can only help with climate and weather forecast questions in ClimaCast."

Region: {region}
Selected date: {selected_date}
Forecast data:
{forecast_text}

Recent conversation (optional):
{recent_context}

Current user question:
{user_question}
""".strip()
        return prompt
    
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
                df = forecast_data.copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])

                # First try exact date match (day-level)
                exact = df[df['date'].dt.date == target_date.date()]
                if not exact.empty:
                    return exact.iloc[0].to_dict()

                # Fallback: nearest available date row
                if not df.empty:
                    nearest_idx = (df['date'] - target_date).abs().idxmin()
                    return df.loc[nearest_idx].to_dict()
            
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

    def _is_generic_response(self, text: str) -> bool:
        """Detect low-value generic responses that ignore user climate context."""
        if not text:
            return True
        t = text.strip().lower()
        generic_markers = [
            "how may i assist",
            "how can i help",
            "hello!",
            "hi there",
            "i'm here to help",
        ]
        return any(m in t for m in generic_markers) or len(t) < 25

    def _looks_broken_or_offtopic(self, text: str) -> bool:
        """Reject clearly bad responses; otherwise allow generated output."""
        if not text:
            return True
        t = text.lower().strip()
        bad_markers = [
            "as an ai language model",
            "consult historical weather data",
            "you can use online tools",
            "cannot provide you with any assistance",
            "i'm sorry, but i cannot provide",
            "no current information available",
            "not possible for me to determine",
            "not possible to determine",
        ]
        return any(m in t for m in bad_markers)

    def _build_minimal_fallback(self, forecast_data: Dict[str, Any], selected_date: str, region: str) -> str:
        """Only used if model call fails completely."""
        row = self._extract_date_forecast(forecast_data, selected_date)
        temp = row.get("temperature", "N/A")
        rain = row.get("rainfall", "N/A")
        co2 = row.get("co2", "N/A")
        return f"Forecast for {region} on {selected_date}: temperature {temp}, rainfall {rain}, CO₂ {co2}."

    def get_chat_response(self, user_question: str, forecast_data: Dict[str, Any], 
                         selected_date: str, region: str) -> Dict[str, Any]:
        """Get AI response for user question."""
        
        prompt = self._build_generation_prompt(user_question, forecast_data, selected_date, region)
        
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
                    if self._is_generic_response(text):
                        # second attempt with stronger instruction instead of preset template
                        stronger_prompt = prompt + "\n\nImportant: Give a direct forecast-grounded answer now with concrete values and actionable guidance."
                        retry_payload = {"model": self.model, "prompt": stronger_prompt, "stream": False}
                        retry_result = self._try_endpoints(retry_payload)
                        if isinstance(retry_result, dict):
                            text = retry_result.get("response") or retry_result.get("raw") or text

                    if self._looks_broken_or_offtopic(text):
                        third_prompt = prompt + "\n\nAnswer in 3-5 lines, reference temperature/rainfall/CO2 values, and directly answer the user intent."
                        third_payload = {"model": self.model, "prompt": third_prompt, "stream": False}
                        third_result = self._try_endpoints(third_payload)
                        if isinstance(third_result, dict):
                            text = third_result.get("response") or third_result.get("raw") or text

                    # Final correction pass for false 'no info' style answers
                    if self._looks_broken_or_offtopic(text):
                        correction_prompt = (
                            prompt +
                            "\n\nYour previous answer incorrectly claimed lack of information. "
                            "You DO have forecast values. Revise now: directly answer the question using the provided values. "
                            "Do not mention data is unavailable unless all variables are literally missing."
                        )
                        correction_payload = {"model": self.model, "prompt": correction_prompt, "stream": False}
                        correction_result = self._try_endpoints(correction_payload)
                        if isinstance(correction_result, dict):
                            text = correction_result.get("response") or correction_result.get("raw") or text

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
            "response": self._build_minimal_fallback(forecast_data, selected_date, region)
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
