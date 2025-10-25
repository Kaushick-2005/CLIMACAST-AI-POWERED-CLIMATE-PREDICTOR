import requests
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import re


class ClimateAIAdvisor:
    """Lightweight AI client for generating climate insights from a local Ollama-like LLM server.

    The client will attempt several plausible endpoints to be compatible with different
    Ollama versions. It also asks the LLM to return structured JSON when possible so the
    application can render stakeholder-specific recommendations (farmers, policymakers, etc.).
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._base_url = None  # To cache the detected working base URL

    def _generate_prompt(self, forecast_data: Dict[str, Any], historical_context: Dict[str, Any],
                         audiences: Optional[List[str]] = None) -> str:
        audiences = audiences or ["farmers", "policymakers", "public_health"]
        aud_list = ", ".join(audiences)

        # Dynamically build the forecast summary string for all available variables
        forecast_summary_parts = []
        for var in forecast_data.get("variables", []):
            mean_val = forecast_data.get("mean", {}).get(var, 0)
            min_val = forecast_data.get("min", {}).get(var, 0)
            max_val = forecast_data.get("max", {}).get(var, 0)
            trend_val = forecast_data.get("trend", {}).get(var, "stable")
            
            # Ensure values are numeric for formatting
            try:
                mean_str = f"{float(mean_val):.2f}" if mean_val != "N/A" else "N/A"
                min_str = f"{float(min_val):.2f}" if min_val != "N/A" else "N/A"
                max_str = f"{float(max_val):.2f}" if max_val != "N/A" else "N/A"
            except (ValueError, TypeError):
                mean_str = str(mean_val)
                min_str = str(min_val)
                max_str = str(max_val)
            
            forecast_summary_parts.append(
                f"- {var.title()}: Mean={mean_str}, Min={min_str}, Max={max_str}, Trend='{trend_val}'"
            )
        forecast_summary = "\n".join(forecast_summary_parts)
        
        missing_vars = ", ".join(forecast_data.get("missing_variables", [])) or "None"

        # Simplified prompt for better Ollama responses
        means = forecast_data.get("mean", {})
        trends = forecast_data.get("trend", {})
        
        prompt = (
            f"Write a detailed climate analysis for {forecast_data.get('location', 'N/A')}. "
            f"Temperature is {means.get('temperature', 0):.1f}°C with {trends.get('temperature', 'stable')} trend. "
            f"Rainfall is {means.get('rainfall', 0):.1f}mm with {trends.get('rainfall', 'stable')} trend. "
            f"CO2 is {means.get('co2', 0):.1f}ppm with {trends.get('co2', 'stable')} trend. "
            f"Historical context: {historical_context.get('historical_avg', 'N/A')}. "
            f"Anomaly level: {historical_context.get('anomaly', 'N/A')}. "
            "Explain what these trends mean, potential risks, and provide recommendations for farmers, policymakers, and public health. Write in clear paragraphs."
        )

        return prompt

    def _get_base_url(self) -> str:
        """Check for a working base URL to accommodate different server versions."""
        if self._base_url:
            return self._base_url

        # List of possible base URLs to test
        possible_urls = [self.ollama_url, f"{self.ollama_url}/api"]

        for url in possible_urls:
            try:
                # A simple GET request to the root should work for most servers
                response = requests.get(url, timeout=5)
                # We expect a 200 OK and a response that indicates it's an Ollama-like server
                if response.status_code == 200 and "ollama" in response.text.lower():
                    self._base_url = url
                    return url
            except requests.RequestException:
                continue  # Try the next URL

        return self.ollama_url # Fallback to the original URL

    def _try_endpoints(self, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        endpoints = [
            # Prefer Ollama-native and OpenAI chat endpoints; exclude deprecated /v1/completions
            "api/generate",           # Ollama native
            "v1/chat/completions",    # OpenAI-compatible chat
            "api/chat",               # Some variants
            "generate",               # Fallbacks
            "chat",
            "complete",
            "api/complete",
        ]

        base_url_to_try = self._get_base_url()
        headers = {"Content-Type": "application/json"}

        last_exc = None
        for ep in endpoints:
            # Construct URL carefully to avoid double slashes
            url = f"{base_url_to_try}/{ep.lstrip('/')}"
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                if resp.status_code == 405:
                    # Method not allowed; try next known-good endpoint
                    last_exc = requests.exceptions.HTTPError(f"{resp.status_code} Method Not Allowed for url: {url}")
                    continue
                if resp.status_code == 404:
                    # Not found; keep trying other endpoints
                    last_exc = requests.exceptions.HTTPError(f"404 Not Found for url: {url}")
                    continue
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    return {"raw": resp.text}
            except requests.exceptions.Timeout:
                last_exc = requests.exceptions.Timeout(f"Request timed out after {timeout} seconds for url: {url}")
                continue
            except requests.exceptions.ConnectionError:
                last_exc = requests.exceptions.ConnectionError(f"Connection failed for url: {url}")
                continue
            except Exception as e:
                last_exc = e # Store the latest exception and try the next endpoint
                continue

        raise last_exc if last_exc is not None else Exception("No compatible endpoint responded")

    def get_insights(self, forecast_data: Dict[str, Any], historical_context: Dict[str, Any],
                     audiences: Optional[List[str]] = None, timeout: int = 30) -> Dict[str, Any]:
        prompt = self._generate_prompt(forecast_data, historical_context, audiences=audiences)

        payload_variants = [
            # Ollama native generate (plain text only)
            {"model": self.model, "prompt": prompt, "stream": False},
            # OpenAI-compatible chat
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        ]

        last_err = None
        for payload in payload_variants:
            try:
                result = self._try_endpoints(payload, timeout=timeout)
                text = None
                if isinstance(result, dict):
                    if "response" in result:  # Ollama /generate
                        text = result.get("response")
                    elif "message" in result and isinstance(result["message"], dict) and "content" in result["message"]:
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

                if not text:
                    continue

                # Attempt to parse the response as JSON, but handle non-JSON responses gracefully
                parsed = None
                ai_insights = ""
                graph_explanation = ""
                audience_recommendations = {}
                
                try:
                    # Clean the text response to ensure it is a valid JSON string
                    # Models sometimes wrap JSON in ```json ... ``` or add extra text.
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                    else:
                        # If no JSON object is found, assume the whole text is a JSON object
                        parsed = json.loads(text)

                    if isinstance(parsed, dict):
                        # Check if the AI model provided at least some of the expected JSON structure
                        # Handle different field name variations
                        ai_insights = (parsed.get("ai_insights", "") or 
                                     parsed.get("ai_Insights", "") or 
                                     parsed.get("AI_insights", "") or "")
                        graph_explanation = (parsed.get("graph_explanation", "") or 
                                           parsed.get("graph_Explanation", "") or 
                                           parsed.get("Graph_explanation", "") or "")
                        audience_recommendations = (parsed.get("audience_recommendations", {}) or 
                                                  parsed.get("audience_Recommendations", {}) or 
                                                  parsed.get("Audience_recommendations", {}) or {})
                        
                except (json.JSONDecodeError, TypeError):
                    # AI model returned non-JSON response - treat the entire response as insights
                    ai_insights = text.strip()
                    graph_explanation = ""
                    audience_recommendations = {}
                
                # Always generate insights and recommendations based on forecast data
                variables = forecast_data.get("variables", [])
                trends = forecast_data.get("trend", {})
                means = forecast_data.get("mean", {})
                location = forecast_data.get("location", "the region")
                
                # If we have insights from Ollama, use them; otherwise generate fallback
                if ai_insights and ai_insights.strip() and len(ai_insights.strip()) > 20:
                    # Ollama provided insights - use them
                    pass
                else:
                    # Generate comprehensive fallback insights
                    insights_parts = []
                    insights_parts.append(f"Climate Analysis for {location}:")
                    insights_parts.append("")
                    
                    for var in variables:
                        mean_val = means.get(var, 0)
                        trend_info = trends.get(var, "stable")
                        
                        if var == "temperature":
                            if "rising" in trend_info.lower():
                                insights_parts.append(f"Temperature is rising (avg {mean_val:.1f}°C), indicating increasing heat stress and potential impacts on agriculture and energy demand.")
                            elif "decreasing" in trend_info.lower():
                                insights_parts.append(f"Temperature is decreasing (avg {mean_val:.1f}°C), which may affect heating demand and agricultural growing seasons.")
                            else:
                                insights_parts.append(f"Temperature is stable (avg {mean_val:.1f}°C), suggesting consistent climate patterns.")
                        
                        elif var == "rainfall":
                            if "decreasing" in trend_info.lower():
                                insights_parts.append(f"Rainfall is decreasing (avg {mean_val:.1f}mm), indicating drought risk and water scarcity concerns.")
                            elif "increasing" in trend_info.lower():
                                insights_parts.append(f"Rainfall is increasing (avg {mean_val:.1f}mm), suggesting wetter conditions and potential flooding risks.")
                            else:
                                insights_parts.append(f"Rainfall is stable (avg {mean_val:.1f}mm), indicating consistent precipitation patterns.")
                        
                        elif var == "co2":
                            if "rising" in trend_info.lower():
                                insights_parts.append(f"CO₂ levels are rising (avg {mean_val:.1f}ppm), contributing to global warming and requiring carbon reduction strategies.")
                            elif "decreasing" in trend_info.lower():
                                insights_parts.append(f"CO₂ levels are decreasing (avg {mean_val:.1f}ppm), indicating positive environmental progress.")
                            else:
                                insights_parts.append(f"CO₂ levels are stable (avg {mean_val:.1f}ppm), suggesting consistent atmospheric conditions.")
                    
                    insights_parts.append("")
                    insights_parts.append("Monitor for extreme weather events and climate anomalies that could impact local communities.")
                    ai_insights = "\n".join(insights_parts)
                
                # Always generate recommendations based on forecast data
                if not audience_recommendations:
                    audience_recommendations = {}
                    
                    # Generate recommendations based on actual forecast data
                    for var in variables:
                        var_trend = trends.get(var, "stable").lower()
                        var_mean = means.get(var, 0)
                            
                        if var == "temperature":
                            if "rising" in var_trend or var_mean > 25:
                                    audience_recommendations["farmers"] = [
                                        "Switch to heat-resistant crop varieties",
                                        "Implement advanced irrigation systems", 
                                        "Monitor soil moisture levels closely",
                                        "Consider shade netting for sensitive crops"
                                    ]
                                    audience_recommendations["policymakers"] = [
                                        "Activate heat emergency response plans",
                                        "Increase cooling center availability",
                                        "Review urban heat island mitigation strategies",
                                        "Implement heat stress monitoring systems"
                                    ]
                                    audience_recommendations["public_health"] = [
                                        "Issue public heat advisories",
                                        "Prepare for heat-related medical emergencies",
                                        "Monitor vulnerable populations (elderly, children)",
                                        "Develop heat stress prevention protocols"
                                    ]
                            
                            elif var == "rainfall":
                                if "decreasing" in var_trend or var_mean < 10:
                                    if "farmers" not in audience_recommendations:
                                        audience_recommendations["farmers"] = []
                                    if "policymakers" not in audience_recommendations:
                                        audience_recommendations["policymakers"] = []
                                    if "public_health" not in audience_recommendations:
                                        audience_recommendations["public_health"] = []
                                    
                                    audience_recommendations["farmers"].extend([
                                        "Implement water conservation techniques",
                                        "Switch to drought-resistant crop varieties",
                                        "Plan water storage and collection systems",
                                        "Adjust planting schedules for dry conditions"
                                    ])
                                    audience_recommendations["policymakers"].extend([
                                        "Implement water usage restrictions",
                                        "Invest in water infrastructure improvements",
                                        "Develop comprehensive drought management plans",
                                        "Support agricultural water conservation programs"
                                    ])
                                    audience_recommendations["public_health"].extend([
                                        "Monitor water quality and availability",
                                        "Prepare for water scarcity impacts",
                                        "Develop health advisories for water conservation",
                                        "Plan for potential water-related health issues"
                                    ])
                            
                            elif var == "co2":
                                if "rising" in var_trend or var_mean > 400:
                                    if "policymakers" not in audience_recommendations:
                                        audience_recommendations["policymakers"] = []
                                    if "public_health" not in audience_recommendations:
                                        audience_recommendations["public_health"] = []
                                    
                                    audience_recommendations["policymakers"].extend([
                                        "Accelerate carbon reduction policies",
                                        "Invest in renewable energy infrastructure",
                                        "Implement stricter emission controls",
                                        "Support carbon capture and storage initiatives"
                                    ])
                                    audience_recommendations["public_health"].extend([
                                        "Monitor air quality impacts on public health",
                                        "Prepare for respiratory health risks",
                                        "Develop air quality advisories and alerts",
                                        "Plan for increased respiratory emergency cases"
                                    ])
                        
                        # If no specific recommendations generated, provide general ones
                        if not audience_recommendations:
                            audience_recommendations = {
                                "farmers": [
                                    "Monitor climate patterns and trends closely",
                                    "Adapt farming practices to forecast conditions",
                                    "Prepare for variable weather patterns",
                                    "Implement climate-smart agricultural techniques"
                                ],
                                "policymakers": [
                                    "Review and update climate adaptation strategies",
                                    "Monitor forecast accuracy and reliability",
                                    "Prepare for climate variability impacts",
                                    "Support community climate resilience programs"
                                ],
                                "public_health": [
                                    "Monitor health impacts of climate changes",
                                    "Prepare for weather-related health risks",
                                    "Develop adaptive health protocols",
                                    "Plan for climate-related emergency responses"
                                ]
                            }
                    
                    return {
                        "success": True,
                        "ai_insights": ai_insights,
                        "audience_recommendations": audience_recommendations,
                        "raw": text,
                    }
                else:
                    # No insights provided
                    return {
                        "success": False,
                        "error": "AI model did not provide any insights",
                        "ai_insights": "No insights generated from AI response",
                        "audience_recommendations": {},
                        "raw": text,
                    }

            except Exception as e:
                last_err = e
                continue

        # Provide more specific error messages based on the type of error
        if isinstance(last_err, requests.exceptions.Timeout):
            friendly = f"AI request timed out after {timeout} seconds. The model may be overloaded or taking too long to respond. Try again or check if Ollama is running properly."
        elif isinstance(last_err, requests.exceptions.ConnectionError):
            friendly = f"Cannot connect to AI server at {self.ollama_url}. Please ensure Ollama is running and accessible."
        elif isinstance(last_err, requests.exceptions.HTTPError):
            friendly = f"HTTP error from AI server: {str(last_err)}. The server may be misconfigured or the model '{self.model}' may not be available."
        else:
            friendly = (
                f"AI advisor is not available. Ensure Ollama is running on {self.ollama_url}, "
                f"the model '{self.model}' is pulled (e.g., 'ollama pull {self.model}'), and that the app "
                f"can reach a compatible endpoint. Last error: {str(last_err)}"
            )
        
        return {
            "success": False,
            "error": friendly,
            "ai_insights": "Unable to generate insights due to an error.",
            "graph_explanation": "",
            "audience_recommendations": {},
        }

    def analyze_extreme_events(self, forecast_data: pd.DataFrame, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        extreme_events = []

        for variable, threshold in thresholds.items():
            if variable in forecast_data.columns:
                extreme_values = forecast_data[forecast_data[variable] > threshold]
                if not extreme_values.empty:
                    extreme_events.append({
                        "variable": variable,
                        "dates": extreme_values.index.tolist(),
                        "values": extreme_values[variable].tolist(),
                        "threshold": threshold,
                    })

        return extreme_events