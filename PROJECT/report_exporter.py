import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import io
import base64
import tempfile
import os

class ReportExporter:
    """Export functionality for graphs and reports"""
    
    def __init__(self):
        self.supported_formats = ["png", "jpg", "pdf", "html", "json"]
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = "png", 
                    width: int = 1200, height: int = 800) -> str:
        """Export chart to various formats"""
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats: {self.supported_formats}")
        
        if format in ["png", "jpg", "pdf"]:
            # For image formats, use kaleido
            try:
                return fig.to_image(format=format, width=width, height=height)
            except Exception as e:
                # Fallback to HTML if kaleido is not available
                print(f"Warning: Could not export as {format}. Falling back to HTML. Error: {e}")
                return self.export_chart(fig, filename, "html", width, height)
        
        elif format == "html":
            return fig.to_html(include_plotlyjs=True, div_id="chart")
        
        elif format == "json":
            # Export chart data as JSON
            chart_data = {
                "data": fig.to_dict()["data"],
                "layout": fig.to_dict()["layout"],
                "export_info": {
                    "filename": filename,
                    "format": format,
                    "exported_at": datetime.now().isoformat()
                }
            }
            return json.dumps(chart_data, indent=2)
        
        return ""
    
    def create_forecast_report(self, forecast_data: Dict[str, Any], region: str, 
                              variable: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Create comprehensive forecast report"""
        
        report = {
            "metadata": {
                "title": f"ClimaCast Forecast Report - {region}",
                "region": region,
                "variable": variable,
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "executive_summary": self._generate_executive_summary(forecast_data, variable),
            "forecast_data": forecast_data,
            "visualizations": self._create_forecast_visualizations(forecast_data, variable),
            "recommendations": self._generate_report_recommendations(forecast_data, variable),
            "appendix": {
                "methodology": "Forecast generated using Prophet time series model",
                "data_sources": "Historical climate data from NOAA/NASA databases",
                "limitations": "Forecasts are probabilistic and subject to uncertainty"
            }
        }
        
        return report
    
    def create_analysis_report(self, analysis_data: Dict[str, Any], region: str, 
                              variable: str, start_year: int, end_year: int) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        
        report = {
            "metadata": {
                "title": f"ClimaCast Historical Analysis Report - {region}",
                "region": region,
                "variable": variable,
                "period": {
                    "start_year": start_year,
                    "end_year": end_year
                },
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "executive_summary": self._generate_analysis_executive_summary(analysis_data, variable),
            "analysis_data": analysis_data,
            "visualizations": self._create_analysis_visualizations(analysis_data, variable),
            "statistical_analysis": self._create_statistical_analysis(analysis_data),
            "recommendations": self._generate_analysis_recommendations(analysis_data, variable),
            "appendix": {
                "methodology": "Historical trend analysis using linear regression and statistical methods",
                "data_sources": "Historical climate data from NOAA/NASA databases",
                "time_period": f"{start_year} to {end_year}"
            }
        }
        
        return report
    
    def _generate_executive_summary(self, forecast_data: Dict[str, Any], variable: str) -> str:
        """Generate executive summary for forecast report"""
        
        if not forecast_data.get("values"):
            return "Insufficient data available for forecast analysis."
        
        values = forecast_data["values"]
        avg_value = np.mean(values)
        trend = forecast_data.get("insights", {}).get("trend_per_year", 0)
        
        unit_map = {
            "temperature": "°C",
            "rainfall": "mm",
            "co2": "ppm"
        }
        
        unit = unit_map.get(variable, "units")
        
        summary = f"Forecast analysis for {variable} shows an average value of {avg_value:.1f} {unit} "
        summary += f"over the forecast period. "
        
        if abs(trend) > 0.1:
            if trend > 0:
                summary += f"The data indicates an increasing trend of {trend:.2f} {unit} per year, "
                summary += "suggesting potential climate impacts that require attention."
            else:
                summary += f"The data indicates a decreasing trend of {abs(trend):.2f} {unit} per year, "
                summary += "which may have significant implications for the region."
        else:
            summary += "The forecast shows relatively stable conditions with minimal trend variation."
        
        return summary
    
    def _generate_analysis_executive_summary(self, analysis_data: Dict[str, Any], variable: str) -> str:
        """Generate executive summary for analysis report"""
        
        stats = analysis_data.get("statistics", {})
        insights = analysis_data.get("insights", [])
        
        mean_value = stats.get("mean", 0)
        annual_trend = stats.get("annual_trend", 0)
        
        unit_map = {
            "temperature": "°C",
            "rainfall": "mm",
            "co2": "ppm"
        }
        
        unit = unit_map.get(variable, "units")
        
        summary = f"Historical analysis of {variable} data reveals an average value of {mean_value:.1f} {unit}. "
        
        if abs(annual_trend) > 0.1:
            if annual_trend > 0:
                summary += f"The analysis shows a significant increasing trend of {annual_trend:.2f} {unit} per year, "
                summary += "indicating potential long-term climate change impacts."
            else:
                summary += f"The analysis shows a significant decreasing trend of {abs(annual_trend):.2f} {unit} per year, "
                summary += "which may have substantial regional implications."
        else:
            summary += "The historical data shows relatively stable patterns over the analysis period."
        
        if insights:
            summary += f" Key findings include: {', '.join(insights[:3])}."
        
        return summary
    
    def _create_forecast_visualizations(self, forecast_data: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
        """Create forecast visualizations"""
        
        if not forecast_data.get("dates") or not forecast_data.get("values"):
            return []
        
        dates = forecast_data["dates"]
        values = forecast_data["values"]
        confidence_lower = forecast_data.get("confidence_lower", [])
        confidence_upper = forecast_data.get("confidence_upper", [])
        
        visualizations = []
        
        # Main forecast chart
        fig = go.Figure()
        
        # Add main forecast line
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=f'Forecast ({variable})',
            line=dict(color=self._get_variable_color(variable), width=3),
            hovertemplate=f'%{{y:.1f}} {self._get_variable_unit(variable)}<extra></extra>'
        ))
        
        # Add confidence intervals if available
        if confidence_lower and confidence_upper:
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_upper,
                mode='lines',
                name='Upper Confidence',
                line=dict(color=self._get_variable_color(variable), width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_lower,
                mode='lines',
                name='Lower Confidence',
                line=dict(color=self._get_variable_color(variable), width=1, dash='dash'),
                fill='tonexty',
                fillcolor=f'rgba{self._get_variable_rgba(variable, 0.2)}',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'{variable.title()} Forecast',
            xaxis_title='Date',
            yaxis_title=f'{variable.title()} ({self._get_variable_unit(variable)})',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        visualizations.append({
            "type": "forecast_chart",
            "figure": fig,
            "description": f"Main forecast chart for {variable} with confidence intervals"
        })
        
        # Trend analysis chart
        if len(values) > 1:
            trend_fig = go.Figure()
            
            # Calculate trend line
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            
            trend_fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Actual Forecast',
                line=dict(color=self._get_variable_color(variable), width=2),
                hovertemplate=f'%{{y:.1f}} {self._get_variable_unit(variable)}<extra></extra>'
            ))
            
            trend_fig.add_trace(go.Scatter(
                x=dates,
                y=p(x),
                mode='lines',
                name='Trend Line',
                line=dict(color='orange', width=2, dash='dash'),
                hoverinfo='skip'
            ))
            
            trend_fig.update_layout(
                title=f'{variable.title()} Trend Analysis',
                xaxis_title='Date',
                yaxis_title=f'{variable.title()} ({self._get_variable_unit(variable)})',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            visualizations.append({
                "type": "trend_chart",
                "figure": trend_fig,
                "description": f"Trend analysis showing the overall direction of {variable} forecast"
            })
        
        return visualizations
    
    def _create_analysis_visualizations(self, analysis_data: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
        """Create analysis visualizations"""
        
        if not analysis_data.get("dates") or not analysis_data.get("values"):
            return []
        
        dates = analysis_data["dates"]
        values = analysis_data["values"]
        
        visualizations = []
        
        # Main historical chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name=f'Historical {variable}',
            line=dict(color=self._get_variable_color(variable), width=3),
            fill='tozeroy',
            fillcolor=f'rgba{self._get_variable_rgba(variable, 0.3)}',
            hovertemplate=f'%{{y:.1f}} {self._get_variable_unit(variable)}<extra></extra>'
        ))
        
        # Add trend line
        if len(values) > 1:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=p(x),
                mode='lines',
                name='Trend Line',
                line=dict(color='orange', width=2, dash='dash'),
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'Historical {variable.title()} Analysis',
            xaxis_title='Date',
            yaxis_title=f'{variable.title()} ({self._get_variable_unit(variable)})',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        visualizations.append({
            "type": "historical_chart",
            "figure": fig,
            "description": f"Historical chart showing {variable} trends over time"
        })
        
        # Statistical distribution chart
        if len(values) > 10:
            stats_fig = go.Figure()
            
            stats_fig.add_trace(go.Histogram(
                x=values,
                nbinsx=30,
                name=f'{variable} Distribution',
                marker_color=self._get_variable_color(variable),
                opacity=0.7
            ))
            
            stats_fig.update_layout(
                title=f'{variable.title()} Distribution',
                xaxis_title=f'{variable.title()} ({self._get_variable_unit(variable)})',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            
            visualizations.append({
                "type": "distribution_chart",
                "figure": stats_fig,
                "description": f"Distribution of {variable} values over the analysis period"
            })
        
        return visualizations
    
    def _create_statistical_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create statistical analysis section"""
        
        stats = analysis_data.get("statistics", {})
        
        statistical_analysis = {
            "descriptive_statistics": {
                "mean": stats.get("mean", 0),
                "median": stats.get("median", 0),
                "minimum": stats.get("min", 0),
                "maximum": stats.get("max", 0),
                "standard_deviation": stats.get("std", 0),
                "data_points": stats.get("data_points", 0)
            },
            "trend_analysis": {
                "annual_trend": stats.get("annual_trend", 0),
                "trend_direction": "increasing" if stats.get("annual_trend", 0) > 0 else "decreasing" if stats.get("annual_trend", 0) < 0 else "stable",
                "trend_significance": "significant" if abs(stats.get("annual_trend", 0)) > 0.1 else "minimal"
            },
            "variability_analysis": {
                "coefficient_of_variation": stats.get("std", 0) / stats.get("mean", 1) if stats.get("mean", 0) != 0 else 0,
                "range": stats.get("max", 0) - stats.get("min", 0),
                "variability_level": "high" if stats.get("std", 0) / stats.get("mean", 1) > 0.3 else "moderate" if stats.get("std", 0) / stats.get("mean", 1) > 0.1 else "low"
            }
        }
        
        return statistical_analysis
    
    def _generate_report_recommendations(self, forecast_data: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
        """Generate recommendations for forecast report"""
        
        recommendations = []
        trend = forecast_data.get("insights", {}).get("trend_per_year", 0)
        
        if variable == "temperature":
            if trend > 0.5:
                recommendations.append({
                    "priority": "high",
                    "category": "Public Health",
                    "recommendation": "Implement heat action plans and increase public cooling centers",
                    "timeline": "Immediate"
                })
            elif trend > 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "Urban Planning",
                    "recommendation": "Increase green spaces and heat-resistant infrastructure",
                    "timeline": "Short-term"
                })
        
        elif variable == "rainfall":
            if trend < -10:
                recommendations.append({
                    "priority": "high",
                    "category": "Water Management",
                    "recommendation": "Implement water conservation measures and alternative water sources",
                    "timeline": "Immediate"
                })
            elif trend < 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "Agriculture",
                    "recommendation": "Adopt drought-resistant crops and efficient irrigation systems",
                    "timeline": "Short-term"
                })
        
        elif variable == "co2":
            if trend > 1:
                recommendations.append({
                    "priority": "high",
                    "category": "Climate Action",
                    "recommendation": "Accelerate carbon reduction initiatives and renewable energy transition",
                    "timeline": "Immediate"
                })
            elif trend > 0:
                recommendations.append({
                    "priority": "medium",
                    "category": "Policy",
                    "recommendation": "Strengthen climate policies and monitoring systems",
                    "timeline": "Short-term"
                })
        
        # Default recommendation
        if not recommendations:
            recommendations.append({
                "priority": "low",
                "category": "Monitoring",
                "recommendation": "Continue monitoring climate variables and update forecasts regularly",
                "timeline": "Ongoing"
            })
        
        return recommendations
    
    def _generate_analysis_recommendations(self, analysis_data: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
        """Generate recommendations for analysis report"""
        
        recommendations = []
        insights = analysis_data.get("insights", [])
        stats = analysis_data.get("statistics", {})
        annual_trend = stats.get("annual_trend", 0)
        
        # Generate recommendations based on insights and trends
        if variable == "temperature":
            if annual_trend > 0.5:
                recommendations.append({
                    "priority": "high",
                    "category": "Climate Adaptation",
                    "recommendation": "Develop comprehensive heat adaptation strategies for urban and rural areas",
                    "timeline": "Medium-term"
                })
        elif variable == "rainfall":
            if annual_trend < -20:
                recommendations.append({
                    "priority": "high",
                    "category": "Water Security",
                    "recommendation": "Implement integrated water resource management and drought preparedness",
                    "timeline": "Medium-term"
                })
        elif variable == "co2":
            if annual_trend > 2:
                recommendations.append({
                    "priority": "high",
                    "category": "Emissions Reduction",
                    "recommendation": "Implement aggressive carbon reduction targets and monitoring systems",
                    "timeline": "Medium-term"
                })
        
        # Add insights-based recommendations
        for insight in insights:
            if "alarming" in insight.lower() or "concerning" in insight.lower():
                recommendations.append({
                    "priority": "high",
                    "category": "Risk Management",
                    "recommendation": f"Address concerning trend: {insight}",
                    "timeline": "Short-term"
                })
        
        # Default recommendation
        if not recommendations:
            recommendations.append({
                "priority": "medium",
                "category": "Data Management",
                "recommendation": "Continue data collection and analysis to monitor climate trends",
                "timeline": "Ongoing"
            })
        
        return recommendations
    
    def _get_variable_color(self, variable: str) -> str:
        """Get color for variable"""
        color_map = {
            "temperature": "#ef4444",  # red
            "rainfall": "#3b82f6",     # blue
            "co2": "#6b7280"          # gray
        }
        return color_map.get(variable, "#6b7280")
    
    def _get_variable_unit(self, variable: str) -> str:
        """Get unit for variable"""
        unit_map = {
            "temperature": "°C",
            "rainfall": "mm",
            "co2": "ppm"
        }
        return unit_map.get(variable, "units")
    
    def _get_variable_rgba(self, variable: str, alpha: float) -> str:
        """Get RGBA color for variable"""
        color_map = {
            "temperature": (239, 68, 68),    # red
            "rainfall": (59, 130, 246),      # blue
            "co2": (107, 114, 128)          # gray
        }
        rgb = color_map.get(variable, (107, 114, 128))
        return f"({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    
    def export_report_to_pdf(self, report: Dict[str, Any], filename: str) -> str:
        """Export report to PDF (placeholder implementation)"""
        
        # This would typically use a library like ReportLab or WeasyPrint
        # For now, we'll return a JSON representation
        pdf_content = json.dumps(report, indent=2)
        
        # In a real implementation, this would generate an actual PDF
        return pdf_content
    
    def export_report_to_html(self, report: Dict[str, Any], filename: str) -> str:
        """Export report to HTML"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['metadata']['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .recommendation {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745; }}
                .high-priority {{ border-left-color: #dc3545; }}
                .medium-priority {{ border-left-color: #ffc107; }}
                .low-priority {{ border-left-color: #28a745; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report['metadata']['title']}</h1>
                <p><strong>Generated:</strong> {report['metadata']['generated_at']}</p>
                <p><strong>Region:</strong> {report['metadata']['region']}</p>
                <p><strong>Variable:</strong> {report['metadata']['variable']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report['executive_summary']}</p>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                {self._generate_stats_html(report.get('statistical_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(report.get('recommendations', []))}
            </div>
            
            <div class="section">
                <h2>Appendix</h2>
                <p><strong>Methodology:</strong> {report['appendix']['methodology']}</p>
                <p><strong>Data Sources:</strong> {report['appendix']['data_sources']}</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_stats_html(self, stats: Dict[str, Any]) -> str:
        """Generate HTML for statistics"""
        
        if not stats:
            return "<p>No statistical data available.</p>"
        
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        for key, value in stats.items():
            if isinstance(value, dict):
                html += f"<tr><td colspan='2'><strong>{key.replace('_', ' ').title()}</strong></td></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
            else:
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate HTML for recommendations"""
        
        if not recommendations:
            return "<p>No recommendations available.</p>"
        
        html = ""
        for rec in recommendations:
            priority_class = f"{rec['priority']}-priority"
            html += f"""
            <div class="recommendation {priority_class}">
                <h4>{rec['category']} - {rec['priority'].title()} Priority</h4>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <p><strong>Timeline:</strong> {rec['timeline']}</p>
            </div>
            """
        
        return html