import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd

class PerformanceVisualizer:
    @staticmethod
    def create_score_radar_chart(scores: Dict[str, float]) -> go.Figure:
        """Create a radar chart of validation scores."""
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(67, 67, 67)'),
            fillcolor='rgba(67, 67, 67, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Response Quality Metrics"
        )
        return fig
    
    @staticmethod
    def create_metrics_timeline(metrics_history: List[Dict[str, float]]) -> go.Figure:
        """Create a timeline of performance metrics."""
        df = pd.DataFrame(metrics_history)
        
        fig = go.Figure()
        for column in df.columns:
            fig.add_trace(go.Scatter(
                y=df[column],
                name=column,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Query Number",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        return fig
    
    @staticmethod
    def create_confidence_gauge(score: float) -> go.Figure:
        """Create a gauge chart for confidence score."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgray"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(
            title="Response Confidence",
            height=250
        )
        return fig

    @staticmethod
    def create_metrics_summary(metrics: Dict[str, float]) -> go.Figure:
        """Create a summary bar chart of metrics."""
        fig = go.Figure([go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color='rgb(67, 67, 67)'
        )])
        
        fig.update_layout(
            title="Metrics Summary",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        return fig