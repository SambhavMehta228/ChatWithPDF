from typing import Dict, List
from collections import defaultdict
import numpy as np
import time

class PerformanceMetrics:
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.response_times = []
        self.total_queries = 0
        self.successful_queries = 0
    
    def add_query_metrics(self, metrics: Dict[str, float]):
        """Add metrics for a single query."""
        self.total_queries += 1
        if metrics.get('final_score', 0) >= 0.6:
            self.successful_queries += 1
        
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
    
    def add_response_time(self, time_taken: float):
        """Add response time measurement."""
        self.response_times.append(time_taken)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics over all queries."""
        return {
            metric: np.mean(values) if values else 0
            for metric, values in self.metrics_history.items()
        }
    
    def get_success_rate(self) -> float:
        """Calculate query success rate."""
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        return np.mean(self.response_times) if self.response_times else 0
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get comprehensive metrics summary."""
        avg_metrics = self.get_average_metrics()
        return {
            **avg_metrics,
            'success_rate': self.get_success_rate(),
            'average_response_time': self.get_average_response_time(),
            'total_queries': self.total_queries
        }
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get historical metrics for visualization."""
        return dict(self.metrics_history)
    def get_scores(self) -> Dict[str, float]:
        """Get the latest scores for radar chart visualization."""
        if not self.metrics_history:
            return {}
        return {metric: values[-1] for metric, values in self.metrics_history.items()}

    def get_history_for_timeline(self) -> List[Dict[str, float]]:
        """Get historical data for metrics timeline visualization."""
        history_length = len(next(iter(self.metrics_history.values()), []))
        timeline = []
        for i in range(history_length):
            entry = {metric: values[i] for metric, values in self.metrics_history.items()}
            timeline.append(entry)
        return timeline

class QueryTimer:
    def __init__(self):
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time