"""
Performance Monitoring and Metrics Collection for NitroAGI NEXUS
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics we collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: float
    processes_count: int
    timestamp: datetime


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    active_connections: int
    request_count: int
    error_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    queue_size: int
    timestamp: datetime


class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.app_metrics: deque = deque(maxlen=max_history)
        self.start_time = time.time()
        
        # Network baseline
        self._network_baseline = self._get_network_stats()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        self.counters[name] += value
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        self.histograms[name].append(value)
        
        # Keep only recent values (last 1000)
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)
    
    def time_function(self, name: str, labels: Dict[str, str] = None):
        """Decorator/context manager for timing function execution"""
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.record_histogram(f"{name}_duration", duration, labels)
        
        return timer()
    
    def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network
            network_current = self._get_network_stats()
            network_sent_mb = (network_current['sent'] - self._network_baseline['sent']) / (1024**2)
            network_recv_mb = (network_current['recv'] - self._network_baseline['recv']) / (1024**2)
            
            # Load average (Linux/Mac)
            try:
                load_average = psutil.getloadavg()[0]
            except AttributeError:
                load_average = 0.0  # Windows doesn't have load average
            
            # Process count
            processes_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk.percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                load_average=load_average,
                processes_count=processes_count,
                timestamp=datetime.now()
            )
            
            self.system_metrics.append(metrics)
            
            # Update individual gauge metrics
            self.set_gauge("system_cpu_percent", cpu_percent)
            self.set_gauge("system_memory_percent", memory.percent)
            self.set_gauge("system_disk_percent", disk.percent)
            self.set_gauge("system_load_average", load_average)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _get_network_stats(self) -> Dict[str, int]:
        """Get network I/O stats"""
        try:
            stats = psutil.net_io_counters()
            return {
                'sent': stats.bytes_sent,
                'recv': stats.bytes_recv
            }
        except:
            return {'sent': 0, 'recv': 0}
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics[name]
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'sum': sum(values),
            'latest': values[-1],
            'p50': self._percentile(values, 50),
            'p90': self._percentile(values, 90),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        current_metrics = self.collect_system_metrics()
        
        if not current_metrics:
            return {'status': 'unknown', 'issues': ['Failed to collect metrics']}
        
        issues = []
        warnings = []
        
        # Check critical thresholds
        if current_metrics.cpu_percent > 90:
            issues.append(f"High CPU usage: {current_metrics.cpu_percent:.1f}%")
        elif current_metrics.cpu_percent > 80:
            warnings.append(f"Elevated CPU usage: {current_metrics.cpu_percent:.1f}%")
        
        if current_metrics.memory_percent > 95:
            issues.append(f"Critical memory usage: {current_metrics.memory_percent:.1f}%")
        elif current_metrics.memory_percent > 85:
            warnings.append(f"High memory usage: {current_metrics.memory_percent:.1f}%")
        
        if current_metrics.disk_percent > 95:
            issues.append(f"Critical disk usage: {current_metrics.disk_percent:.1f}%")
        elif current_metrics.disk_percent > 85:
            warnings.append(f"High disk usage: {current_metrics.disk_percent:.1f}%")
        
        # Determine overall status
        if issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'uptime': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        return '\n'.join(lines)


class PerformanceMonitor:
    """High-level performance monitoring service"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_percent': 90.0,
            'error_rate': 5.0,
            'response_time_p95': 2.0
        }
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.collector.collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _check_alerts(self):
        """Check metrics against alert thresholds"""
        if not self.system_metrics:
            return
        
        latest = self.collector.system_metrics[-1]
        alerts = []
        
        if latest.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'value': latest.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
        
        if latest.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'value': latest.memory_percent,
                'threshold': self.alert_thresholds['memory_percent']
            })
        
        if latest.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_high',
                'value': latest.disk_percent,
                'threshold': self.alert_thresholds['disk_percent']
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def record_request(
        self, 
        endpoint: str, 
        method: str, 
        status_code: int, 
        response_time: float
    ):
        """Record API request metrics"""
        labels = {
            'endpoint': endpoint,
            'method': method,
            'status': str(status_code)
        }
        
        # Increment request counter
        self.collector.increment_counter('http_requests_total', labels=labels)
        
        # Record response time
        self.collector.record_histogram('http_request_duration_seconds', response_time, labels)
        
        # Track errors
        if status_code >= 400:
            self.collector.increment_counter('http_errors_total', labels=labels)
    
    def get_performance_report(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # System metrics summary
        system_summary = {}
        if self.collector.system_metrics:
            latest_system = self.collector.system_metrics[-1]
            system_summary = {
                'cpu_percent': latest_system.cpu_percent,
                'memory_percent': latest_system.memory_percent,
                'disk_percent': latest_system.disk_percent,
                'load_average': latest_system.load_average,
                'processes_count': latest_system.processes_count
            }
        
        # Application metrics
        app_metrics = {}
        for metric_name in ['http_requests_total', 'http_errors_total', 'http_request_duration_seconds']:
            summary = self.collector.get_metric_summary(metric_name, duration_minutes)
            if summary:
                app_metrics[metric_name] = summary
        
        # Health status
        health = self.collector.get_health_status()
        
        return {
            'report_duration_minutes': duration_minutes,
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_summary,
            'application_metrics': app_metrics,
            'health_status': health,
            'uptime_seconds': time.time() - self.collector.start_time
        }
    
    @property
    def system_metrics(self):
        """Get system metrics deque"""
        return self.collector.system_metrics
    
    @property
    def metrics(self):
        """Get all metrics"""
        return self.collector.metrics


# Global monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


# Decorator for monitoring function performance
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    monitor.collector.increment_counter(f"{name}_calls_total")
                    return result
                except Exception as e:
                    monitor.collector.increment_counter(f"{name}_errors_total")
                    raise
                finally:
                    duration = time.time() - start_time
                    monitor.collector.record_histogram(f"{name}_duration_seconds", duration)
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    monitor.collector.increment_counter(f"{name}_calls_total")
                    return result
                except Exception as e:
                    monitor.collector.increment_counter(f"{name}_errors_total")
                    raise
                finally:
                    duration = time.time() - start_time
                    monitor.collector.record_histogram(f"{name}_duration_seconds", duration)
            
            return sync_wrapper
    
    return decorator