"""Memory monitoring utilities."""
"""
Memory Monitor - Complete Implementation
Monitors system memory usage for ECG processing
"""

import gc
import logging
import os
import psutil
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics data class"""
    timestamp: datetime
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_percent: float
    swap_total: int
    swap_used: int
    swap_percent: float


@dataclass
class MemoryAlert:
    """Memory alert data class"""
    timestamp: datetime
    alert_type: str
    threshold: float
    current_value: float
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class MemoryMonitor:
    """System memory monitor with alerts and optimization"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "percent": psutil.virtual_memory().percent
        }

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 100.0  # Default value
    def __init__(
        self,
        check_interval: int = 60,
        memory_threshold: float = 80.0,
        process_threshold: float = 70.0,
        enable_auto_gc: bool = True
    ):
        """
        Initialize memory monitor
        
        Args:
            check_interval: Seconds between memory checks
            memory_threshold: System memory usage threshold (%)
            process_threshold: Process memory usage threshold (%)
            enable_auto_gc: Enable automatic garbage collection
        """
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.process_threshold = process_threshold
        self.enable_auto_gc = enable_auto_gc
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[MemoryAlert], None]] = []
        
        # Statistics
        self._stats_history: List[MemoryStats] = []
        self._alerts_history: List[MemoryAlert] = []
        self._max_history_size = 1000
        
        # Process info
        self._process = psutil.Process(os.getpid())
        
    def start_monitoring(self) -> None:
        """Start background memory monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already active")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                self._stats_history.append(stats)
                
                # Check thresholds
                alerts = self.check_memory_threshold(stats)
                for alert in alerts:
                    self._handle_alert(alert)
                
                # Trim history
                if len(self._stats_history) > self._max_history_size:
                    self._stats_history = self._stats_history[-self._max_history_size:]
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        vm = psutil.virtual_memory()
        
        # Process memory
        process_info = self._process.memory_info()
        process_percent = self._process.memory_percent()
        
        # Swap memory
        swap = psutil.swap_memory()
        
        return MemoryStats(
            timestamp=datetime.utcnow(),
            total_memory=vm.total,
            available_memory=vm.available,
            used_memory=vm.used,
            memory_percent=vm.percent,
            process_memory=process_info.rss,
            process_percent=process_percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent
        )
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory info in dictionary format (legacy support)"""
        stats = self.get_memory_stats()
        return {
            "timestamp": stats.timestamp.isoformat(),
            "system": {
                "total": stats.total_memory,
                "available": stats.available_memory,
                "used": stats.used_memory,
                "percent": stats.memory_percent
            },
            "process": {
                "memory": stats.process_memory,
                "percent": stats.process_percent
            },
            "swap": {
                "total": stats.swap_total,
                "used": stats.swap_used,
                "percent": stats.swap_percent
            }
        }
    
    def check_memory_threshold(
        self,
        stats: Optional[MemoryStats] = None
    ) -> List[MemoryAlert]:
        """Check if memory usage exceeds thresholds"""
        if stats is None:
            stats = self.get_memory_stats()
            
        alerts = []
        
        # Check system memory
        if stats.memory_percent > self.memory_threshold:
            severity = self._get_severity(stats.memory_percent, self.memory_threshold)
            alerts.append(MemoryAlert(
                timestamp=stats.timestamp,
                alert_type="system_memory",
                threshold=self.memory_threshold,
                current_value=stats.memory_percent,
                message=f"System memory usage ({stats.memory_percent:.1f}%) exceeds threshold ({self.memory_threshold}%)",
                severity=severity
            ))
        
        # Check process memory
        if stats.process_percent > self.process_threshold:
            severity = self._get_severity(stats.process_percent, self.process_threshold)
            alerts.append(MemoryAlert(
                timestamp=stats.timestamp,
                alert_type="process_memory",
                threshold=self.process_threshold,
                current_value=stats.process_percent,
                message=f"Process memory usage ({stats.process_percent:.1f}%) exceeds threshold ({self.process_threshold}%)",
                severity=severity
            ))
        
        # Check swap usage
        if stats.swap_percent > 50:
            alerts.append(MemoryAlert(
                timestamp=stats.timestamp,
                alert_type="swap_memory",
                threshold=50.0,
                current_value=stats.swap_percent,
                message=f"High swap usage detected ({stats.swap_percent:.1f}%)",
                severity="high" if stats.swap_percent > 80 else "medium"
            ))
        
        return alerts
    
    def _get_severity(self, current: float, threshold: float) -> str:
        """Determine alert severity based on how much threshold is exceeded"""
        excess = current - threshold
        
        if excess > 20:
            return "critical"
        elif excess > 10:
            return "high"
        elif excess > 5:
            return "medium"
        else:
            return "low"
    
    def _handle_alert(self, alert: MemoryAlert) -> None:
        """Handle memory alert"""
        # Log alert
        log_level = {
            "low": logging.WARNING,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, alert.message)
        
        # Store alert
        self._alerts_history.append(alert)
        if len(self._alerts_history) > self._max_history_size:
            self._alerts_history = self._alerts_history[-self._max_history_size:]
        
        # Execute callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Auto garbage collection for critical alerts
        if self.enable_auto_gc and alert.severity in ["high", "critical"]:
            self.optimize_memory()
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        before_stats = self.get_memory_stats()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get stats after optimization
        after_stats = self.get_memory_stats()
        
        # Calculate freed memory
        freed_system = before_stats.used_memory - after_stats.used_memory
        freed_process = before_stats.process_memory - after_stats.process_memory
        
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "garbage_collected": collected,
            "freed_system_memory": freed_system,
            "freed_process_memory": freed_process,
            "before": {
                "system_percent": before_stats.memory_percent,
                "process_percent": before_stats.process_percent
            },
            "after": {
                "system_percent": after_stats.memory_percent,
                "process_percent": after_stats.process_percent
            }
        }
        
        logger.info(f"Memory optimization completed: {result}")
        return result
    
    def add_alert_callback(self, callback: Callable[[MemoryAlert], None]) -> None:
        """Add callback for memory alerts"""
        self._callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[MemoryAlert], None]) -> None:
        """Remove alert callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_stats_history(
        self,
        minutes: Optional[int] = None
    ) -> List[MemoryStats]:
        """Get memory statistics history"""
        if minutes is None:
            return self._stats_history.copy()
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [s for s in self._stats_history if s.timestamp >= cutoff]
    
    def get_alerts_history(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None
    ) -> List[MemoryAlert]:
        """Get alerts history with optional filtering"""
        alerts = self._alerts_history.copy()
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory monitoring summary"""
        current_stats = self.get_memory_stats()
        
        # Calculate averages from history
        if self._stats_history:
            avg_system = sum(s.memory_percent for s in self._stats_history[-100:]) / len(self._stats_history[-100:])
            avg_process = sum(s.process_percent for s in self._stats_history[-100:]) / len(self._stats_history[-100:])
        else:
            avg_system = current_stats.memory_percent
            avg_process = current_stats.process_percent
        
        # Count alerts by severity
        alert_counts = {}
        for severity in ["low", "medium", "high", "critical"]:
            alert_counts[severity] = len([a for a in self._alerts_history if a.severity == severity])
        
        return {
            "monitoring_active": self._monitoring,
            "current": {
                "system_percent": current_stats.memory_percent,
                "process_percent": current_stats.process_percent,
                "swap_percent": current_stats.swap_percent
            },
            "average": {
                "system_percent": avg_system,
                "process_percent": avg_process
            },
            "thresholds": {
                "system": self.memory_threshold,
                "process": self.process_threshold
            },
            "alerts": {
                "total": len(self._alerts_history),
                "by_severity": alert_counts
            },
            "stats_history_size": len(self._stats_history),
            "monitoring_duration_minutes": (
                (self._stats_history[-1].timestamp - self._stats_history[0].timestamp).total_seconds() / 60
                if len(self._stats_history) > 1 else 0
            )
        }
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory during an operation"""
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            duration = time.time() - start_time
            
            memory_delta = end_stats.used_memory - start_stats.used_memory
            process_delta = end_stats.process_memory - start_stats.process_memory
            
            logger.info(
                f"Operation '{operation_name}' completed in {duration:.2f}s. "
                f"Memory change: system={memory_delta/1024/1024:.1f}MB, "
                f"process={process_delta/1024/1024:.1f}MB"
            )
            
            # Check if operation caused memory spike
            if end_stats.memory_percent > self.memory_threshold:
                logger.warning(
                    f"Operation '{operation_name}' resulted in high memory usage: "
                    f"{end_stats.memory_percent:.1f}%"
                )
    
    def estimate_memory_for_ecg(
        self,
        duration_seconds: float,
        sampling_rate: int = 500,
        channels: int = 12
    ) -> Dict[str, Any]:
        """Estimate memory requirements for ECG processing"""
        # Calculate data size
        samples = int(duration_seconds * sampling_rate)
        
        # Float32 = 4 bytes per sample
        raw_size = samples * channels * 4
        
        # Processing overhead (preprocessing, features, etc.)
        # Typically 3-5x raw size for complete processing
        processing_overhead = 4
        total_estimated = raw_size * processing_overhead
        
        # Check if enough memory available
        current_stats = self.get_memory_stats()
        available = current_stats.available_memory
        can_process = total_estimated < available * 0.5  # Use max 50% of available
        
        return {
            "duration_seconds": duration_seconds,
            "sampling_rate": sampling_rate,
            "channels": channels,
            "samples": samples,
            "raw_size_mb": raw_size / 1024 / 1024,
            "estimated_total_mb": total_estimated / 1024 / 1024,
            "available_memory_mb": available / 1024 / 1024,
            "can_process": can_process,
            "recommendation": (
                "Processing feasible" if can_process else
                "Consider processing in chunks or upgrading system memory"
            )
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
        
    def __del__(self):
        """Cleanup on deletion"""
        if self._monitoring:
            self.stop_monitoring()


# Global instance for convenience
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        monitor = get_memory_monitor()
        with monitor.monitor_operation(func.__name__):
            return func(*args, **kwargs)
    return wrapper
