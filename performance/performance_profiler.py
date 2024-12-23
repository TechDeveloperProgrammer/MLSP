import os
import sys
import time
import psutil
import threading
import multiprocessing
import logging
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import platform
import GPUtil
import resource
import tracemalloc

class PerformanceProfiler:
    """
    Advanced Multi-dimensional Performance Profiling System
    
    Features:
    - Comprehensive system resource monitoring
    - Detailed performance metrics collection
    - Cross-platform support
    - Machine learning-driven performance insights
    - Adaptive optimization recommendations
    """
    
    @dataclass
    class PerformanceMetrics:
        """Comprehensive performance metrics data class"""
        timestamp: float
        cpu_usage: float
        memory_usage: float
        memory_total: float
        memory_available: float
        disk_io_read: float
        disk_io_write: float
        network_io_sent: float
        network_io_recv: float
        gpu_usage: Optional[float] = None
        gpu_memory_usage: Optional[float] = None
        process_count: int = 0
        thread_count: int = 0
        context_switches: int = 0
        python_memory_usage: Optional[float] = None
    
    def __init__(
        self, 
        database_path: str = '/opt/mlsp/performance_metrics.sqlite',
        log_dir: str = '/var/log/mlsp/performance'
    ):
        """
        Initialize Performance Profiler
        
        :param database_path: Path to SQLite metrics database
        :param log_dir: Directory for performance logs
        """
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        # Configuration
        self.database_path = database_path
        self.log_dir = log_dir
        self.os_type = platform.system().lower()
        
        # Logging configuration
        self.logger = logging.getLogger('MLSPPerformanceProfiler')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'performance_profiler.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Performance tracking
        self.is_profiling = False
        self.profiling_thread = None
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for performance metrics"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        memory_total REAL,
                        memory_available REAL,
                        disk_io_read REAL,
                        disk_io_write REAL,
                        network_io_sent REAL,
                        network_io_recv REAL,
                        gpu_usage REAL,
                        gpu_memory_usage REAL,
                        process_count INTEGER,
                        thread_count INTEGER,
                        context_switches INTEGER,
                        python_memory_usage REAL
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """
        Collect comprehensive system performance metrics
        
        :return: PerformanceMetrics instance
        """
        try:
            # CPU Usage
            cpu_usage = psutil.cpu_percent()
            
            # Memory Metrics
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process and Thread Metrics
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter())
            
            # Context Switches (Linux-specific)
            context_switches = 0
            if self.os_type == 'linux':
                with open('/proc/stat', 'r') as f:
                    for line in f:
                        if line.startswith('ctxt'):
                            context_switches = int(line.split()[1])
                            break
            
            # GPU Metrics (if available)
            gpu_usage, gpu_memory_usage = None, None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100
                    gpu_memory_usage = gpu.memoryUsed / gpu.memoryTotal * 100
            except Exception:
                pass
            
            # Python Memory Tracking
            tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return self.PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_total=memory.total,
                memory_available=memory.available,
                disk_io_read=disk_io.read_bytes,
                disk_io_write=disk_io.write_bytes,
                network_io_sent=net_io.bytes_sent,
                network_io_recv=net_io.bytes_recv,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                process_count=process_count,
                thread_count=thread_count,
                context_switches=context_switches,
                python_memory_usage=current
            )
        
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
            return self.PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=0,
                memory_usage=0,
                memory_total=0,
                memory_available=0,
                disk_io_read=0,
                disk_io_write=0,
                network_io_sent=0,
                network_io_recv=0
            )
    
    def _store_metrics(self, metrics: PerformanceMetrics):
        """
        Store performance metrics in SQLite database
        
        :param metrics: Performance metrics to store
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        timestamp, cpu_usage, memory_usage, memory_total, 
                        memory_available, disk_io_read, disk_io_write, 
                        network_io_sent, network_io_recv, gpu_usage, 
                        gpu_memory_usage, process_count, thread_count, 
                        context_switches, python_memory_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.memory_total,
                    metrics.memory_available,
                    metrics.disk_io_read,
                    metrics.disk_io_write,
                    metrics.network_io_sent,
                    metrics.network_io_recv,
                    metrics.gpu_usage,
                    metrics.gpu_memory_usage,
                    metrics.process_count,
                    metrics.thread_count,
                    metrics.context_switches,
                    metrics.python_memory_usage
                ))
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Metrics storage error: {e}")
    
    def _performance_monitoring_loop(self, interval: float = 5.0):
        """
        Continuous performance monitoring loop
        
        :param interval: Monitoring interval in seconds
        """
        while self.is_profiling:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Sleep for specified interval
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                break
    
    def start_profiling(self, interval: float = 5.0):
        """
        Start performance profiling
        
        :param interval: Monitoring interval in seconds
        """
        if not self.is_profiling:
            self.is_profiling = True
            self.profiling_thread = threading.Thread(
                target=self._performance_monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.profiling_thread.start()
            self.logger.info("Performance profiling started")
    
    def stop_profiling(self):
        """Stop performance profiling"""
        if self.is_profiling:
            self.is_profiling = False
            if self.profiling_thread:
                self.profiling_thread.join()
            self.logger.info("Performance profiling stopped")
    
    def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze historical performance metrics
        
        :param hours: Number of hours of historical data to analyze
        :return: Performance analysis report
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.database_path) as conn:
                # Load metrics into pandas DataFrame
                query = f'''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp >= {cutoff_time} 
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn)
                
                # Performance analysis
                analysis = {
                    'cpu': {
                        'average': df['cpu_usage'].mean(),
                        'max': df['cpu_usage'].max(),
                        'min': df['cpu_usage'].min()
                    },
                    'memory': {
                        'average_usage': df['memory_usage'].mean(),
                        'total': df['memory_total'].max(),
                        'available': df['memory_available'].mean()
                    },
                    'disk_io': {
                        'read_total': df['disk_io_read'].max() - df['disk_io_read'].min(),
                        'write_total': df['disk_io_write'].max() - df['disk_io_write'].min()
                    },
                    'network_io': {
                        'sent_total': df['network_io_sent'].max() - df['network_io_sent'].min(),
                        'recv_total': df['network_io_recv'].max() - df['network_io_recv'].min()
                    },
                    'processes': {
                        'average_count': df['process_count'].mean(),
                        'max_count': df['process_count'].max()
                    }
                }
                
                # GPU metrics (if available)
                if 'gpu_usage' in df.columns and df['gpu_usage'].notna().any():
                    analysis['gpu'] = {
                        'average_usage': df['gpu_usage'].mean(),
                        'max_usage': df['gpu_usage'].max()
                    }
                
                return analysis
        
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
            return {}
    
    def generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate performance optimization recommendations
        
        :param analysis: Performance analysis report
        :return: List of optimization recommendations
        """
        recommendations = []
        
        # CPU recommendations
        if analysis.get('cpu', {}).get('average', 0) > 80:
            recommendations.append("High CPU usage detected. Consider optimizing server-side processes.")
        
        # Memory recommendations
        if analysis.get('memory', {}).get('average_usage', 0) > 85:
            recommendations.append("High memory consumption. Increase allocated memory or optimize memory-intensive plugins.")
        
        # Disk I/O recommendations
        if analysis.get('disk_io', {}).get('write_total', 0) > 1_000_000_000:  # 1 GB
            recommendations.append("High disk write activity. Consider using SSD or optimizing file operations.")
        
        # Network I/O recommendations
        if analysis.get('network_io', {}).get('sent_total', 0) > 10_000_000_000:  # 10 GB
            recommendations.append("High network traffic. Review network configuration and bandwidth usage.")
        
        # Process count recommendations
        if analysis.get('processes', {}).get('average_count', 0) > 500:
            recommendations.append("High process count. Review and optimize background processes.")
        
        # GPU recommendations
        if 'gpu' in analysis and analysis['gpu'].get('average_usage', 0) > 70:
            recommendations.append("High GPU usage. Consider optimizing GPU-intensive plugins or tasks.")
        
        return recommendations

def main():
    """Example usage of Performance Profiler"""
    profiler = PerformanceProfiler()
    
    try:
        # Start profiling
        profiler.start_profiling(interval=10.0)
        
        # Run for a while (simulated)
        time.sleep(60)
        
        # Stop profiling
        profiler.stop_profiling()
        
        # Analyze performance
        performance_analysis = profiler.analyze_performance(hours=1)
        print("Performance Analysis:", json.dumps(performance_analysis, indent=2))
        
        # Generate optimization recommendations
        recommendations = profiler.generate_optimization_recommendations(performance_analysis)
        print("\nOptimization Recommendations:")
        for rec in recommendations:
            print(f"- {rec}")
    
    except Exception as e:
        print(f"Performance profiling error: {e}")

if __name__ == '__main__':
    main()
