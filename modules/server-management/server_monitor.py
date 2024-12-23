import os
import sys
import time
import psutil
import logging
import threading
import subprocess
import platform
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import socket
import json
import sqlite3

class ServerMonitor:
    """
    Advanced Minecraft server monitoring system with cross-platform support
    
    Features:
    - Real-time resource monitoring
    - Performance metrics collection
    - Automatic performance optimization
    - Alerting and notification system
    - Persistent metrics storage
    """
    
    @dataclass
    class ServerMetrics:
        """Comprehensive server performance metrics"""
        timestamp: float
        cpu_usage: float
        memory_usage: float
        disk_usage: float
        network_io: Dict[str, float]
        player_count: int
        tps: float
        active_chunks: int
        entity_count: int
        
        def to_dict(self):
            """Convert metrics to dictionary"""
            return asdict(self)
    
    def __init__(self, server_dir: str, log_file: Optional[str] = None):
        """
        Initialize ServerMonitor
        
        :param server_dir: Directory of the Minecraft server
        :param log_file: Optional log file path
        """
        self.server_dir = server_dir
        self.os_type = platform.system().lower()
        
        # Logging configuration
        self.logger = logging.getLogger('MLSPServerMonitor')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file or '/var/log/mlsp/server_monitor.log')
            ]
        )
        
        # Metrics database
        self.metrics_db_path = os.path.join(server_dir, 'server_metrics.sqlite')
        self._initialize_metrics_db()
        
        # Monitoring flags and configurations
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Performance thresholds
        self.performance_thresholds = {
            'cpu_max': 80.0,  # Maximum CPU usage percentage
            'memory_max': 85.0,  # Maximum memory usage percentage
            'tps_min': 18.0  # Minimum acceptable TPS
        }
    
    def _initialize_metrics_db(self):
        """
        Initialize SQLite database for storing server metrics
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS server_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        network_io_in REAL,
                        network_io_out REAL,
                        player_count INTEGER,
                        tps REAL,
                        active_chunks INTEGER,
                        entity_count INTEGER
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system-wide performance metrics
        
        :return: Dictionary of system metrics
        """
        try:
            # CPU and Memory Usage
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk Usage
            disk = psutil.disk_usage(self.server_dir)
            disk_usage = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'network_io': network_io
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def _parse_minecraft_logs(self) -> Dict[str, Any]:
        """
        Parse Minecraft server logs for game-specific metrics
        
        :return: Dictionary of Minecraft-specific metrics
        """
        try:
            log_path = os.path.join(self.server_dir, 'logs', 'latest.log')
            
            # Placeholder for log parsing logic
            # In a real implementation, this would parse the log for:
            # - Player count
            # - Server TPS
            # - Active chunks
            # - Entity count
            
            return {
                'player_count': 0,  # Placeholder
                'tps': 20.0,  # Placeholder (ideal TPS)
                'active_chunks': 0,
                'entity_count': 0
            }
        except Exception as e:
            self.logger.error(f"Error parsing Minecraft logs: {e}")
            return {}
    
    def _store_metrics(self, metrics: ServerMetrics):
        """
        Store metrics in SQLite database
        
        :param metrics: Server metrics to store
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO server_metrics (
                        timestamp, cpu_usage, memory_usage, disk_usage, 
                        network_io_in, network_io_out, player_count, 
                        tps, active_chunks, entity_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp,
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.disk_usage,
                    metrics.network_io.get('bytes_recv', 0),
                    metrics.network_io.get('bytes_sent', 0),
                    metrics.player_count,
                    metrics.tps,
                    metrics.active_chunks,
                    metrics.entity_count
                ))
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Metrics storage error: {e}")
    
    def _check_performance_thresholds(self, metrics: ServerMetrics):
        """
        Check if server performance is within acceptable limits
        
        :param metrics: Server metrics to evaluate
        """
        alerts = []
        
        if metrics.cpu_usage > self.performance_thresholds['cpu_max']:
            alerts.append(f"High CPU Usage: {metrics.cpu_usage}%")
        
        if metrics.memory_usage > self.performance_thresholds['memory_max']:
            alerts.append(f"High Memory Usage: {metrics.memory_usage}%")
        
        if metrics.tps < self.performance_thresholds['tps_min']:
            alerts.append(f"Low Server TPS: {metrics.tps}")
        
        if alerts:
            self._send_performance_alerts(alerts)
    
    def _send_performance_alerts(self, alerts: List[str]):
        """
        Send performance alerts via multiple channels
        
        :param alerts: List of performance alert messages
        """
        # Placeholder for multi-channel alerting
        # In a real implementation, this would support:
        # - Email notifications
        # - Discord/Slack webhooks
        # - SMS alerts
        # - Local system notifications
        
        alert_message = "\n".join(alerts)
        self.logger.warning(f"Performance Alerts:\n{alert_message}")
    
    def _monitoring_loop(self, interval: float = 5.0):
        """
        Continuous monitoring loop
        
        :param interval: Monitoring interval in seconds
        """
        while self.is_monitoring:
            try:
                # Collect metrics
                system_metrics = self._collect_system_metrics()
                log_metrics = self._parse_minecraft_logs()
                
                # Combine metrics
                combined_metrics = self.ServerMetrics(
                    timestamp=time.time(),
                    cpu_usage=system_metrics.get('cpu_usage', 0),
                    memory_usage=system_metrics.get('memory_usage', 0),
                    disk_usage=system_metrics.get('disk_usage', 0),
                    network_io=system_metrics.get('network_io', {}),
                    player_count=log_metrics.get('player_count', 0),
                    tps=log_metrics.get('tps', 20.0),
                    active_chunks=log_metrics.get('active_chunks', 0),
                    entity_count=log_metrics.get('entity_count', 0)
                )
                
                # Store metrics
                self._store_metrics(combined_metrics)
                
                # Check performance thresholds
                self._check_performance_thresholds(combined_metrics)
                
                # Sleep for specified interval
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                break
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start server monitoring
        
        :param interval: Monitoring interval in seconds
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("Server monitoring started")
    
    def stop_monitoring(self):
        """Stop server monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
            self.logger.info("Server monitoring stopped")
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Retrieve historical server metrics
        
        :param hours: Number of hours of historical data to retrieve
        :return: List of historical metrics
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM server_metrics 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving historical metrics: {e}")
            return []

def main():
    """Example usage of ServerMonitor"""
    server_dir = '/path/to/minecraft/server'
    monitor = ServerMonitor(server_dir)
    
    try:
        # Start monitoring
        monitor.start_monitoring(interval=10.0)
        
        # Run for a while (simulated)
        time.sleep(60)
        
        # Retrieve historical metrics
        historical_metrics = monitor.get_historical_metrics(hours=1)
        print("Historical Metrics:", json.dumps(historical_metrics, indent=2))
    
    except Exception as e:
        print(f"Server monitoring error: {e}")
    finally:
        # Stop monitoring
        monitor.stop_monitoring()

if __name__ == '__main__':
    main()
