import os
import sys
import platform
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from cryptography.fernet import Fernet

@dataclass
class PlatformConfig:
    """Platform-specific configuration settings"""
    platform: str
    java_path: str
    docker_path: str
    python_path: str
    node_path: str
    postgresql_path: str

@dataclass
class MACPConfig:
    """
    Minecraft Autonomous Creation Platform Core Configuration
    
    Provides system-level configuration and path management
    """
    
    # Platform Identification
    platform_name: str = "Minecraft Autonomous Creation Platform"
    version: str = "0.1.0-alpha"
    
    # System Paths
    base_dir: Path = field(default_factory=lambda: Path.home() / '.macp')
    
    # OS-Specific Paths
    paths: Dict[str, Path] = field(default_factory=lambda: {
        'java': _detect_java_path(),
        'docker': _detect_docker_path(),
        'python': _detect_python_path(),
        'nodejs': _detect_nodejs_path(),
        'postgresql': _detect_postgresql_path()
    })
    
    # Environment Variables
    env_vars: Dict[str, str] = field(default_factory=lambda: {
        'ADMIN_PASSWORD': 'ChaosVT',  # Default admin password
        'TLS_ENCRYPTION': 'enabled',
        'MOD_VALIDATION': 'strict'
    })
    
    # Security Configuration
    security: Dict[str, Any] = field(default_factory=lambda: {
        'tls_enabled': True,
        'download_validation': True,
        'mod_scanning': True
    })
    
    # Logging Configuration
    logging_config: Dict[str, Any] = field(default_factory=lambda: {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    })
    
    def __post_init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self._load_config()
        self._setup_encryption()
        self._validate_environment()
    
    def _setup_logging(self):
        """Configure logging system"""
        log_path = self.base_dir / 'macp.log'
        
        logging.basicConfig(
            level=self.logging_config['level'],
            format=self.logging_config['format'],
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('MACPConfig')
    
    def _load_config(self):
        """Load configuration from file"""
        config_file = self.base_dir / 'config.yaml'
        
        if not config_file.exists():
            self.logger.info("Creating default configuration")
            self._create_default_config()
        else:
            try:
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
                self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config = {
            'platform': self._detect_platform(),
            'api_keys': {
                'modrinth': '',
                'curseforge': '',
                'spigot': ''
            },
            'resources': {
                'max_memory': '4G',
                'cpu_cores': 2,
                'storage_limit': '10G'
            }
        }
        
        config_file = self.base_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def _detect_platform(self):
        """Detect current operating system"""
        import platform
        system = platform.system().lower()
        
        if system == 'linux':
            # Check if it's Archcraft
            try:
                with open('/etc/os-release', 'r') as f:
                    if 'archcraft' in f.read().lower():
                        return 'archcraft'
            except:
                pass
            return 'linux'
        elif system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        else:
            raise ValueError(f"Unsupported platform: {system}")
    
    def _setup_encryption(self):
        """Initialize encryption for sensitive data"""
        key_file = self.base_dir / 'encryption.key'
        
        if not key_file.exists():
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
        else:
            with open(key_file, 'rb') as f:
                key = f.read()
        
        self.cipher_suite = Fernet(key)
    
    def _validate_environment(self):
        """Validate required dependencies and paths"""
        platform = self._detect_platform()
        paths = self.paths
        
        # Validate Java
        java_version = os.popen(f"{paths['java']} -version 2>&1").read()
        if '17' not in java_version:
            self.logger.warning("Java 17 not detected")
        
        # Validate Docker
        docker_version = os.popen(f"{paths['docker']} --version").read()
        if not docker_version:
            self.logger.warning("Docker not detected")
        
        # Validate Python
        python_version = os.popen(f"{paths['python']} --version").read()
        if not python_version:
            self.logger.warning("Python not detected")
    
    def get_platform_config(self):
        """
        Get platform-specific configuration
        
        :return: PlatformConfig object
        """
        return PlatformConfig(
            platform=self._detect_platform(),
            java_path=self.paths['java'],
            docker_path=self.paths['docker'],
            python_path=self.paths['python'],
            node_path=self.paths['nodejs'],
            postgresql_path=self.paths['postgresql']
        )
    
    def update_api_key(self, service: str, key: str):
        """
        Update API key for a service
        
        :param service: Service name
        :param key: API key
        """
        if service not in self.config['api_keys']:
            raise ValueError(f"Unknown service: {service}")
        
        self.config['api_keys'][service] = key
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        config_file = self.base_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def validate_admin_password(self, password: str) -> bool:
        """
        Validate administrative password
        
        :param password: Password to validate
        :return: True if password is valid
        """
        stored_password = self.env_vars['ADMIN_PASSWORD']
        return password == stored_password

def _detect_java_path():
    """Detect Java path"""
    if platform.system().lower() == 'linux':
        return '/usr/lib/jvm/java-17-openjdk/bin/java'
    elif platform.system().lower() == 'darwin':
        return '/usr/local/opt/openjdk@17/bin/java'
    elif platform.system().lower() == 'windows':
        return 'C:\\Program Files\\Java\\jdk-17\\bin\\java.exe'
    else:
        raise ValueError(f"Unsupported platform: {platform.system().lower()}")

def _detect_docker_path():
    """Detect Docker path"""
    if platform.system().lower() == 'linux':
        return '/usr/bin/docker'
    elif platform.system().lower() == 'darwin':
        return '/usr/local/bin/docker'
    elif platform.system().lower() == 'windows':
        return 'C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe'
    else:
        raise ValueError(f"Unsupported platform: {platform.system().lower()}")

def _detect_python_path():
    """Detect Python path"""
    if platform.system().lower() == 'linux':
        return '/usr/bin/python'
    elif platform.system().lower() == 'darwin':
        return '/usr/local/bin/python3'
    elif platform.system().lower() == 'windows':
        return 'C:\\Python39\\python.exe'
    else:
        raise ValueError(f"Unsupported platform: {platform.system().lower()}")

def _detect_nodejs_path():
    """Detect Node.js path"""
    if platform.system().lower() == 'linux':
        return '/usr/bin/node'
    elif platform.system().lower() == 'darwin':
        return '/usr/local/bin/node'
    elif platform.system().lower() == 'windows':
        return 'C:\\Program Files\\nodejs\\node.exe'
    else:
        raise ValueError(f"Unsupported platform: {platform.system().lower()}")

def _detect_postgresql_path():
    """Detect PostgreSQL path"""
    if platform.system().lower() == 'linux':
        return '/usr/bin/postgres'
    elif platform.system().lower() == 'darwin':
        return '/usr/local/bin/postgres'
    elif platform.system().lower() == 'windows':
        return 'C:\\Program Files\\PostgreSQL\\13\\bin\\postgres.exe'
    else:
        raise ValueError(f"Unsupported platform: {platform.system().lower()}")

def main():
    """Initialize and test MACP configuration"""
    config = MACPConfig()
    platform_config = config.get_platform_config()
    print(f"Initialized MACP for platform: {platform_config.platform}")
    print("Configuration loaded successfully")

if __name__ == '__main__':
    main()
