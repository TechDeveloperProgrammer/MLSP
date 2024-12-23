import os
import sys
import platform
import subprocess
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mlsp/server_manager.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MLSPServerManager')

@dataclass
class ServerConfig:
    """Comprehensive Minecraft server configuration"""
    name: str
    version: str
    server_type: str
    max_players: int = 20
    difficulty: str = 'normal'
    gamemode: str = 'survival'
    pvp: bool = True
    allow_flight: bool = False
    view_distance: int = 10
    simulation_distance: int = 10
    memory: Dict[str, str] = None
    plugins: List[str] = None
    world_settings: Dict[str, str] = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = {
                'min': '1G',
                'max': '4G'
            }
        if self.plugins is None:
            self.plugins = []
        if self.world_settings is None:
            self.world_settings = {}

class MinecraftServerManager:
    """Cross-platform Minecraft server management"""
    
    def __init__(self, base_dir: str = '/opt/mlsp/servers'):
        self.base_dir = base_dir
        self.os_type = platform.system().lower()
        self._validate_base_dir()
    
    def _validate_base_dir(self):
        """Ensure base directory exists with correct permissions"""
        os.makedirs(self.base_dir, exist_ok=True)
        if self.os_type != 'windows':
            subprocess.run(['chmod', '-R', '755', self.base_dir], check=True)
    
    def create_server(self, config: ServerConfig) -> str:
        """
        Create a new Minecraft server instance
        
        :param config: Server configuration
        :return: Path to created server directory
        """
        server_dir = os.path.join(self.base_dir, config.name)
        os.makedirs(server_dir, exist_ok=True)
        
        # Generate server.properties
        self._generate_server_properties(server_dir, config)
        
        # Download appropriate server jar
        self._download_server_jar(server_dir, config)
        
        # Install plugins
        self._install_plugins(server_dir, config.plugins)
        
        # Generate startup script
        startup_script = self._generate_startup_script(server_dir, config)
        
        logger.info(f"Created server {config.name} at {server_dir}")
        return server_dir
    
    def _download_server_jar(self, server_dir: str, config: ServerConfig):
        """Download appropriate server JAR based on type and version"""
        server_jars = {
            'spigot': f'https://download.spigotmc.org/spigot/{config.version}/spigot-{config.version}.jar',
            'paper': f'https://papermc.io/api/v2/projects/paper/versions/{config.version}/builds/latest/downloads/paper-{config.version}.jar',
            'forge': f'https://files.minecraftforge.net/maven/net/minecraftforge/forge/{config.version}/forge-{config.version}.jar',
            'fabric': f'https://maven.fabricmc.net/net/fabricmc/fabric-loader/{config.version}/fabric-loader-{config.version}.jar'
        }
        
        jar_url = server_jars.get(config.server_type.lower())
        if not jar_url:
            raise ValueError(f"Unsupported server type: {config.server_type}")
        
        # Use platform-specific download method
        if self.os_type == 'windows':
            subprocess.run(['powershell', '-Command', f'Invoke-WebRequest -Uri {jar_url} -OutFile {os.path.join(server_dir, "server.jar")}'], check=True)
        else:
            subprocess.run(['wget', '-O', os.path.join(server_dir, 'server.jar'), jar_url], check=True)
    
    def _generate_server_properties(self, server_dir: str, config: ServerConfig):
        """Generate server.properties file"""
        properties = {
            'server-name': config.name,
            'server-port': '25565',
            'max-players': str(config.max_players),
            'difficulty': config.difficulty,
            'gamemode': config.gamemode,
            'pvp': str(config.pvp).lower(),
            'allow-flight': str(config.allow_flight).lower(),
            'view-distance': str(config.view_distance),
            'simulation-distance': str(config.simulation_distance),
            **config.world_settings
        }
        
        with open(os.path.join(server_dir, 'server.properties'), 'w') as f:
            for key, value in properties.items():
                f.write(f"{key}={value}\n")
    
    def _install_plugins(self, server_dir: str, plugins: List[str]):
        """Install server plugins"""
        plugins_dir = os.path.join(server_dir, 'plugins')
        os.makedirs(plugins_dir, exist_ok=True)
        
        for plugin in plugins:
            # Placeholder for plugin download logic
            logger.info(f"Installing plugin: {plugin}")
    
    def _generate_startup_script(self, server_dir: str, config: ServerConfig):
        """Generate cross-platform startup script"""
        script_path = os.path.join(server_dir, 'start.' + ('bat' if self.os_type == 'windows' else 'sh'))
        
        # Java arguments for memory management
        java_args = [
            f'-Xms{config.memory["min"]}',
            f'-Xmx{config.memory["max"]}',
            '-jar', 'server.jar', 'nogui'
        ]
        
        if self.os_type == 'windows':
            script_content = f"""@echo off
java {' '.join(java_args)}
pause"""
        else:
            script_content = f"""#!/bin/bash
java {' '.join(java_args)}"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable on Unix-like systems
        if self.os_type != 'windows':
            subprocess.run(['chmod', '+x', script_path], check=True)
        
        return script_path
    
    def start_server(self, server_name: str):
        """Start a specific server instance"""
        server_dir = os.path.join(self.base_dir, server_name)
        
        if not os.path.exists(server_dir):
            raise ValueError(f"Server {server_name} does not exist")
        
        startup_script = os.path.join(server_dir, 'start.bat' if self.os_type == 'windows' else 'start.sh')
        
        try:
            subprocess.Popen(
                [startup_script],
                cwd=server_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Started server: {server_name}")
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            raise
    
    def backup_server(self, server_name: str, backup_dir: Optional[str] = None):
        """Create a backup of a server instance"""
        if not backup_dir:
            backup_dir = os.path.join(self.base_dir, server_name, 'backups')
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Use platform-specific compression
        if self.os_type == 'windows':
            subprocess.run(['powershell', 'Compress-Archive', 
                            f'-Path {os.path.join(self.base_dir, server_name)}', 
                            f'-DestinationPath {os.path.join(backup_dir, f"{server_name}_{platform.timestamp()}.zip")}'], 
                           check=True)
        else:
            subprocess.run(['tar', '-czvf', 
                            os.path.join(backup_dir, f"{server_name}_{platform.timestamp()}.tar.gz"),
                            os.path.join(self.base_dir, server_name)], 
                           check=True)
        
        logger.info(f"Backed up server: {server_name}")

def main():
    """Example usage of the Minecraft Server Manager"""
    manager = MinecraftServerManager()
    
    # Example server configuration
    example_config = ServerConfig(
        name='mlsp_survival_server',
        version='1.20.1',
        server_type='paper',
        max_players=50,
        plugins=['essentials', 'worldedit']
    )
    
    try:
        # Create server
        server_path = manager.create_server(example_config)
        print(f"Server created at: {server_path}")
        
        # Start server
        manager.start_server('mlsp_survival_server')
        
        # Backup server
        manager.backup_server('mlsp_survival_server')
    
    except Exception as e:
        logger.error(f"Server management error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
