import os
import requests
import logging
import hashlib
import platform
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

class PluginManager:
    """
    Cross-platform Minecraft server plugin management system
    Supports downloading, installing, and managing plugins
    """
    
    def __init__(self, plugins_dir: str = '/opt/mlsp/plugins'):
        """
        Initialize PluginManager
        
        :param plugins_dir: Directory to store downloaded plugins
        """
        self.plugins_dir = plugins_dir
        self.os_type = platform.system().lower()
        
        # Ensure plugins directory exists
        os.makedirs(plugins_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('MLSPPluginManager')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @dataclass
    class Plugin:
        """Plugin metadata and management information"""
        name: str
        version: str
        download_url: str
        description: Optional[str] = None
        author: Optional[str] = None
        dependencies: Optional[List[str]] = None
        checksum: Optional[str] = None
    
    def download_plugin(self, plugin: Plugin) -> str:
        """
        Download a plugin from its source URL
        
        :param plugin: Plugin metadata
        :return: Path to downloaded plugin file
        """
        try:
            response = requests.get(plugin.download_url, stream=True)
            response.raise_for_status()
            
            # Generate filename
            filename = f"{plugin.name}-{plugin.version}.jar"
            filepath = os.path.join(self.plugins_dir, filename)
            
            # Download and verify checksum
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Optional checksum verification
            if plugin.checksum:
                self._verify_checksum(filepath, plugin.checksum)
            
            self.logger.info(f"Downloaded plugin: {plugin.name} v{plugin.version}")
            return filepath
        
        except requests.RequestException as e:
            self.logger.error(f"Plugin download error for {plugin.name}: {e}")
            raise
    
    def _verify_checksum(self, filepath: str, expected_checksum: str):
        """
        Verify plugin file integrity via checksum
        
        :param filepath: Path to plugin file
        :param expected_checksum: Expected SHA-256 checksum
        """
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        calculated_checksum = file_hash.hexdigest()
        
        if calculated_checksum != expected_checksum:
            os.remove(filepath)
            raise ValueError(f"Checksum verification failed for {filepath}")
    
    def bulk_download_plugins(self, plugins: List[Plugin], max_workers: int = 5):
        """
        Bulk download multiple plugins concurrently
        
        :param plugins: List of plugins to download
        :param max_workers: Maximum number of concurrent downloads
        :return: List of downloaded plugin paths
        """
        downloaded_plugins = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_plugin = {
                executor.submit(self.download_plugin, plugin): plugin 
                for plugin in plugins
            }
            
            for future in as_completed(future_to_plugin):
                plugin = future_to_plugin[future]
                try:
                    downloaded_path = future.result()
                    downloaded_plugins.append(downloaded_path)
                except Exception as e:
                    self.logger.error(f"Failed to download {plugin.name}: {e}")
        
        return downloaded_plugins
    
    def list_installed_plugins(self) -> List[str]:
        """
        List all installed plugins
        
        :return: List of installed plugin filenames
        """
        return [f for f in os.listdir(self.plugins_dir) if f.endswith('.jar')]
    
    def remove_plugin(self, plugin_name: str):
        """
        Remove a specific plugin
        
        :param plugin_name: Name of the plugin to remove
        """
        plugin_path = os.path.join(self.plugins_dir, plugin_name)
        
        try:
            os.remove(plugin_path)
            self.logger.info(f"Removed plugin: {plugin_name}")
        except FileNotFoundError:
            self.logger.warning(f"Plugin not found: {plugin_name}")
        except PermissionError:
            self.logger.error(f"Permission denied when removing {plugin_name}")
            raise

def main():
    """Example usage of the PluginManager"""
    plugin_manager = PluginManager()
    
    # Example plugins
    plugins = [
        PluginManager.Plugin(
            name='essentials',
            version='2.20.1',
            download_url='https://example.com/essentials.jar',
            description='Core server management plugin'
        ),
        PluginManager.Plugin(
            name='worldedit',
            version='7.2.5',
            download_url='https://example.com/worldedit.jar',
            description='World editing and generation tool'
        )
    ]
    
    try:
        # Bulk download plugins
        downloaded_plugins = plugin_manager.bulk_download_plugins(plugins)
        print("Downloaded Plugins:", downloaded_plugins)
        
        # List installed plugins
        installed_plugins = plugin_manager.list_installed_plugins()
        print("Installed Plugins:", installed_plugins)
    
    except Exception as e:
        print(f"Plugin management error: {e}")

if __name__ == '__main__':
    main()
