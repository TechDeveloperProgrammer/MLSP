import os
import yaml
import json
from typing import Dict, Any, Optional
import logging

class ConfigParser:
    """
    Cross-platform configuration parser for Minecraft server configurations
    Supports YAML, JSON, and environment variable configurations
    """
    
    def __init__(self, config_dir: str = '/etc/mlsp/servers'):
        """
        Initialize ConfigParser with a default configuration directory
        
        :param config_dir: Directory containing server configuration files
        """
        self.config_dir = config_dir
        self.logger = logging.getLogger('MLSPConfigParser')
        
        # Ensure configuration directory exists
        os.makedirs(config_dir, exist_ok=True)
    
    def load_config(self, server_name: str, config_type: str = 'yaml') -> Dict[str, Any]:
        """
        Load server configuration from file
        
        :param server_name: Name of the server
        :param config_type: Type of configuration file (yaml or json)
        :return: Parsed configuration dictionary
        """
        config_file_map = {
            'yaml': f'{server_name}.yml',
            'json': f'{server_name}.json'
        }
        
        config_filename = config_file_map.get(config_type.lower())
        if not config_filename:
            raise ValueError(f"Unsupported configuration type: {config_type}")
        
        config_path = os.path.join(self.config_dir, config_filename)
        
        try:
            with open(config_path, 'r') as config_file:
                if config_type.lower() == 'yaml':
                    return yaml.safe_load(config_file)
                else:
                    return json.load(config_file)
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def save_config(self, server_name: str, config: Dict[str, Any], config_type: str = 'yaml'):
        """
        Save server configuration to file
        
        :param server_name: Name of the server
        :param config: Configuration dictionary to save
        :param config_type: Type of configuration file (yaml or json)
        """
        config_file_map = {
            'yaml': f'{server_name}.yml',
            'json': f'{server_name}.json'
        }
        
        config_filename = config_file_map.get(config_type.lower())
        if not config_filename:
            raise ValueError(f"Unsupported configuration type: {config_type}")
        
        config_path = os.path.join(self.config_dir, config_filename)
        
        try:
            with open(config_path, 'w') as config_file:
                if config_type.lower() == 'yaml':
                    yaml.dump(config, config_file, default_flow_style=False)
                else:
                    json.dump(config, config_file, indent=2)
            
            self.logger.info(f"Configuration saved: {config_path}")
        except IOError as e:
            self.logger.error(f"Error saving configuration file: {e}")
            raise
    
    def get_env_config(self, prefix: str = 'MLSP_') -> Dict[str, str]:
        """
        Extract configuration from environment variables
        
        :param prefix: Prefix for environment variables
        :return: Dictionary of environment configurations
        """
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = value
        
        return env_config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries
        
        :param configs: Configuration dictionaries to merge
        :return: Merged configuration dictionary
        """
        merged_config = {}
        for config in configs:
            merged_config.update(config)
        
        return merged_config

def main():
    """Example usage of the ConfigParser"""
    parser = ConfigParser()
    
    # Example configuration
    example_config = {
        'server_name': 'example_server',
        'version': '1.20.1',
        'max_players': 50,
        'difficulty': 'hard',
        'plugins': ['essentials', 'worldedit']
    }
    
    try:
        # Save configuration
        parser.save_config('example_server', example_config)
        
        # Load configuration
        loaded_config = parser.load_config('example_server')
        print("Loaded Configuration:", loaded_config)
        
        # Get environment configurations
        env_config = parser.get_env_config()
        print("Environment Configuration:", env_config)
    
    except Exception as e:
        print(f"Configuration error: {e}")

if __name__ == '__main__':
    main()
