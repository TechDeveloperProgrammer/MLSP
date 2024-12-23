import os
import sys
import importlib
import inspect
import json
import hashlib
import zipfile
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import yaml
import jsonschema

class PluginCategory(Enum):
    """Comprehensive plugin categories"""
    WORLD_GENERATION = auto()
    MOD_GENERATION = auto()
    TERRAIN_SIMULATION = auto()
    BIOME_GENERATION = auto()
    ML_EXTENSION = auto()
    VISUALIZATION = auto()
    CUSTOM = auto()

class PluginStatus(Enum):
    """Plugin lifecycle status"""
    UNINSTALLED = auto()
    INSTALLED = auto()
    ACTIVE = auto()
    DISABLED = auto()
    ERROR = auto()

@dataclass
class PluginMetadata:
    """Comprehensive plugin metadata"""
    plugin_id: str
    name: str
    version: str
    description: str = ''
    author: str = 'Unknown'
    category: PluginCategory = PluginCategory.CUSTOM
    dependencies: List[str] = field(default_factory=list)
    status: PluginStatus = PluginStatus.UNINSTALLED
    entry_point: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None

class PluginValidator:
    """Advanced plugin validation system"""
    
    @staticmethod
    def validate_plugin_config(
        plugin_metadata: PluginMetadata, 
        config: Dict[str, Any]
    ) -> bool:
        """
        Validate plugin configuration against schema
        
        :param plugin_metadata: Plugin metadata
        :param config: Plugin configuration
        :return: Validation result
        """
        if not plugin_metadata.config_schema:
            return True
        
        try:
            jsonschema.validate(instance=config, schema=plugin_metadata.config_schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False

class PluginManager:
    """
    Advanced Plugin and Extension Management System
    
    Features:
    - Dynamic plugin loading
    - Dependency management
    - Configuration validation
    - Lifecycle management
    """
    
    def __init__(
        self, 
        plugins_dir: str = '/home/veronicae/CascadeProjects/MLSP/plugins'
    ):
        """
        Initialize plugin manager
        
        :param plugins_dir: Directory to store plugins
        """
        self.plugins_dir = plugins_dir
        os.makedirs(plugins_dir, exist_ok=True)
        
        # Plugin metadata storage
        self.plugins_file = os.path.join(plugins_dir, 'plugins.json')
        self.plugins: Dict[str, PluginMetadata] = self._load_plugins()
        
        # Active plugin instances
        self._active_plugins: Dict[str, Any] = {}
    
    def _load_plugins(self) -> Dict[str, PluginMetadata]:
        """Load existing plugin metadata"""
        if os.path.exists(self.plugins_file):
            with open(self.plugins_file, 'r') as f:
                plugins_data = json.load(f)
                return {
                    plugin_id: PluginMetadata(**plugin) 
                    for plugin_id, plugin in plugins_data.items()
                }
        return {}
    
    def _save_plugins(self):
        """Save plugin metadata"""
        plugins_data = {
            plugin_id: asdict(plugin) 
            for plugin_id, plugin in self.plugins.items()
        }
        
        with open(self.plugins_file, 'w') as f:
            json.dump(plugins_data, f, indent=2)
    
    def _generate_plugin_hash(self, plugin_path: str) -> str:
        """Generate unique plugin hash"""
        with open(plugin_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def install_plugin(
        self, 
        plugin_path: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> PluginMetadata:
        """
        Install a plugin
        
        :param plugin_path: Path to plugin file/directory
        :param config: Optional plugin configuration
        :return: Plugin metadata
        """
        # Handle zip plugin
        if plugin_path.endswith('.zip'):
            with zipfile.ZipFile(plugin_path, 'r') as zip_ref:
                plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]
                extract_path = os.path.join(self.plugins_dir, plugin_name)
                zip_ref.extractall(extract_path)
                plugin_path = extract_path
        
        # Find plugin metadata
        metadata_path = os.path.join(plugin_path, 'plugin.yml')
        if not os.path.exists(metadata_path):
            raise ValueError("Invalid plugin: missing plugin.yml")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = yaml.safe_load(f)
        
        # Create plugin metadata
        plugin_metadata = PluginMetadata(
            plugin_id=metadata_dict.get('id', hashlib.md5(
                plugin_path.encode()).hexdigest()),
            name=metadata_dict.get('name', 'Unnamed Plugin'),
            version=metadata_dict.get('version', '0.1.0'),
            description=metadata_dict.get('description', ''),
            author=metadata_dict.get('author', 'Unknown'),
            category=PluginCategory[
                metadata_dict.get('category', 'CUSTOM').upper()
            ],
            dependencies=metadata_dict.get('dependencies', []),
            entry_point=metadata_dict.get('entry_point'),
            config_schema=metadata_dict.get('config_schema')
        )
        
        # Validate configuration
        if config and not PluginValidator.validate_plugin_config(
            plugin_metadata, config
        ):
            raise ValueError("Invalid plugin configuration")
        
        # Generate hash
        plugin_metadata.hash = self._generate_plugin_hash(plugin_path)
        
        # Store plugin metadata
        self.plugins[plugin_metadata.plugin_id] = plugin_metadata
        self._save_plugins()
        
        return plugin_metadata
    
    def load_plugin(
        self, 
        plugin_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Load and activate a plugin
        
        :param plugin_id: Plugin identifier
        :param config: Optional plugin configuration
        :return: Loaded plugin instance
        """
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        plugin_metadata = self.plugins[plugin_id]
        
        # Check dependencies
        for dependency in plugin_metadata.dependencies:
            if dependency not in self.plugins:
                raise ValueError(f"Missing dependency: {dependency}")
        
        # Import plugin module
        try:
            sys.path.insert(0, self.plugins_dir)
            module_path = plugin_metadata.entry_point.replace('.', '/')
            spec = importlib.util.spec_from_file_location(
                plugin_id, 
                os.path.join(self.plugins_dir, f"{module_path}.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_classes = [
                cls for name, cls in inspect.getmembers(module, inspect.isclass)
                if hasattr(cls, 'initialize')
            ]
            
            if not plugin_classes:
                raise ValueError("No plugin class found")
            
            # Instantiate plugin
            plugin_class = plugin_classes[0]
            plugin_instance = plugin_class(config or {})
            plugin_instance.initialize()
            
            # Store active plugin
            self._active_plugins[plugin_id] = plugin_instance
            
            # Update plugin status
            plugin_metadata.status = PluginStatus.ACTIVE
            self._save_plugins()
            
            return plugin_instance
        
        except Exception as e:
            # Update plugin status
            plugin_metadata.status = PluginStatus.ERROR
            self._save_plugins()
            raise
    
    def unload_plugin(self, plugin_id: str):
        """
        Unload a plugin
        
        :param plugin_id: Plugin identifier
        """
        if plugin_id in self._active_plugins:
            plugin = self._active_plugins[plugin_id]
            
            # Call plugin cleanup method
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()
            
            del self._active_plugins[plugin_id]
        
        # Update plugin status
        if plugin_id in self.plugins:
            self.plugins[plugin_id].status = PluginStatus.INSTALLED
            self._save_plugins()
    
    def list_plugins(
        self, 
        category: Optional[PluginCategory] = None,
        status: Optional[PluginStatus] = None
    ) -> List[PluginMetadata]:
        """
        List plugins with optional filtering
        
        :param category: Optional category filter
        :param status: Optional status filter
        :return: List of plugin metadata
        """
        return [
            plugin for plugin in self.plugins.values()
            if (not category or plugin.category == category) and
               (not status or plugin.status == status)
        ]

def main():
    """Demonstration of plugin management system"""
    # Initialize plugin manager
    plugin_manager = PluginManager()
    
    # Install a plugin
    world_gen_plugin = plugin_manager.install_plugin(
        '/path/to/world_generation_plugin.zip',
        config={
            'seed': 12345,
            'complexity': 'advanced'
        }
    )
    
    # Load and activate plugin
    world_gen_instance = plugin_manager.load_plugin(
        world_gen_plugin.plugin_id
    )
    
    # List installed plugins
    print("Installed Plugins:")
    for plugin in plugin_manager.list_plugins():
        print(f"{plugin.name} (v{plugin.version})")
    
    # Unload plugin
    plugin_manager.unload_plugin(world_gen_plugin.plugin_id)

if __name__ == '__main__':
    main()
