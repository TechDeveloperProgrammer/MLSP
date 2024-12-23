import os
import pytest
import tempfile
import platform
from unittest.mock import patch, MagicMock

# Import the modules to test
from server_manager import MinecraftServerManager, ServerConfig
from config_parser import ConfigParser
from plugin_manager import PluginManager

@pytest.fixture
def server_manager():
    """Create a temporary server manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield MinecraftServerManager(base_dir=temp_dir)

@pytest.fixture
def config_parser():
    """Create a temporary config parser for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield ConfigParser(config_dir=temp_dir)

@pytest.fixture
def plugin_manager():
    """Create a temporary plugin manager for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield PluginManager(plugins_dir=temp_dir)

def test_server_config_creation():
    """Test server configuration creation"""
    config = ServerConfig(
        name='test_server',
        version='1.20.1',
        server_type='paper',
        max_players=50
    )
    
    assert config.name == 'test_server'
    assert config.version == '1.20.1'
    assert config.server_type == 'paper'
    assert config.max_players == 50

def test_server_manager_initialization(server_manager):
    """Test server manager initialization"""
    assert os.path.exists(server_manager.base_dir)
    assert server_manager.os_type in ['linux', 'darwin', 'windows']

def test_server_creation(server_manager):
    """Test server creation process"""
    config = ServerConfig(
        name='test_survival_server',
        version='1.20.1',
        server_type='paper',
        max_players=50,
        plugins=['essentials', 'worldedit']
    )
    
    server_path = server_manager.create_server(config)
    
    assert os.path.exists(server_path)
    assert os.path.exists(os.path.join(server_path, 'server.jar'))
    assert os.path.exists(os.path.join(server_path, 'server.properties'))
    assert os.path.exists(os.path.join(server_path, f'start.{"bat" if server_manager.os_type == "windows" else "sh"}'))

@patch('subprocess.Popen')
def test_server_start(mock_popen, server_manager):
    """Test server start process"""
    config = ServerConfig(
        name='start_test_server',
        version='1.20.1',
        server_type='paper'
    )
    
    # Create server first
    server_manager.create_server(config)
    
    # Attempt to start server
    server_manager.start_server('start_test_server')
    
    # Verify subprocess was called
    mock_popen.assert_called_once()

def test_server_backup(server_manager):
    """Test server backup process"""
    config = ServerConfig(
        name='backup_test_server',
        version='1.20.1',
        server_type='paper'
    )
    
    # Create server first
    server_manager.create_server(config)
    
    # Create backup
    server_manager.backup_server('backup_test_server')
    
    # Check backup directory exists
    backup_dir = os.path.join(server_manager.base_dir, 'backup_test_server', 'backups')
    assert os.path.exists(backup_dir)
    assert len(os.listdir(backup_dir)) > 0

def test_config_parser_save_load(config_parser):
    """Test configuration saving and loading"""
    test_config = {
        'server_name': 'test_config_server',
        'version': '1.20.1',
        'max_players': 50
    }
    
    # Save configuration
    config_parser.save_config('test_config_server', test_config)
    
    # Load configuration
    loaded_config = config_parser.load_config('test_config_server')
    
    assert loaded_config == test_config

def test_plugin_manager_download(plugin_manager):
    """Test plugin download functionality"""
    test_plugin = PluginManager.Plugin(
        name='test_plugin',
        version='1.0.0',
        download_url='https://example.com/test_plugin.jar'
    )
    
    # Mock the download request
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'test_content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Download plugin
        downloaded_path = plugin_manager.download_plugin(test_plugin)
        
        assert os.path.exists(downloaded_path)
        assert os.path.basename(downloaded_path) == 'test_plugin-1.0.0.jar'

def test_plugin_manager_bulk_download(plugin_manager):
    """Test bulk plugin download"""
    test_plugins = [
        PluginManager.Plugin(
            name='plugin1',
            version='1.0.0',
            download_url='https://example.com/plugin1.jar'
        ),
        PluginManager.Plugin(
            name='plugin2',
            version='1.0.0',
            download_url='https://example.com/plugin2.jar'
        )
    ]
    
    # Mock the download request
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'test_content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Bulk download plugins
        downloaded_plugins = plugin_manager.bulk_download_plugins(test_plugins)
        
        assert len(downloaded_plugins) == 2
        assert all(os.path.exists(path) for path in downloaded_plugins)

def test_cross_platform_compatibility():
    """Verify cross-platform compatibility checks"""
    os_type = platform.system().lower()
    
    # Ensure the system is one of the supported platforms
    assert os_type in ['linux', 'darwin', 'windows'], f"Unsupported platform: {os_type}"

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
