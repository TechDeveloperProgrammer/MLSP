import os
import sys
import numpy as np
import pytest
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from procedural.world_generator import ProceduralWorldGenerator

@pytest.fixture
def world_generator():
    """Create a world generator with a fixed seed for reproducibility"""
    return ProceduralWorldGenerator(seed=42, world_size=(256, 256), complexity=0.5)

def test_world_generator_initialization(world_generator):
    """Test world generator initialization"""
    assert world_generator.seed == 42
    assert world_generator.world_size == (256, 256)
    assert world_generator.complexity == 0.5
    
    # Check noise layers
    assert hasattr(world_generator, 'height_layer')
    assert hasattr(world_generator, 'moisture_layer')
    assert hasattr(world_generator, 'temperature_layer')
    
    assert world_generator.height_layer.shape == (256, 256)
    assert world_generator.moisture_layer.shape == (256, 256)
    assert world_generator.temperature_layer.shape == (256, 256)

def test_biome_classification(world_generator):
    """Test biome classification method"""
    test_cases = [
        (0.1, 0.7, 0.8, 'deep_ocean'),
        (0.25, 0.5, 0.75, 'ocean'),
        (0.4, 0.1, 0.8, 'desert'),
        (0.45, 0.6, 0.5, 'forest'),
        (0.8, 0.2, 0.3, 'snow_mountains'),
        (0.4, 0.4, 0.5, 'plains')
    ]
    
    for height, moisture, temperature, expected_biome in test_cases:
        biome = world_generator._classify_biome(height, moisture, temperature)
        assert biome == expected_biome

def test_world_generation(world_generator):
    """Test complete world generation"""
    world_metadata = world_generator.generate_world()
    
    # Check metadata structure
    assert 'seed' in world_metadata
    assert 'size' in world_metadata
    assert 'biome_distribution' in world_metadata
    assert 'resources' in world_metadata
    assert 'structures' in world_metadata
    
    # Check biome distribution
    biome_dist = world_metadata['biome_distribution']
    assert len(biome_dist) > 0
    assert all(0 <= percentage <= 100 for percentage in biome_dist.values())
    assert sum(biome_dist.values()) == pytest.approx(100, rel=1)

def test_resource_distribution(world_generator):
    """Test resource distribution"""
    world_metadata = world_generator.generate_world()
    resources = world_metadata['resources']
    
    assert 'iron_ore' in resources
    assert 'gold_ore' in resources
    assert 'diamond_ore' in resources
    assert 'trees' in resources
    assert 'water_sources' in resources
    
    # Check resource types
    for resource_type, locations in resources.items():
        assert isinstance(locations, list)
        for location in locations:
            assert len(location) == 2
            assert all(isinstance(coord, int) for coord in location)

def test_structural_generation(world_generator):
    """Test structural generation"""
    world_metadata = world_generator.generate_world()
    structures = world_metadata['structures']
    
    assert 'villages' in structures
    assert 'temples' in structures
    assert 'dungeons' in structures
    assert 'abandoned_mines' in structures
    
    # Check structure types
    for structure_type, locations in structures.items():
        assert isinstance(locations, list)
        for location in locations:
            assert len(location) == 2
            assert all(isinstance(coord, int) for coord in location)

def test_world_map_saving(world_generator, tmp_path):
    """Test world map and metadata saving"""
    # Temporarily override save directory
    original_save_method = world_generator._save_world_map
    
    def mock_save_method(world_map, metadata):
        np.save(tmp_path / f'world_map_{world_generator.seed}.npy', world_map)
        with open(tmp_path / f'world_metadata_{world_generator.seed}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    world_generator._save_world_map = mock_save_method
    
    # Generate world
    world_metadata = world_generator.generate_world()
    
    # Check saved files
    assert (tmp_path / f'world_map_{world_generator.seed}.npy').exists()
    assert (tmp_path / f'world_metadata_{world_generator.seed}.json').exists()
    
    # Restore original save method
    world_generator._save_world_map = original_save_method

def test_reproducibility(world_generator):
    """Test world generation reproducibility"""
    # Generate world twice with same seed
    world_generator1 = ProceduralWorldGenerator(seed=123, world_size=(256, 256))
    world_generator2 = ProceduralWorldGenerator(seed=123, world_size=(256, 256))
    
    metadata1 = world_generator1.generate_world()
    metadata2 = world_generator2.generate_world()
    
    # Compare critical metadata
    assert metadata1['seed'] == metadata2['seed']
    assert metadata1['size'] == metadata2['size']
    
    # Biome distribution should be very similar
    for biome, percentage in metadata1['biome_distribution'].items():
        assert abs(percentage - metadata2['biome_distribution'][biome]) < 5

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
