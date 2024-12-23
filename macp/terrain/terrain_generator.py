import os
import random
import numpy as np
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import noise
import scipy.ndimage
import skimage.transform

class TerrainFeature(Enum):
    """Comprehensive terrain feature types"""
    MOUNTAIN = auto()
    VALLEY = auto()
    PLATEAU = auto()
    CANYON = auto()
    RIVER = auto()
    COASTLINE = auto()
    VOLCANIC = auto()
    GLACIAL = auto()

class BiomeType(Enum):
    """Detailed biome classifications"""
    TUNDRA = auto()
    TAIGA = auto()
    TEMPERATE_FOREST = auto()
    TROPICAL_RAINFOREST = auto()
    DESERT = auto()
    GRASSLAND = auto()
    SAVANNA = auto()
    ALPINE = auto()
    VOLCANIC = auto()
    ARCTIC = auto()

@dataclass
class TerrainMetadata:
    """Comprehensive terrain generation metadata"""
    terrain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    seed: int = field(default_factory=lambda: random.randint(1, 1000000))
    size: Tuple[int, int] = (256, 256)
    features: List[TerrainFeature] = field(default_factory=list)
    biomes: List[BiomeType] = field(default_factory=list)
    complexity: float = 1.0
    generation_time: Optional[float] = None
    height_range: Tuple[float, float] = (0.0, 1.0)

class TerrainGenerator:
    """
    Advanced Terrain Simulation and Generation Framework
    
    Features:
    - Procedural terrain generation
    - Multi-layered noise generation
    - Biome and feature simulation
    - Customizable terrain characteristics
    """
    
    def __init__(
        self, 
        output_dir: str = '/home/veronicae/CascadeProjects/MLSP/terrain_outputs'
    ):
        """
        Initialize terrain generator
        
        :param output_dir: Directory to store generated terrains
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _generate_base_noise(
        self, 
        metadata: TerrainMetadata
    ) -> np.ndarray:
        """
        Generate base terrain noise
        
        :param metadata: Terrain generation metadata
        :return: Base noise array
        """
        width, height = metadata.size
        scale = 100.0 * metadata.complexity
        octaves = int(3 * metadata.complexity)
        persistence = 0.5
        lacunarity = 2.0
        
        noise_map = np.zeros((width, height))
        
        for i in range(width):
            for j in range(height):
                noise_map[i][j] = noise.pnoise2(
                    i/scale, 
                    j/scale, 
                    octaves=octaves, 
                    persistence=persistence, 
                    lacunarity=lacunarity, 
                    repeatx=width, 
                    repeaty=height, 
                    base=metadata.seed
                )
        
        # Normalize noise
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        return noise_map
    
    def _add_terrain_features(
        self, 
        noise_map: np.ndarray, 
        metadata: TerrainMetadata
    ) -> np.ndarray:
        """
        Add specific terrain features
        
        :param noise_map: Base noise map
        :param metadata: Terrain generation metadata
        :return: Enhanced noise map
        """
        for feature in metadata.features:
            if feature == TerrainFeature.MOUNTAIN:
                # Mountain generation
                mountain_mask = noise_map > 0.7
                noise_map[mountain_mask] += 0.3
            
            elif feature == TerrainFeature.VALLEY:
                # Valley generation
                valley_mask = noise_map < 0.3
                noise_map[valley_mask] -= 0.2
            
            elif feature == TerrainFeature.PLATEAU:
                # Plateau generation
                plateau_mask = (noise_map > 0.5) & (noise_map < 0.7)
                noise_map[plateau_mask] = 0.6
            
            elif feature == TerrainFeature.RIVER:
                # River generation
                river_mask = (noise_map > 0.4) & (noise_map < 0.5)
                noise_map[river_mask] -= 0.1
        
        return noise_map
    
    def _assign_biomes(
        self, 
        noise_map: np.ndarray, 
        metadata: TerrainMetadata
    ) -> np.ndarray:
        """
        Assign biomes based on terrain characteristics
        
        :param noise_map: Terrain noise map
        :param metadata: Terrain generation metadata
        :return: Biome map
        """
        biome_map = np.zeros_like(noise_map, dtype=int)
        
        for biome in metadata.biomes:
            if biome == BiomeType.TUNDRA:
                biome_mask = noise_map < 0.2
                biome_map[biome_mask] = biome.value
            
            elif biome == BiomeType.DESERT:
                biome_mask = (noise_map > 0.7) & (noise_map < 0.9)
                biome_map[biome_mask] = biome.value
            
            elif biome == BiomeType.TEMPERATE_FOREST:
                biome_mask = (noise_map > 0.3) & (noise_map < 0.6)
                biome_map[biome_mask] = biome.value
            
            elif biome == BiomeType.ALPINE:
                biome_mask = noise_map > 0.9
                biome_map[biome_mask] = biome.value
        
        return biome_map
    
    def generate_terrain(
        self, 
        metadata: Optional[TerrainMetadata] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive terrain
        
        :param metadata: Optional terrain generation metadata
        :return: Generated terrain data
        """
        # Create metadata if not provided
        if not metadata:
            metadata = TerrainMetadata(
                features=[
                    TerrainFeature.MOUNTAIN, 
                    TerrainFeature.RIVER
                ],
                biomes=[
                    BiomeType.TEMPERATE_FOREST, 
                    BiomeType.ALPINE
                ]
            )
        
        # Generate base noise
        noise_map = self._generate_base_noise(metadata)
        
        # Add terrain features
        noise_map = self._add_terrain_features(noise_map, metadata)
        
        # Assign biomes
        biome_map = self._assign_biomes(noise_map, metadata)
        
        # Prepare output
        output = {
            'metadata': asdict(metadata),
            'height_map': noise_map.tolist(),
            'biome_map': biome_map.tolist()
        }
        
        # Save terrain data
        output_file = os.path.join(
            self.output_dir, 
            f'{metadata.terrain_id}_terrain.json'
        )
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    def visualize_terrain(
        self, 
        terrain_data: Dict[str, Any], 
        output_format: str = 'png'
    ) -> str:
        """
        Visualize generated terrain
        
        :param terrain_data: Terrain generation output
        :param output_format: Output image format
        :return: Path to visualization
        """
        import matplotlib.pyplot as plt
        
        height_map = np.array(terrain_data['height_map'])
        biome_map = np.array(terrain_data['biome_map'])
        
        # Create figure
        plt.figure(figsize=(10, 5))
        
        # Height map visualization
        plt.subplot(1, 2, 1)
        plt.title('Height Map')
        plt.imshow(height_map, cmap='terrain')
        plt.colorbar(label='Elevation')
        
        # Biome map visualization
        plt.subplot(1, 2, 2)
        plt.title('Biome Map')
        plt.imshow(biome_map, cmap='tab20')
        plt.colorbar(
            ticks=range(len(BiomeType)), 
            label='Biome Types'
        )
        plt.tight_layout()
        
        # Save visualization
        visualization_path = os.path.join(
            self.output_dir, 
            f'{terrain_data["metadata"]["terrain_id"]}_visualization.{output_format}'
        )
        plt.savefig(visualization_path)
        plt.close()
        
        return visualization_path

def main():
    """Demonstration of terrain generation system"""
    # Initialize terrain generator
    terrain_generator = TerrainGenerator()
    
    # Generate terrain
    terrain_metadata = TerrainMetadata(
        size=(512, 512),
        features=[
            TerrainFeature.MOUNTAIN,
            TerrainFeature.RIVER,
            TerrainFeature.CANYON
        ],
        biomes=[
            BiomeType.ALPINE,
            BiomeType.TEMPERATE_FOREST,
            BiomeType.VOLCANIC
        ],
        complexity=1.5
    )
    
    terrain_data = terrain_generator.generate_terrain(terrain_metadata)
    
    # Visualize terrain
    visualization_path = terrain_generator.visualize_terrain(terrain_data)
    
    print(f"Terrain Generated: {terrain_data['metadata']['terrain_id']}")
    print(f"Visualization saved: {visualization_path}")

if __name__ == '__main__':
    main()
