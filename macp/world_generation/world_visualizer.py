import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from macp.world_generation.procedural_world import ProceduralWorldGenerator, TerrainParameters, BiomeType

class WorldVisualizer:
    """
    Advanced World Visualization System
    
    Features:
    - Multi-layer terrain visualization
    - Biome distribution analysis
    - Geological feature rendering
    """
    
    @staticmethod
    def visualize_world(world_data: Dict[str, np.ndarray], output_dir: str = None):
        """
        Create comprehensive world visualization
        
        :param world_data: Generated world data
        :param output_dir: Directory to save visualizations
        """
        plt.figure(figsize=(20, 15))
        plt.suptitle('Procedural World Generation Analysis', fontsize=16)
        
        # Visualization layers
        layers = [
            ('Elevation Heightmap', world_data['heightmap'], 'terrain'),
            ('Biome Distribution', world_data['biome_map'], 'coolwarm'),
            ('Geological Erosion', world_data['terrain'], 'viridis'),
            ('River Systems', world_data['rivers'], 'Blues'),
            ('Underground Caves', world_data['caves'], 'Greys'),
            ('Ore Deposits', world_data['ore_deposits'], 'Reds')
        ]
        
        for i, (title, data, cmap) in enumerate(layers, 1):
            plt.subplot(2, 3, i)
            plt.title(title)
            plt.imshow(data, cmap=cmap)
            plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save or display
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'world_visualization.png'), dpi=300)
        else:
            plt.show()
    
    @staticmethod
    def analyze_biome_distribution(world_data: Dict[str, np.ndarray]):
        """
        Detailed biome distribution analysis
        
        :param world_data: Generated world data
        """
        biome_map = world_data['biome_map']
        
        # Calculate biome percentages
        unique, counts = np.unique(biome_map, return_counts=True)
        biome_percentages = dict(zip(
            [BiomeType(u).name for u in unique], 
            counts / biome_map.size * 100
        ))
        
        # Pie chart of biome distribution
        plt.figure(figsize=(10, 8))
        plt.title('Biome Distribution')
        plt.pie(
            list(biome_percentages.values()), 
            labels=[f"{k} ({v:.2f}%)" for k, v in biome_percentages.items()],
            autopct='%1.1f%%'
        )
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def terrain_complexity_analysis(world_data: Dict[str, np.ndarray]):
        """
        Analyze terrain complexity and variation
        
        :param world_data: Generated world data
        """
        terrain = world_data['terrain']
        
        plt.figure(figsize=(15, 5))
        
        # Elevation histogram
        plt.subplot(131)
        plt.title('Elevation Distribution')
        plt.hist(terrain.ravel(), bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Elevation')
        plt.ylabel('Frequency')
        
        # Terrain slope analysis
        plt.subplot(132)
        terrain_gradient = np.gradient(terrain)
        plt.title('Terrain Slope Variation')
        plt.imshow(np.sqrt(terrain_gradient[0]**2 + terrain_gradient[1]**2), cmap='hot')
        plt.colorbar(label='Slope Intensity')
        
        # Terrain roughness
        plt.subplot(133)
        plt.title('Local Terrain Roughness')
        local_variance = np.var(terrain, axis=(0, 1))
        sns.heatmap(local_variance, cmap='YlGnBu')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Demonstrate world visualization capabilities
    """
    # Custom terrain parameters
    custom_params = TerrainParameters(
        seed=42,
        world_size=(512, 512),
        roughness=0.7,
        erosion_factor=0.5
    )
    
    # Generate world
    world_gen = ProceduralWorldGenerator(custom_params)
    world_data = world_gen.generate_world()
    
    # Visualize world
    WorldVisualizer.visualize_world(world_data)
    
    # Analyze biome distribution
    WorldVisualizer.analyze_biome_distribution(world_data)
    
    # Terrain complexity analysis
    WorldVisualizer.terrain_complexity_analysis(world_data)

if __name__ == '__main__':
    main()
