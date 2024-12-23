import numpy as np
import noise
import random
from typing import Dict, List, Tuple, Optional
import logging
import json
import os

class ProceduralWorldGenerator:
    """
    Advanced AI-Driven Procedural World Generation System
    
    Features:
    - Multi-layered terrain generation
    - Biome-aware world creation
    - Ecological complexity
    - Resource distribution
    - Structural generation
    """
    
    def __init__(
        self, 
        seed: Optional[int] = None, 
        world_size: Tuple[int, int] = (1024, 1024),
        complexity: float = 0.5
    ):
        """
        Initialize Procedural World Generator
        
        :param seed: Random seed for consistent generation
        :param world_size: Dimensions of the world
        :param complexity: Generation complexity (0-1)
        """
        # Seed management
        self.seed = seed or random.randint(1, 1_000_000)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # World parameters
        self.world_size = world_size
        self.complexity = complexity
        
        # Logging setup
        self.logger = logging.getLogger('ProceduralWorldGenerator')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Generation layers
        self._initialize_generation_layers()
    
    def _initialize_generation_layers(self):
        """
        Initialize multi-layered noise generation
        Layers represent different world characteristics
        """
        # Terrain height layer
        self.height_layer = self._generate_noise_map(
            scale=100 * self.complexity, 
            octaves=6
        )
        
        # Moisture layer
        self.moisture_layer = self._generate_noise_map(
            scale=50 * self.complexity, 
            octaves=4
        )
        
        # Temperature layer
        self.temperature_layer = self._generate_noise_map(
            scale=75 * self.complexity, 
            octaves=5
        )
    
    def _generate_noise_map(
        self, 
        scale: float = 100.0, 
        octaves: int = 6
    ) -> np.ndarray:
        """
        Generate a noise map using Perlin noise
        
        :param scale: Noise scale factor
        :param octaves: Number of noise octaves
        :return: 2D noise map
        """
        world_width, world_height = self.world_size
        noise_map = np.zeros((world_width, world_height))
        
        for i in range(world_width):
            for j in range(world_height):
                noise_map[i][j] = noise.pnoise2(
                    i/scale, 
                    j/scale, 
                    octaves=octaves, 
                    persistence=0.5, 
                    lacunarity=2.0, 
                    repeatx=world_width, 
                    repeaty=world_height, 
                    base=self.seed
                )
        
        return noise_map
    
    def _classify_biome(
        self, 
        height: float, 
        moisture: float, 
        temperature: float
    ) -> str:
        """
        Classify biome based on terrain characteristics
        
        :param height: Terrain height
        :param moisture: Moisture level
        :param temperature: Temperature
        :return: Biome classification
        """
        biome_map = {
            # Water Biomes
            (height < 0.2, moisture > 0.6): 'deep_ocean',
            (height < 0.3, moisture > 0.4): 'ocean',
            
            # Desert Biomes
            (height >= 0.3, temperature > 0.7, moisture < 0.2): 'desert',
            (height >= 0.3, temperature > 0.6, moisture < 0.3): 'desert_hills',
            
            # Forest Biomes
            (height >= 0.4, moisture > 0.5, temperature > 0.4): 'forest',
            (height >= 0.5, moisture > 0.6, temperature > 0.5): 'dense_forest',
            
            # Mountain Biomes
            (height > 0.7, temperature < 0.4): 'snow_mountains',
            (height > 0.6, temperature > 0.4): 'mountains',
            
            # Plains and Grasslands
            (height >= 0.3, height < 0.5, moisture > 0.3, moisture < 0.6): 'plains',
            (height >= 0.3, height < 0.5, moisture > 0.4, temperature > 0.5): 'grasslands',
            
            # Tundra and Cold Regions
            (height >= 0.4, temperature < 0.3): 'tundra',
            (height >= 0.3, temperature < 0.2): 'ice_plains'
        }
        
        # Default to plains if no match
        for (h_cond, m_cond), biome in biome_map.items():
            if h_cond and m_cond:
                return biome
        
        return 'plains'
    
    def generate_world(self) -> Dict[str, Any]:
        """
        Generate a complete procedural world
        
        :return: World generation metadata
        """
        world_width, world_height = self.world_size
        world_map = np.zeros((world_width, world_height), dtype='<U20')
        
        # Biome generation
        for x in range(world_width):
            for y in range(world_height):
                height = (self.height_layer[x][y] + 1) / 2  # Normalize
                moisture = (self.moisture_layer[x][y] + 1) / 2
                temperature = (self.temperature_layer[x][y] + 1) / 2
                
                world_map[x][y] = self._classify_biome(
                    height, moisture, temperature
                )
        
        # Resource distribution
        resources = self._distribute_resources(world_map)
        
        # Structural generation
        structures = self._generate_structures(world_map)
        
        # World metadata
        world_metadata = {
            'seed': self.seed,
            'size': self.world_size,
            'biome_distribution': self._analyze_biome_distribution(world_map),
            'resources': resources,
            'structures': structures
        }
        
        # Optional: Save world map
        self._save_world_map(world_map, world_metadata)
        
        return world_metadata
    
    def _distribute_resources(
        self, 
        world_map: np.ndarray
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Distribute resources across the world
        
        :param world_map: Generated world map
        :return: Resource distribution
        """
        resources = {
            'iron_ore': [],
            'gold_ore': [],
            'diamond_ore': [],
            'trees': [],
            'water_sources': []
        }
        
        world_width, world_height = world_map.shape
        
        for _ in range(int(world_width * world_height * 0.01)):  # 1% resource density
            x, y = random.randint(0, world_width-1), random.randint(0, world_height-1)
            biome = world_map[x][y]
            
            # Biome-specific resource distribution
            if biome == 'mountains':
                if random.random() < 0.3:
                    resources['iron_ore'].append((x, y))
                if random.random() < 0.1:
                    resources['gold_ore'].append((x, y))
                if random.random() < 0.05:
                    resources['diamond_ore'].append((x, y))
            
            if biome in ['forest', 'dense_forest']:
                if random.random() < 0.5:
                    resources['trees'].append((x, y))
            
            if biome in ['ocean', 'deep_ocean']:
                if random.random() < 0.2:
                    resources['water_sources'].append((x, y))
        
        return resources
    
    def _generate_structures(
        self, 
        world_map: np.ndarray
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Generate world structures
        
        :param world_map: Generated world map
        :return: Structural generation metadata
        """
        structures = {
            'villages': [],
            'temples': [],
            'dungeons': [],
            'abandoned_mines': []
        }
        
        world_width, world_height = world_map.shape
        
        for _ in range(int(world_width * world_height * 0.005)):  # 0.5% structure density
            x, y = random.randint(0, world_width-1), random.randint(0, world_height-1)
            biome = world_map[x][y]
            
            # Biome-specific structure generation
            if biome in ['plains', 'grasslands'] and random.random() < 0.3:
                structures['villages'].append((x, y))
            
            if biome in ['desert', 'mountains'] and random.random() < 0.2:
                structures['temples'].append((x, y))
            
            if biome not in ['ocean', 'deep_ocean'] and random.random() < 0.1:
                structures['dungeons'].append((x, y))
                
            if biome in ['mountains', 'hills'] and random.random() < 0.15:
                structures['abandoned_mines'].append((x, y))
        
        return structures
    
    def _analyze_biome_distribution(
        self, 
        world_map: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze biome distribution
        
        :param world_map: Generated world map
        :return: Biome percentage distribution
        """
        unique, counts = np.unique(world_map, return_counts=True)
        total_cells = world_map.size
        
        return {
            biome: count / total_cells * 100 
            for biome, count in dict(zip(unique, counts)).items()
        }
    
    def _save_world_map(
        self, 
        world_map: np.ndarray, 
        metadata: Dict[str, Any]
    ):
        """
        Save generated world map and metadata
        
        :param world_map: Generated world map
        :param metadata: World generation metadata
        """
        output_dir = '/opt/mlsp/world_generation'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save world map
        np.save(os.path.join(output_dir, f'world_map_{self.seed}.npy'), world_map)
        
        # Save metadata
        with open(os.path.join(output_dir, f'world_metadata_{self.seed}.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"World map and metadata saved for seed {self.seed}")

def main():
    """Example usage of Procedural World Generator"""
    generator = ProceduralWorldGenerator(
        seed=42, 
        world_size=(512, 512), 
        complexity=0.7
    )
    
    world_metadata = generator.generate_world()
    print(json.dumps(world_metadata, indent=2))

if __name__ == '__main__':
    main()
