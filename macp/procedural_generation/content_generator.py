import os
import json
import uuid
import math
import secrets
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import numpy as np
import noise
import networkx as nx

class ContentType(Enum):
    """Comprehensive content generation types"""
    STRUCTURE = auto()
    VEGETATION = auto()
    MINERAL_DEPOSIT = auto()
    CAVE_SYSTEM = auto()
    LANDMARK = auto()
    ECOSYSTEM = auto()
    ARTIFACT = auto()
    CUSTOM = auto()

class GenerationStrategy(Enum):
    """Content generation strategies"""
    RANDOM = auto()
    NOISE_BASED = auto()
    GRAPH_BASED = auto()
    RULE_BASED = auto()
    PROCEDURAL = auto()
    CUSTOM = auto()

@dataclass
class GenerationContext:
    """Comprehensive generation context"""
    seed: int = field(default_factory=lambda: secrets.randbelow(1000000) + 1)
    world_size: Tuple[int, int] = (256, 256)
    terrain_height_map: Optional[np.ndarray] = None
    biome_map: Optional[np.ndarray] = None
    complexity: float = 1.0

@dataclass
class ContentGenerationConfig:
    """Detailed content generation configuration"""
    generation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_type: ContentType = ContentType.STRUCTURE
    generation_strategy: GenerationStrategy = GenerationStrategy.NOISE_BASED
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class ProceduralContentGenerator:
    """
    Advanced Procedural Content Generation System
    
    Features:
    - Multi-type content generation
    - Flexible generation strategies
    - Context-aware generation
    - Customizable parameters
    """
    
    def __init__(
        self, 
        output_dir: str = '/home/veronicae/CascadeProjects/MLSP/procedural_content'
    ):
        """
        Initialize procedural content generator
        
        :param output_dir: Directory to store generated content
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Content generation history
        self.generation_history_file = os.path.join(output_dir, 'generation_history.json')
        self.generation_history: Dict[str, Dict[str, Any]] = self._load_generation_history()
    
    def _load_generation_history(self) -> Dict[str, Dict[str, Any]]:
        """Load existing generation history"""
        if os.path.exists(self.generation_history_file):
            with open(self.generation_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_generation_history(
        self, 
        generation_id: str, 
        content_data: Dict[str, Any]
    ):
        """Save generated content to history"""
        self.generation_history[generation_id] = content_data
        
        with open(self.generation_history_file, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
    
    def _generate_noise_map(
        self, 
        context: GenerationContext, 
        scale: float = 100.0
    ) -> np.ndarray:
        """
        Generate noise map for content generation
        
        :param context: Generation context
        :param scale: Noise scale factor
        :return: Generated noise map
        """
        width, height = context.world_size
        octaves = int(3 * context.complexity)
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
                    base=context.seed
                )
        
        # Normalize noise
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        
        return noise_map
    
    def generate_structures(
        self, 
        context: GenerationContext, 
        config: Optional[ContentGenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate structures using various strategies
        
        :param context: Generation context
        :param config: Content generation configuration
        :return: Generated structure data
        """
        # Create default configuration
        if not config:
            config = ContentGenerationConfig(
                content_type=ContentType.STRUCTURE,
                generation_strategy=GenerationStrategy.NOISE_BASED
            )
        
        # Generate noise map
        noise_map = self._generate_noise_map(context)
        
        # Structure generation strategies
        structures = []
        
        if config.generation_strategy == GenerationStrategy.NOISE_BASED:
            # Generate structures based on noise threshold
            threshold = config.parameters.get('threshold', 0.7)
            structure_types = config.parameters.get(
                'structure_types', 
                ['castle', 'tower', 'temple', 'ruins']
            )
            
            for x in range(context.world_size[0]):
                for y in range(context.world_size[1]):
                    if noise_map[x][y] > threshold:
                        structures.append({
                            'type': secrets.choice(structure_types),
                            'x': x,
                            'y': y,
                            'size': secrets.randbelow(150) / 100 + 0.5
                        })
        
        elif config.generation_strategy == GenerationStrategy.GRAPH_BASED:
            # Generate structures using graph-based approach
            G = nx.grid_2d_graph(*context.world_size)
            
            for node in G.nodes():
                if secrets.randbelow(100) < 5:  # 5% chance of structure
                    structures.append({
                        'type': 'settlement',
                        'x': node[0],
                        'y': node[1],
                        'connections': list(G.neighbors(node))
                    })
        
        # Prepare generation results
        results = {
            'metadata': asdict(config),
            'context': asdict(context),
            'structures': structures
        }
        
        # Save generation history
        self._save_generation_history(config.generation_id, results)
        
        return results
    
    def generate_vegetation(
        self, 
        context: GenerationContext, 
        config: Optional[ContentGenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate vegetation using various strategies
        
        :param context: Generation context
        :param config: Content generation configuration
        :return: Generated vegetation data
        """
        # Create default configuration
        if not config:
            config = ContentGenerationConfig(
                content_type=ContentType.VEGETATION,
                generation_strategy=GenerationStrategy.NOISE_BASED
            )
        
        # Generate noise map
        noise_map = self._generate_noise_map(context)
        
        # Vegetation generation
        vegetation = []
        
        if config.generation_strategy == GenerationStrategy.NOISE_BASED:
            # Generate vegetation based on noise and terrain
            vegetation_types = config.parameters.get(
                'vegetation_types', 
                ['tree', 'bush', 'grass', 'flower']
            )
            
            for x in range(context.world_size[0]):
                for y in range(context.world_size[1]):
                    # Consider terrain height and noise
                    if (context.terrain_height_map is not None and 
                        context.terrain_height_map[x][y] > 0.3 and 
                        noise_map[x][y] > 0.5):
                        vegetation.append({
                            'type': secrets.choice(vegetation_types),
                            'x': x,
                            'y': y,
                            'density': noise_map[x][y]
                        })
        
        # Prepare generation results
        results = {
            'metadata': asdict(config),
            'context': asdict(context),
            'vegetation': vegetation
        }
        
        # Save generation history
        self._save_generation_history(config.generation_id, results)
        
        return results
    
    def generate_mineral_deposits(
        self, 
        context: GenerationContext, 
        config: Optional[ContentGenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Generate mineral deposits using various strategies
        
        :param context: Generation context
        :param config: Content generation configuration
        :return: Generated mineral deposit data
        """
        # Create default configuration
        if not config:
            config = ContentGenerationConfig(
                content_type=ContentType.MINERAL_DEPOSIT,
                generation_strategy=GenerationStrategy.RULE_BASED
            )
        
        # Generate noise map
        noise_map = self._generate_noise_map(context)
        
        # Mineral deposit generation
        mineral_deposits = []
        
        if config.generation_strategy == GenerationStrategy.RULE_BASED:
            # Rule-based mineral deposit generation
            mineral_types = config.parameters.get(
                'mineral_types', 
                ['iron', 'gold', 'diamond', 'coal', 'copper']
            )
            
            for x in range(context.world_size[0]):
                for y in range(context.world_size[1]):
                    # Depth-based mineral generation
                    depth = 1 - noise_map[x][y]
                    
                    if depth > 0.7:  # Deep layers
                        mineral = secrets.choice(
                            mineral_types
                        )
                        
                        mineral_deposits.append({
                            'type': mineral,
                            'x': x,
                            'y': y,
                            'depth': depth,
                            'quantity': secrets.randbelow(90) + 10
                        })
        
        # Prepare generation results
        results = {
            'metadata': asdict(config),
            'context': asdict(context),
            'mineral_deposits': mineral_deposits
        }
        
        # Save generation history
        self._save_generation_history(config.generation_id, results)
        
        return results

def main():
    """Demonstration of procedural content generation system"""
    # Initialize procedural content generator
    content_generator = ProceduralContentGenerator()
    
    # Create generation context
    context = GenerationContext(
        seed=42,
        world_size=(512, 512),
        complexity=1.5
    )
    
    # Generate structures
    structure_config = ContentGenerationConfig(
        content_type=ContentType.STRUCTURE,
        generation_strategy=GenerationStrategy.NOISE_BASED,
        parameters={
            'threshold': 0.75,
            'structure_types': ['castle', 'temple', 'ruins']
        }
    )
    
    structures = content_generator.generate_structures(
        context, 
        structure_config
    )
    
    print("Generated Structures:")
    print(json.dumps(structures, indent=2))
    
    # Generate vegetation
    vegetation_config = ContentGenerationConfig(
        content_type=ContentType.VEGETATION,
        generation_strategy=GenerationStrategy.NOISE_BASED,
        parameters={
            'vegetation_types': ['oak', 'pine', 'birch', 'flower']
        }
    )
    
    vegetation = content_generator.generate_vegetation(
        context, 
        vegetation_config
    )
    
    print("\nGenerated Vegetation:")
    print(json.dumps(vegetation, indent=2))
    
    # Generate mineral deposits
    mineral_config = ContentGenerationConfig(
        content_type=ContentType.MINERAL_DEPOSIT,
        generation_strategy=GenerationStrategy.RULE_BASED
    )
    
    minerals = content_generator.generate_mineral_deposits(
        context, 
        mineral_config
    )
    
    print("\nGenerated Mineral Deposits:")
    print(json.dumps(minerals, indent=2))

if __name__ == '__main__':
    main()
