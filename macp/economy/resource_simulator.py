import os
import json
import uuid
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import numpy as np
import networkx as nx
import simpy

class ResourceType(Enum):
    """Comprehensive resource types"""
    RAW_MATERIAL = auto()
    PROCESSED_MATERIAL = auto()
    ENERGY = auto()
    FOOD = auto()
    TECHNOLOGY = auto()
    LABOR = auto()
    LAND = auto()
    INFRASTRUCTURE = auto()

class EconomicActivity(Enum):
    """Economic activities and interactions"""
    EXTRACTION = auto()
    PRODUCTION = auto()
    TRADE = auto()
    CONSUMPTION = auto()
    INNOVATION = auto()
    INFRASTRUCTURE_DEVELOPMENT = auto()
    RESEARCH = auto()

@dataclass
class ResourceSimulationContext:
    """Comprehensive resource simulation context"""
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    seed: int = field(default_factory=lambda: random.randint(1, 1000000))
    world_size: Tuple[int, int] = (256, 256)
    initial_resources: Dict[ResourceType, float] = field(default_factory=dict)
    simulation_duration: int = 365  # Days
    complexity: float = 1.0

@dataclass
class ResourceSimulationConfig:
    """Detailed resource simulation configuration"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active_resources: List[ResourceType] = field(default_factory=list)
    economic_activities: List[EconomicActivity] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class ResourceSimulator:
    """
    Advanced Resource Management and Economic Simulation System
    
    Features:
    - Multi-resource economic simulation
    - Dynamic resource interactions
    - Complex economic activities
    - Customizable simulation parameters
    """
    
    def __init__(
        self, 
        output_dir: str = '/home/veronicae/CascadeProjects/MLSP/resource_simulations'
    ):
        """
        Initialize resource simulator
        
        :param output_dir: Directory to store simulation outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation history storage
        self.simulation_history_file = os.path.join(output_dir, 'simulation_history.json')
        self.simulation_history: Dict[str, Dict[str, Any]] = self._load_simulation_history()
    
    def _load_simulation_history(self) -> Dict[str, Dict[str, Any]]:
        """Load existing simulation history"""
        if os.path.exists(self.simulation_history_file):
            with open(self.simulation_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_simulation_history(
        self, 
        simulation_id: str, 
        simulation_data: Dict[str, Any]
    ):
        """Save simulation results to history"""
        self.simulation_history[simulation_id] = simulation_data
        
        with open(self.simulation_history_file, 'w') as f:
            json.dump(self.simulation_history, f, indent=2)
    
    def _generate_initial_resource_distribution(
        self, 
        context: ResourceSimulationContext
    ) -> Dict[ResourceType, np.ndarray]:
        """
        Generate initial resource distribution
        
        :param context: Simulation context
        :return: Resource distribution maps
        """
        width, height = context.world_size
        resource_maps = {}
        
        # Default initial resources
        default_resources = {
            ResourceType.RAW_MATERIAL: 100.0,
            ResourceType.PROCESSED_MATERIAL: 50.0,
            ResourceType.ENERGY: 75.0,
            ResourceType.FOOD: 200.0,
            ResourceType.TECHNOLOGY: 25.0,
            ResourceType.LABOR: 500.0,
            ResourceType.LAND: 1000.0,
            ResourceType.INFRASTRUCTURE: 50.0
        }
        
        # Merge with context-provided initial resources
        initial_resources = {**default_resources, **context.initial_resources}
        
        for resource_type, base_value in initial_resources.items():
            # Create resource distribution map
            resource_map = np.random.normal(
                base_value, 
                base_value * 0.2, 
                (width, height)
            )
            
            # Spatial variation based on world characteristics
            for x in range(width):
                for y in range(height):
                    # Distance from center influences resource distribution
                    distance_factor = math.sqrt(
                        ((x - width/2) ** 2 + (y - height/2) ** 2) / 
                        ((width/2) ** 2 + (height/2) ** 2)
                    )
                    
                    resource_map[x][y] *= (1 - distance_factor * 0.5)
            
            resource_maps[resource_type] = resource_map
        
        return resource_maps
    
    def simulate_resource_extraction(
        self, 
        context: ResourceSimulationContext, 
        config: Optional[ResourceSimulationConfig] = None
    ) -> Dict[str, Any]:
        """
        Simulate resource extraction dynamics
        
        :param context: Simulation context
        :param config: Simulation configuration
        :return: Resource extraction simulation results
        """
        # Create default configuration
        if not config:
            config = ResourceSimulationConfig(
                active_resources=[
                    ResourceType.RAW_MATERIAL,
                    ResourceType.ENERGY,
                    ResourceType.LAND
                ],
                economic_activities=[
                    EconomicActivity.EXTRACTION
                ]
            )
        
        # Generate initial resource distribution
        resource_maps = self._generate_initial_resource_distribution(context)
        
        # Resource extraction simulation
        extraction_results = {}
        
        for resource_type in config.active_resources:
            resource_map = resource_maps[resource_type]
            extraction_map = np.zeros_like(resource_map)
            
            for x in range(context.world_size[0]):
                for y in range(context.world_size[1]):
                    # Extraction rate based on resource availability
                    base_extraction_rate = config.parameters.get(
                        f'{resource_type.name.lower()}_extraction_rate', 
                        0.1
                    )
                    
                    # Resource depletion factor
                    depletion_factor = config.parameters.get(
                        f'{resource_type.name.lower()}_depletion_rate', 
                        0.02
                    )
                    
                    extraction_amount = (
                        base_extraction_rate * resource_map[x][y] * 
                        (1 - depletion_factor * context.simulation_duration/365)
                    )
                    
                    extraction_map[x][y] = extraction_amount
                    resource_map[x][y] -= extraction_amount
            
            extraction_results[resource_type] = {
                'initial_distribution': resource_maps[resource_type].tolist(),
                'extraction_map': extraction_map.tolist(),
                'total_extraction': np.sum(extraction_map)
            }
        
        # Prepare simulation results
        results = {
            'metadata': asdict(config),
            'context': asdict(context),
            'extraction_results': extraction_results
        }
        
        # Save simulation history
        self._save_simulation_history(context.simulation_id, results)
        
        return results
    
    def simulate_economic_production(
        self, 
        context: ResourceSimulationContext, 
        config: Optional[ResourceSimulationConfig] = None
    ) -> Dict[str, Any]:
        """
        Simulate economic production dynamics
        
        :param context: Simulation context
        :param config: Simulation configuration
        :return: Economic production simulation results
        """
        # Create default configuration
        if not config:
            config = ResourceSimulationConfig(
                active_resources=[
                    ResourceType.RAW_MATERIAL,
                    ResourceType.PROCESSED_MATERIAL,
                    ResourceType.TECHNOLOGY
                ],
                economic_activities=[
                    EconomicActivity.PRODUCTION,
                    EconomicActivity.INNOVATION
                ]
            )
        
        # Generate initial resource distribution
        resource_maps = self._generate_initial_resource_distribution(context)
        
        # Economic production simulation
        production_results = {}
        
        for resource_type in config.active_resources:
            input_resources = config.parameters.get(
                f'{resource_type.name.lower()}_inputs', 
                [ResourceType.RAW_MATERIAL]
            )
            
            production_map = np.zeros_like(resource_maps[resource_type])
            
            for x in range(context.world_size[0]):
                for y in range(context.world_size[1]):
                    # Production efficiency based on input resources
                    production_efficiency = 1.0
                    for input_resource in input_resources:
                        input_availability = resource_maps[input_resource][x][y]
                        production_efficiency *= min(1, input_availability / 100)
                    
                    # Base production rate
                    base_production_rate = config.parameters.get(
                        f'{resource_type.name.lower()}_production_rate', 
                        0.05
                    )
                    
                    # Technology and innovation factor
                    innovation_factor = config.parameters.get(
                        'innovation_multiplier', 
                        1.0 + resource_maps[ResourceType.TECHNOLOGY][x][y] / 100
                    )
                    
                    production_amount = (
                        base_production_rate * 
                        production_efficiency * 
                        innovation_factor
                    )
                    
                    production_map[x][y] = production_amount
                    
                    # Consume input resources
                    for input_resource in input_resources:
                        resource_maps[input_resource][x][y] -= production_amount
            
            production_results[resource_type] = {
                'production_map': production_map.tolist(),
                'total_production': np.sum(production_map)
            }
        
        # Prepare simulation results
        results = {
            'metadata': asdict(config),
            'context': asdict(context),
            'production_results': production_results
        }
        
        # Save simulation history
        self._save_simulation_history(context.simulation_id, results)
        
        return results

def main():
    """Demonstration of resource simulation system"""
    # Initialize resource simulator
    resource_simulator = ResourceSimulator()
    
    # Create simulation context
    context = ResourceSimulationContext(
        seed=42,
        world_size=(512, 512),
        initial_resources={
            ResourceType.RAW_MATERIAL: 150.0,
            ResourceType.ENERGY: 100.0
        },
        simulation_duration=365
    )
    
    # Simulate resource extraction
    extraction_config = ResourceSimulationConfig(
        active_resources=[
            ResourceType.RAW_MATERIAL,
            ResourceType.ENERGY,
            ResourceType.LAND
        ],
        economic_activities=[EconomicActivity.EXTRACTION],
        parameters={
            'raw_material_extraction_rate': 0.15,
            'energy_extraction_rate': 0.1,
            'land_extraction_rate': 0.05
        }
    )
    
    extraction_simulation = resource_simulator.simulate_resource_extraction(
        context, 
        extraction_config
    )
    
    print("Resource Extraction Simulation:")
    for resource, result in extraction_simulation['extraction_results'].items():
        print(f"{resource.name} Total Extraction: {result['total_extraction']}")
    
    # Simulate economic production
    production_config = ResourceSimulationConfig(
        active_resources=[
            ResourceType.RAW_MATERIAL,
            ResourceType.PROCESSED_MATERIAL,
            ResourceType.TECHNOLOGY
        ],
        economic_activities=[
            EconomicActivity.PRODUCTION,
            EconomicActivity.INNOVATION
        ],
        parameters={
            'processed_material_inputs': [ResourceType.RAW_MATERIAL],
            'processed_material_production_rate': 0.08,
            'innovation_multiplier': 1.2
        }
    )
    
    production_simulation = resource_simulator.simulate_economic_production(
        context, 
        production_config
    )
    
    print("\nEconomic Production Simulation:")
    for resource, result in production_simulation['production_results'].items():
        print(f"{resource.name} Total Production: {result['total_production']}")

if __name__ == '__main__':
    main()
