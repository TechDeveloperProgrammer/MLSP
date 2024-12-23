import numpy as np
import scipy.stats as stats
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ClimateZone(Enum):
    """Comprehensive Climate Zone Classification"""
    TROPICAL = auto()
    SUBTROPICAL = auto()
    TEMPERATE = auto()
    MEDITERRANEAN = auto()
    ALPINE = auto()
    DESERT = auto()
    TUNDRA = auto()
    POLAR = auto()
    MONSOON = auto()
    MARITIME = auto()

class EcosystemType(Enum):
    """Detailed Ecosystem Classification"""
    RAINFOREST = auto()
    DECIDUOUS_FOREST = auto()
    CONIFEROUS_FOREST = auto()
    GRASSLAND = auto()
    SAVANNA = auto()
    DESERT = auto()
    TUNDRA = auto()
    WETLAND = auto()
    ALPINE = auto()
    CORAL_REEF = auto()

class ClimateEcosystemModel:
    """
    Advanced Climate and Ecosystem Simulation System
    
    Features:
    - Multi-factor climate modeling
    - Ecosystem dynamics simulation
    - Biodiversity estimation
    - Climate change projection
    """
    
    def __init__(
        self, 
        world_size: Tuple[int, int] = (512, 512),
        seed: int = 42
    ):
        """
        Initialize climate and ecosystem model
        
        :param world_size: Dimensions of the climate map
        :param seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.world_size = world_size
        
        # Initialize climate parameters
        self.temperature_map = self._generate_temperature_map()
        self.precipitation_map = self._generate_precipitation_map()
        self.wind_patterns = self._simulate_wind_patterns()
        
        # Generate ecosystem layers
        self.climate_zones = self._classify_climate_zones()
        self.ecosystem_map = self._generate_ecosystem_distribution()
        
        # Biodiversity estimation
        self.biodiversity_index = self._calculate_biodiversity()
    
    def _generate_temperature_map(self) -> np.ndarray:
        """
        Generate global temperature distribution
        
        :return: Temperature map
        """
        # Latitude-based temperature gradient
        y = np.linspace(0, 1, self.world_size[0])
        x = np.linspace(0, 1, self.world_size[1])
        Y, X = np.meshgrid(y, x)
        
        # Simulate temperature variation with latitude
        base_temp = 15  # Global mean temperature
        latitude_effect = 30 * (0.5 - Y)  # Temperature variation based on latitude
        noise = np.random.normal(0, 2, self.world_size)
        
        temperature = base_temp + latitude_effect + noise
        return temperature
    
    def _generate_precipitation_map(self) -> np.ndarray:
        """
        Generate global precipitation distribution
        
        :return: Precipitation map
        """
        # Simulate precipitation patterns
        base_precipitation = np.random.gamma(5, 2, self.world_size)
        
        # Apply spatial correlation
        precipitation = ndimage.gaussian_filter(base_precipitation, sigma=3)
        
        return precipitation
    
    def _simulate_wind_patterns(self) -> np.ndarray:
        """
        Simulate global wind circulation
        
        :return: Wind direction and intensity map
        """
        # Simulate Hadley cell and wind circulation
        y = np.linspace(0, 1, self.world_size[0])
        x = np.linspace(0, 1, self.world_size[1])
        Y, X = np.meshgrid(y, x)
        
        # Wind direction and intensity
        wind_direction = np.arctan2(0.5 - Y, 0.5 - X)
        wind_intensity = np.sqrt((0.5 - X)**2 + (0.5 - Y)**2)
        
        return wind_direction * wind_intensity
    
    def _classify_climate_zones(self) -> np.ndarray:
        """
        Classify climate zones based on temperature and precipitation
        
        :return: Climate zone map
        """
        # Multi-factor climate classification
        climate_map = np.zeros(self.world_size, dtype=int)
        
        for i, zone in enumerate(ClimateZone):
            # Climate zone rules
            if zone == ClimateZone.TROPICAL:
                mask = (self.temperature_map > 25) & (self.precipitation_map > 2000)
            elif zone == ClimateZone.DESERT:
                mask = (self.temperature_map > 30) & (self.precipitation_map < 250)
            elif zone == ClimateZone.TUNDRA:
                mask = (self.temperature_map < 0) & (self.precipitation_map < 500)
            else:
                # Random assignment for other zones
                mask = np.random.rand(*self.world_size) < 0.1
            
            climate_map[mask] = i
        
        return climate_map
    
    def _generate_ecosystem_distribution(self) -> np.ndarray:
        """
        Generate ecosystem distribution based on climate
        
        :return: Ecosystem type map
        """
        ecosystem_map = np.zeros(self.world_size, dtype=int)
        
        for i, ecosystem in enumerate(EcosystemType):
            # Ecosystem distribution rules
            if ecosystem == EcosystemType.RAINFOREST:
                mask = (self.climate_zones == ClimateZone.TROPICAL.value)
            elif ecosystem == EcosystemType.TUNDRA:
                mask = (self.climate_zones == ClimateZone.POLAR.value)
            elif ecosystem == EcosystemType.DESERT:
                mask = (self.climate_zones == ClimateZone.DESERT.value)
            else:
                # Random distribution for other ecosystems
                mask = np.random.rand(*self.world_size) < 0.05
            
            ecosystem_map[mask] = i
        
        return ecosystem_map
    
    def _calculate_biodiversity(self) -> np.ndarray:
        """
        Calculate biodiversity index
        
        :return: Biodiversity map
        """
        # Biodiversity based on ecosystem complexity and climate variation
        biodiversity = np.zeros(self.world_size)
        
        for ecosystem in EcosystemType:
            ecosystem_mask = (self.ecosystem_map == ecosystem.value)
            
            # Assign biodiversity scores
            if ecosystem in {
                EcosystemType.RAINFOREST, 
                EcosystemType.CORAL_REEF
            }:
                biodiversity[ecosystem_mask] = np.random.uniform(0.8, 1.0)
            elif ecosystem in {
                EcosystemType.DECIDUOUS_FOREST, 
                EcosystemType.WETLAND
            }:
                biodiversity[ecosystem_mask] = np.random.uniform(0.6, 0.8)
            else:
                biodiversity[ecosystem_mask] = np.random.uniform(0.2, 0.5)
        
        return biodiversity
    
    def simulate_climate_change(
        self, 
        temperature_increase: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate potential climate change impacts
        
        :param temperature_increase: Global temperature increase in °C
        :return: Climate change impact maps
        """
        # Modify temperature map
        modified_temp = self.temperature_map + temperature_increase
        
        # Estimate ecosystem shifts
        ecosystem_shift = np.zeros_like(self.ecosystem_map)
        
        for ecosystem in EcosystemType:
            # Simple ecosystem migration rules
            if ecosystem == EcosystemType.TUNDRA:
                # Tundra shrinks
                ecosystem_shift[modified_temp > 5] = EcosystemType.ALPINE.value
            elif ecosystem == EcosystemType.DESERT:
                # Deserts expand
                ecosystem_shift[modified_temp > 35] = EcosystemType.DESERT.value
        
        return {
            'temperature_change': modified_temp,
            'ecosystem_shift': ecosystem_shift
        }
    
    def visualize_climate_data(self):
        """
        Visualize climate and ecosystem simulation results
        """
        plt.figure(figsize=(20, 15))
        
        # Temperature Map
        plt.subplot(2, 3, 1)
        plt.title('Global Temperature Distribution')
        plt.imshow(self.temperature_map, cmap='RdYlBu_r')
        plt.colorbar(label='Temperature (°C)')
        
        # Precipitation Map
        plt.subplot(2, 3, 2)
        plt.title('Precipitation Distribution')
        plt.imshow(self.precipitation_map, cmap='Blues')
        plt.colorbar(label='Precipitation (mm)')
        
        # Climate Zones
        plt.subplot(2, 3, 3)
        plt.title('Climate Zones')
        plt.imshow(self.climate_zones, cmap='Set3')
        plt.colorbar(
            ticks=range(len(ClimateZone)),
            label='Climate Zones'
        )
        
        # Ecosystem Distribution
        plt.subplot(2, 3, 4)
        plt.title('Ecosystem Types')
        plt.imshow(self.ecosystem_map, cmap='Set2')
        plt.colorbar(
            ticks=range(len(EcosystemType)),
            label='Ecosystem Types'
        )
        
        # Biodiversity Index
        plt.subplot(2, 3, 5)
        plt.title('Biodiversity Index')
        plt.imshow(self.biodiversity_index, cmap='YlGnBu')
        plt.colorbar(label='Biodiversity Score')
        
        # Wind Patterns
        plt.subplot(2, 3, 6)
        plt.title('Global Wind Patterns')
        plt.imshow(self.wind_patterns, cmap='coolwarm')
        plt.colorbar(label='Wind Intensity')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Demonstrate climate and ecosystem simulation capabilities
    """
    # Initialize climate and ecosystem model
    climate_model = ClimateEcosystemModel(
        world_size=(512, 512),
        seed=42
    )
    
    # Visualize climate data
    climate_model.visualize_climate_data()
    
    # Simulate climate change
    climate_change_impact = climate_model.simulate_climate_change(
        temperature_increase=2.0
    )
    
    # Optional: Visualize climate change impacts
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.title('Original Temperature')
    plt.imshow(climate_model.temperature_map, cmap='RdYlBu_r')
    plt.colorbar()
    
    plt.subplot(122)
    plt.title('Temperature After Climate Change')
    plt.imshow(climate_change_impact['temperature_change'], cmap='RdYlBu_r')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
