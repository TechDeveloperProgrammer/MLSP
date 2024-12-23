import os
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

class VisualizationType(Enum):
    """Comprehensive visualization types"""
    SCATTER = auto()
    LINE = auto()
    BAR = auto()
    HEATMAP = auto()
    HISTOGRAM = auto()
    BOX_PLOT = auto()
    NETWORK = auto()
    GEOGRAPHIC = auto()
    TERRAIN = auto()
    CUSTOM = auto()

class ColorPalette(Enum):
    """Predefined color palettes"""
    SEQUENTIAL = auto()
    DIVERGING = auto()
    QUALITATIVE = auto()
    TERRAIN = auto()
    CUSTOM = auto()

@dataclass
class VisualizationConfig:
    """Comprehensive visualization configuration"""
    visualization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = 'Unnamed Visualization'
    description: str = ''
    type: VisualizationType = VisualizationType.SCATTER
    color_palette: ColorPalette = ColorPalette.SEQUENTIAL
    custom_palette: Optional[List[str]] = None
    interactive: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['png', 'svg'])

class DataVisualizer:
    """
    Advanced Data Visualization and Analysis Framework
    
    Features:
    - Multi-format visualization
    - Interactive and static plotting
    - Advanced data analysis
    - Customizable visualization styles
    """
    
    def __init__(
        self, 
        output_dir: str = '/home/veronicae/CascadeProjects/MLSP/visualizations'
    ):
        """
        Initialize data visualizer
        
        :param output_dir: Directory to store visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _select_color_palette(
        self, 
        config: VisualizationConfig
    ) -> Union[str, List[str]]:
        """
        Select color palette
        
        :param config: Visualization configuration
        :return: Selected color palette
        """
        if config.custom_palette:
            return config.custom_palette
        
        palettes = {
            ColorPalette.SEQUENTIAL: 'Blues',
            ColorPalette.DIVERGING: 'RdBu',
            ColorPalette.QUALITATIVE: 'Set2',
            ColorPalette.TERRAIN: 'terrain'
        }
        
        return palettes.get(config.color_palette, 'viridis')
    
    def visualize_data(
        self, 
        data: Union[np.ndarray, pd.DataFrame], 
        config: Optional[VisualizationConfig] = None
    ) -> Dict[str, str]:
        """
        Visualize data with multiple options
        
        :param data: Input data
        :param config: Visualization configuration
        :return: Visualization file paths
        """
        # Create default configuration if not provided
        if not config:
            config = VisualizationConfig()
        
        # Convert numpy array to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Select color palette
        palette = self._select_color_palette(config)
        
        # Visualization outputs
        visualization_paths = {}
        
        # Static matplotlib visualization
        plt.figure(figsize=(10, 6))
        plt.title(config.title)
        
        if config.type == VisualizationType.SCATTER:
            sns.scatterplot(data=data, palette=palette)
        
        elif config.type == VisualizationType.LINE:
            sns.lineplot(data=data, palette=palette)
        
        elif config.type == VisualizationType.BAR:
            sns.barplot(data=data, palette=palette)
        
        elif config.type == VisualizationType.HEATMAP:
            sns.heatmap(data, cmap=palette, annot=True)
        
        elif config.type == VisualizationType.HISTOGRAM:
            sns.histplot(data=data, palette=palette)
        
        elif config.type == VisualizationType.BOX_PLOT:
            sns.boxplot(data=data, palette=palette)
        
        plt.tight_layout()
        
        # Save static visualizations
        for fmt in config.export_formats:
            static_path = os.path.join(
                self.output_dir, 
                f'{config.visualization_id}_static.{fmt}'
            )
            plt.savefig(static_path)
            visualization_paths[f'static_{fmt}'] = static_path
        
        plt.close()
        
        # Interactive Plotly visualization
        if config.interactive:
            if config.type == VisualizationType.SCATTER:
                fig = px.scatter(
                    data, 
                    title=config.title, 
                    color_discrete_sequence=palette
                )
            
            elif config.type == VisualizationType.LINE:
                fig = px.line(
                    data, 
                    title=config.title, 
                    color_discrete_sequence=palette
                )
            
            elif config.type == VisualizationType.BAR:
                fig = px.bar(
                    data, 
                    title=config.title, 
                    color_discrete_sequence=palette
                )
            
            # Save interactive HTML
            interactive_path = os.path.join(
                self.output_dir, 
                f'{config.visualization_id}_interactive.html'
            )
            fig.write_html(interactive_path)
            visualization_paths['interactive'] = interactive_path
        
        return visualization_paths
    
    def visualize_network(
        self, 
        graph: nx.Graph, 
        config: Optional[VisualizationConfig] = None
    ) -> Dict[str, str]:
        """
        Visualize network graph
        
        :param graph: NetworkX graph
        :param config: Visualization configuration
        :return: Visualization file paths
        """
        # Create default configuration
        if not config:
            config = VisualizationConfig(
                type=VisualizationType.NETWORK,
                title='Network Visualization'
            )
        
        # Static network visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        nx.draw(
            graph, 
            pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=500, 
            font_size=10
        )
        plt.title(config.title)
        
        # Save static network visualization
        visualization_paths = {}
        for fmt in config.export_formats:
            static_path = os.path.join(
                self.output_dir, 
                f'{config.visualization_id}_network_static.{fmt}'
            )
            plt.savefig(static_path)
            visualization_paths[f'static_{fmt}'] = static_path
        
        plt.close()
        
        # Interactive network visualization
        if config.interactive:
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10
                )
            )
            
            # Add edges and nodes
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            for node in graph.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['text'] += (str(node),)
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=config.title,
                    showlegend=False,
                    hovermode='closest'
                )
            )
            
            # Save interactive network visualization
            interactive_path = os.path.join(
                self.output_dir, 
                f'{config.visualization_id}_network_interactive.html'
            )
            fig.write_html(interactive_path)
            visualization_paths['interactive'] = interactive_path
        
        return visualization_paths

def main():
    """Demonstration of data visualization system"""
    # Initialize data visualizer
    visualizer = DataVisualizer()
    
    # Generate synthetic terrain data
    np.random.seed(42)
    terrain_data = np.random.rand(100, 100)
    
    # Create visualization configuration
    terrain_config = VisualizationConfig(
        title='Terrain Elevation Heatmap',
        type=VisualizationType.HEATMAP,
        color_palette=ColorPalette.TERRAIN,
        interactive=True
    )
    
    # Visualize terrain data
    visualization_paths = visualizer.visualize_data(
        terrain_data, 
        config=terrain_config
    )
    
    print("Terrain Visualization Paths:")
    print(json.dumps(visualization_paths, indent=2))
    
    # Create network graph
    G = nx.erdos_renyi_graph(50, 0.1)
    
    # Visualize network
    network_config = VisualizationConfig(
        title='Random Network Graph',
        type=VisualizationType.NETWORK,
        interactive=True
    )
    
    network_paths = visualizer.visualize_network(G, config=network_config)
    
    print("\nNetwork Visualization Paths:")
    print(json.dumps(network_paths, indent=2))

if __name__ == '__main__':
    main()
