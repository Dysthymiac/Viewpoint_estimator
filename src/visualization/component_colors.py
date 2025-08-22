"""Component to color mapping utilities."""

from __future__ import annotations

import numpy as np


def get_kelly_colors() -> list[tuple[float, float, float]]:
    """Get Kelly's 22 visually distinct colors as RGB tuples (0-1 range)."""
    # Kelly's maximum contrast colors (excluding white/black)
    kelly_hex = [
        '#F2F3F4',  # White (excluded in practice)
        '#222222',  # Black (excluded in practice) 
        '#F3C300',  # Vivid Yellow
        '#875692',  # Strong Purple
        '#F38400',  # Vivid Orange
        '#A1CAF1',  # Very Light Blue
        '#BE0032',  # Vivid Red
        '#C2B280',  # Grayish Yellow
        '#848482',  # Medium Gray
        '#008856',  # Vivid Green
        '#E68FAC',  # Strong Purplish Pink
        '#0067A5',  # Strong Blue
        '#F99379',  # Strong Yellowish Pink
        '#604E97',  # Strong Violet
        '#F6A600',  # Vivid Orange Yellow
        '#B3446C',  # Strong Purplish Red
        '#DCD300',  # Vivid Greenish Yellow
        '#882D17',  # Strong Reddish Brown
        '#8DB600',  # Vivid Yellowish Green
        '#654522',  # Deep Yellowish Brown
        '#E25822',  # Vivid Reddish Orange
        '#2B3D26'   # Dark Olive Green
    ]
    
    # Convert hex to RGB (0-1 range) and exclude white/black
    rgb_colors = []
    for hex_color in kelly_hex[2:]:  # Skip white and black
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0  
        b = int(hex_color[4:6], 16) / 255.0
        rgb_colors.append((r, g, b))
    
    return rgb_colors


def generate_hsv_colors(n_colors: int) -> np.ndarray:
    """Generate visually distinct colors using HSV color space."""
    import colorsys
    
    colors = []
    for i in range(n_colors):
        # Distribute hues evenly around color wheel
        hue = i / n_colors
        # Use high saturation and value for vivid colors
        saturation = 0.8
        value = 0.9
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return np.array(colors)


def get_color_palette(n_components: int) -> np.ndarray:
    """Get color palette for n components."""
    if n_components <= 3:
        # Use RGB colors for ≤3 components
        if n_components == 1:
            return np.array([[1.0, 0.0, 0.0]])  # Red
        elif n_components == 2:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Red, Green
        else:  # n_components == 3
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # RGB
    else:
        kelly_colors = get_kelly_colors()
        
        if n_components <= len(kelly_colors):
            # Use Kelly colors for 4-20 components
            selected_colors = kelly_colors[:n_components]
            return np.array(selected_colors)
        else:
            # Fallback to HSV-generated colors for >20 components
            return generate_hsv_colors(n_components)