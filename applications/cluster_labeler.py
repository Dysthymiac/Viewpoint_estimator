"""
Interactive Cluster Labeling Interface using Gradio

A clean, web-based interface for assigning semantic labels to analysis clusters.
All visualization parameters are exposed and adjustable in the GUI.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Optional

# Add parent directory to Python path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))

# Import existing utilities
from src.utils.config import load_config
from src.data.cvat_loader import CVATLoader
from src.data.coco_loader import COCOLoader
from src.models.dinov2_extractor import DINOv2PatchExtractor
from src.utils.analysis_utils import load_analysis_model

# Import extracted visualization functions
from src.visualization.cluster_visualization import (
    create_highlighted_clusters,
    create_probabilities_view,
    create_uncertainty_view, 
    create_labeled_clusters_view
)
try:
    from src.visualization.morphological_postprocessing import apply_morphological_cleanup
    MORPHOLOGICAL_AVAILABLE = True
except ImportError:
    MORPHOLOGICAL_AVAILABLE = False

class ClusterLabeler:
    """Main cluster labeling application."""
    
    def __init__(self):
        self.config = None
        self.loader = None
        self.extractor = None
        self.analysis_method = None
        self.annotations = []
        self.cluster_labels = {}
        self.n_clusters = 0
        
        # Semantic label options
        self.semantic_labels = [
            "background", "head", "body", "legs", "tail", "snout", "neck", "thighs",
            "belly", "back", "ears", "fur_pattern", "shadow", 
            "vegetation", "ground", "other"
        ]
        
        # Image cache
        self.image_cache = {}
    
    def load_data(self):
        """Load configuration, dataset, and analysis model."""
        try:
            # Load config
            self.config = load_config(Path("config_zebra_test.yaml"))
            
            # Load dataset
            dataset_root = Path(self.config['dataset']['root_path'])
            crop_to_bbox = self.config['dataset'].get('crop_to_bbox', False)
            # self.loader = CVATLoader(dataset_root, crop_to_bbox=crop_to_bbox)
            dataset_config = self.config['dataset']
            dataset_root = Path(dataset_config['root_path'])
            crop_to_bbox = dataset_config.get('crop_to_bbox', True)
            dataset_format = dataset_config.get('format', 'cvat')  # Default to CVAT for backward compatibility
    
            # Use images_dir to override root_path if provided
            if 'images_dir' in dataset_config and dataset_config['images_dir']:
                image_root = Path(dataset_config['images_dir'])
                if not image_root.is_absolute():
                    image_root = dataset_root / dataset_config['images_dir']
            else:
                image_root = dataset_root
            if dataset_format.lower() == 'coco':
                annotations_file = dataset_config['annotations_file']
                coco_json_path = dataset_root / annotations_file
                self.loader = COCOLoader(image_root, coco_json_path, crop_to_bbox=crop_to_bbox)
            elif dataset_format.lower() == 'cvat':
                self.loader = CVATLoader(dataset_root, crop_to_bbox=crop_to_bbox)
            self.annotations = self.loader.annotations.copy()

            # Initialize extractor
            model_config = self.config['model']
            self.extractor = DINOv2PatchExtractor(
                model_name=model_config['dinov2_model'],
                device=model_config['device'],
                image_size=model_config['image_size']
            )
            
            # Load analysis model
            analysis_config = self.config['analysis']
            model_path = Path(analysis_config['output_dir']) / analysis_config['model_filename']
            self.analysis_method = load_analysis_model(model_path)
            self.n_clusters = self.analysis_method.get_n_components()
            
            return f"✅ Loaded: {len(self.annotations)} images, {self.n_clusters} clusters ({self.analysis_method.get_method_name()})"
            
        except Exception as e:
            return f"❌ Error loading data: {e}"
    
    def get_image_data(self, image_idx: int):
        """Get cached image data or process if not cached."""
        if image_idx in self.image_cache:
            return self.image_cache[image_idx]
        
        annotation = self.annotations[image_idx]
        
        # Load and process image
        image = self.loader.load_image(annotation)
        patch_features, patch_coordinates, relative_patch_size = self.extractor.extract_patch_features(image)
        
        # Transform with analysis method
        raw_output = self.analysis_method.transform(patch_features)
        if raw_output.ndim == 1:
            # K-Means labels - convert to one-hot encoding
            patch_components = np.zeros((len(raw_output), self.n_clusters))
            patch_components[np.arange(len(raw_output)), raw_output] = 1.0
        else:
            patch_components = raw_output
        
        # Cache results
        data = {
            'image': image,
            'patch_components': patch_components,
            'patch_coordinates': patch_coordinates,
            'relative_patch_size': relative_patch_size,
            'annotation': annotation
        }
        self.image_cache[image_idx] = data
        return data
    
    def get_original_image(self, image_idx: int) -> Image.Image:
        """Get original image - only changes when image_idx changes."""
        if not self.annotations or image_idx >= len(self.annotations):
            return Image.new('RGB', (400, 300), (240, 240, 240))
        
        data = self.get_image_data(image_idx)
        return data['image']
    
    def create_highlighted_clusters(
        self, 
        image_idx: int, 
        selected_clusters: List[int],
        alpha: float,
        smooth: bool
    ) -> Image.Image:
        """Create cluster highlight overlay - uses extracted visualization function."""
        if not self.annotations or image_idx >= len(self.annotations):
            return Image.new('RGB', (400, 300), (240, 240, 240))
        
        data = self.get_image_data(image_idx)
        return create_highlighted_clusters(
            data['image'],
            data['patch_components'],
            data['patch_coordinates'],
            data['relative_patch_size'],
            selected_clusters,
            alpha,
            smooth
        )
    
    def create_probabilities_view(
        self, 
        image_idx: int,
        normalization: str,
        temperature: float,
        alpha: float,
        smooth: bool,
        component_indices: Optional[List[int]] = None,
        morphological_cleanup: bool = False,
        morph_operations: List[str] = None,
        morph_kernel_size: int = 3,
        morph_kernel_shape: str = "disk"
    ) -> Image.Image:
        """Create probabilities visualization - uses extracted visualization function."""
        if not self.annotations or image_idx >= len(self.annotations):
            return Image.new('RGB', (400, 300), (240, 240, 240))
        
        data = self.get_image_data(image_idx)
        return create_probabilities_view(
            data['image'],
            data['patch_components'],
            data['patch_coordinates'],
            data['relative_patch_size'],
            normalization,
            temperature,
            alpha,
            smooth,
            component_indices,
            morphological_cleanup,
            morph_operations,
            morph_kernel_size,
            morph_kernel_shape
        )
    
    def create_uncertainty_view(
        self, 
        image_idx: int,
        uncertainty_method: str,
        colormap: str,
        alpha: float,
        smooth: bool,
        component_indices: Optional[List[int]] = None,
        normalization: str = "linear",
        temperature: float = 1.0,
        morphological_cleanup: bool = False,
        morph_operations: List[str] = None,
        morph_kernel_size: int = 3,
        morph_kernel_shape: str = "disk"
    ) -> Image.Image:
        """Create uncertainty visualization - uses extracted visualization function."""
        if not self.annotations or image_idx >= len(self.annotations):
            return Image.new('RGB', (400, 300), (240, 240, 240))
        
        data = self.get_image_data(image_idx)
        return create_uncertainty_view(
            data['image'],
            data['patch_components'],
            data['patch_coordinates'],
            data['relative_patch_size'],
            uncertainty_method,
            colormap,
            alpha,
            smooth,
            component_indices,
            normalization,
            temperature,
            morphological_cleanup,
            morph_operations,
            morph_kernel_size,
            morph_kernel_shape
        )
    
    def create_labeled_clusters_view(
        self,
        image_idx: int,
        alpha: float,
        smooth: bool,
        morphological_cleanup: bool = False,
        morph_operations: List[str] = None,
        morph_kernel_size: int = 3,
        morph_kernel_shape: str = "disk"
    ) -> Image.Image:
        """Create visualization showing all labeled clusters - uses extracted visualization function."""
        if not self.annotations or image_idx >= len(self.annotations):
            return Image.new('RGB', (400, 300), (240, 240, 240))
        
        if not self.cluster_labels:
            data = self.get_image_data(image_idx)
            return data['image']
        
        data = self.get_image_data(image_idx)
        return create_labeled_clusters_view(
            data['image'],
            data['patch_components'],
            data['patch_coordinates'],
            data['relative_patch_size'],
            self.cluster_labels,
            alpha,
            smooth,
            morphological_cleanup,
            morph_operations,
            morph_kernel_size,
            morph_kernel_shape
        )
    
    def format_assignments_display(self) -> str:
        """Format current cluster assignments for display."""
        if not self.cluster_labels:
            return "No assignments yet"
        
        lines = []
        for label, clusters in self.cluster_labels.items():
            cluster_list = ", ".join(map(str, sorted(clusters)))
            lines.append(f"**{label}**: clusters {cluster_list}")
        
        # Show unassigned clusters  
        assigned_clusters = set()
        for clusters in self.cluster_labels.values():
            assigned_clusters.update(clusters)
        
        unassigned = [i for i in range(self.n_clusters) if i not in assigned_clusters]
        if unassigned:
            cluster_list = ", ".join(map(str, unassigned))
            lines.append(f"**unassigned**: clusters {cluster_list}")
        
        return "\n".join(lines)
    
    def assign_clusters_to_label(
        self, 
        selected_clusters: List[int], 
        semantic_label: str
    ) -> tuple[str, List[int]]:
        """Assign selected clusters to semantic label."""
        if not selected_clusters or not semantic_label:
            return self.format_assignments_display(), []
        
        # Remove clusters from any existing assignments
        for label in list(self.cluster_labels.keys()):
            self.cluster_labels[label] = [c for c in self.cluster_labels[label] if c not in selected_clusters]
            if not self.cluster_labels[label]:  # Remove empty labels
                del self.cluster_labels[label]
        
        # Add to new label
        if semantic_label not in self.cluster_labels:
            self.cluster_labels[semantic_label] = []
        self.cluster_labels[semantic_label].extend(selected_clusters)
        self.cluster_labels[semantic_label] = sorted(list(set(self.cluster_labels[semantic_label])))
        
        return self.format_assignments_display(), []  # Clear selection
    
    def select_clusters_by_label(self, semantic_label: str) -> List[int]:
        """Select all clusters assigned to a semantic label."""
        return self.cluster_labels.get(semantic_label, [])
    
    def export_labels(self) -> str:
        """Export cluster labels to JSON."""
        if not self.cluster_labels:
            return "No labels to export"
        
        export_path = Path("cluster_labels.json")
        with open(export_path, 'w') as f:
            json.dump(self.cluster_labels, f, indent=2)
        
        return f"✅ Labels exported to {export_path}"
    
    def load_labels(self, file_path: str) -> str:
        """Load cluster labels from JSON file."""
        try:
            if not file_path:
                return "No file selected"
            
            load_path = Path(file_path)
            if not load_path.exists():
                return f"❌ File not found: {load_path}"
            
            with open(load_path, 'r') as f:
                loaded_labels = json.load(f)
            
            # Validate the loaded labels structure
            if not isinstance(loaded_labels, dict):
                return "❌ Invalid file format: expected JSON object"
            
            # Convert string keys back to integers and validate cluster IDs
            validated_labels = {}
            for label, clusters in loaded_labels.items():
                if not isinstance(clusters, list):
                    return f"❌ Invalid format for label '{label}': expected list of cluster IDs"
                
                # Validate cluster IDs are within range
                for cluster_id in clusters:
                    if not isinstance(cluster_id, int) or cluster_id < 0 or cluster_id >= self.n_clusters:
                        return f"❌ Invalid cluster ID {cluster_id} for label '{label}'"
                
                validated_labels[label] = clusters
            
            # Replace current labels with loaded ones
            self.cluster_labels = validated_labels
            
            return f"✅ Labels loaded from {load_path}"
            
        except json.JSONDecodeError as e:
            return f"❌ Invalid JSON file: {e}"
        except Exception as e:
            return f"❌ Error loading labels: {e}"


def create_interface():
    """Create the Gradio interface."""
    labeler = ClusterLabeler()
    
    with gr.Blocks(title="Cluster Labeling Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🏷️ Interactive Cluster Labeling")
        
        # Initialize data
        with gr.Row():
            load_btn = gr.Button("Load Data", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False, value="Click 'Load Data' to begin")
        
        # Main interface (hidden until data loaded)
        with gr.Row(visible=False) as main_interface:
            # Left panel - Cluster assignment
            with gr.Column(scale=1):
                gr.Markdown("### Cluster Selection")
                
                cluster_checkboxes = gr.CheckboxGroup(
                    choices=[], label="Select Clusters", value=[]
                )
                
                with gr.Row():
                    semantic_dropdown = gr.Dropdown(
                        choices=labeler.semantic_labels,
                        label="Semantic Label",
                        value=labeler.semantic_labels[0]
                    )
                    assign_btn = gr.Button("Assign Selected", variant="secondary")
                
                gr.Markdown("### Current Assignments")
                assignments_display = gr.Markdown("No assignments yet")
                
                gr.Markdown("### Select by Label")
                with gr.Row():
                    label_selector = gr.Dropdown(
                        choices=[], label="Existing Labels", value=None
                    )
                    select_btn = gr.Button("Select These Clusters", variant="secondary")
                
                # Export section
                gr.Markdown("### Export")
                with gr.Row():
                    export_btn = gr.Button("Export Labels", variant="primary")
                    load_btn_labels = gr.Button("Load Labels", variant="secondary")
                
                load_file = gr.File(
                    label="Select JSON file to load",
                    file_types=[".json"],
                    visible=False
                )
                export_status = gr.Textbox(label="Export/Load Status", interactive=False)
            
            # Right panel - Visualization
            with gr.Column(scale=2):
                gr.Markdown("### Image Navigation")
                image_slider = gr.Slider(
                    minimum=0, maximum=100, step=1, value=0,
                    label="Image Index"
                )
                
                gr.Markdown("### Visualization")
                with gr.Row():
                    original_image = gr.Image(label="Original Image", type="pil")
                    highlighted_image = gr.Image(label="Selected Clusters", type="pil")
                
                # Third view selection
                with gr.Row():
                    third_view_type = gr.Radio(
                        choices=["Probabilities", "Uncertainty", "Labeled Clusters"], 
                        label="Additional View", 
                        value="Probabilities"
                    )
                
                third_view_image = gr.Image(label="Probabilities/Uncertainty/Labeled", type="pil")
                
                # ALL visualization parameters exposed
                with gr.Accordion("Visualization Parameters", open=False):
                    with gr.Tabs():
                        # Core parameters tab
                        with gr.TabItem("Core"):
                            normalization = gr.Radio(
                                choices=["linear", "softmax"], 
                                label="Normalization", 
                                value="linear"
                            )
                            temperature = gr.Slider(0.1, 5.0, value=1.0, label="Temperature")
                            alpha = gr.Slider(0.1, 1.0, value=0.7, label="Overlay Alpha")
                            smooth = gr.Checkbox(label="Smooth Rendering", value=False)
                        
                        # Uncertainty parameters tab
                        with gr.TabItem("Uncertainty"):
                            uncertainty_method = gr.Dropdown(
                                choices=["entropy", "max_ratio", "variance"],
                                label="Uncertainty Method",
                                value="entropy"
                            )
                            uncertainty_colormap = gr.Dropdown(
                                choices=["viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
                                label="Uncertainty Colormap",
                                value="viridis"
                            )
                        
                        # Component selection tab
                        with gr.TabItem("Components"):
                            component_selection = gr.CheckboxGroup(
                                choices=[], 
                                label="Select Components (empty = all)", 
                                value=[]
                            )
                        
                        # Morphological processing tab
                        with gr.TabItem("Morphological"):
                            morphological_cleanup = gr.Checkbox(
                                label="Enable Morphological Cleanup", 
                                value=False,
                                visible=MORPHOLOGICAL_AVAILABLE
                            )
                            morph_operations = gr.CheckboxGroup(
                                choices=["opening", "closing", "erosion", "dilation"],
                                label="Morphological Operations",
                                value=["opening", "closing"],
                                visible=MORPHOLOGICAL_AVAILABLE
                            )
                            morph_kernel_size = gr.Slider(
                                1, 10, step=1, value=3,
                                label="Kernel Size",
                                visible=MORPHOLOGICAL_AVAILABLE
                            )
                            morph_kernel_shape = gr.Dropdown(
                                choices=["disk", "square", "diamond"],
                                label="Kernel Shape",
                                value="disk",
                                visible=MORPHOLOGICAL_AVAILABLE
                            )
                            
                            if not MORPHOLOGICAL_AVAILABLE:
                                gr.Markdown("⚠️ Morphological processing not available. Install scikit-image to enable.")
        
        # Event handlers
        def load_data_handler():
            status = labeler.load_data()
            if "✅" in status:
                # Update interface with loaded data
                cluster_choices = [f"Cluster {i}" for i in range(labeler.n_clusters)]
                component_choices = [f"Component {i}" for i in range(labeler.n_clusters)]
                return (
                    status,
                    gr.update(visible=True),  # Show main interface
                    gr.update(choices=cluster_choices),  # Update cluster checkboxes
                    gr.update(maximum=len(labeler.annotations)-1),  # Update image slider
                    gr.update(choices=component_choices)  # Update component selection
                )
            else:
                return status, gr.update(visible=False), gr.update(), gr.update(), gr.update()
        
        def update_original_image(image_idx):
            """Update original image - only depends on image index."""
            return labeler.get_original_image(int(image_idx))
        
        def update_highlighted_clusters(image_idx, selected_cluster_names, alpha, smooth):
            """Update highlighted clusters view."""
            selected_clusters = []
            if selected_cluster_names:
                for name in selected_cluster_names:
                    cluster_id = int(name.split()[-1])  # Extract number from "Cluster X"
                    selected_clusters.append(cluster_id)
            
            return labeler.create_highlighted_clusters(
                int(image_idx), selected_clusters, alpha, smooth
            )
        
        def update_third_view(
            image_idx, third_view_type, normalization, temperature, alpha, smooth,
            uncertainty_method, uncertainty_colormap, component_names,
            morphological_cleanup, morph_operations, morph_kernel_size, morph_kernel_shape
        ):
            """Update probabilities, uncertainty, or labeled clusters view."""
            # Convert component names to indices
            component_indices = None
            if component_names:
                component_indices = [int(name.split()[-1]) for name in component_names]
            
            if third_view_type == "Probabilities":
                return labeler.create_probabilities_view(
                    int(image_idx), normalization, temperature, alpha, smooth,
                    component_indices, morphological_cleanup, morph_operations,
                    morph_kernel_size, morph_kernel_shape
                )
            elif third_view_type == "Uncertainty":
                return labeler.create_uncertainty_view(
                    int(image_idx), uncertainty_method, uncertainty_colormap, alpha, smooth,
                    component_indices, normalization, temperature, morphological_cleanup,
                    morph_operations, morph_kernel_size, morph_kernel_shape
                )
            else:  # Labeled Clusters
                return labeler.create_labeled_clusters_view(
                    int(image_idx), alpha, smooth, morphological_cleanup,
                    morph_operations, morph_kernel_size, morph_kernel_shape
                )
        
        def assign_handler(selected_cluster_names, semantic_label):
            """Handle cluster assignment."""
            selected_clusters = []
            if selected_cluster_names:
                for name in selected_cluster_names:
                    cluster_id = int(name.split()[-1])
                    selected_clusters.append(cluster_id)
            
            assignments_text, cleared_selection = labeler.assign_clusters_to_label(
                selected_clusters, semantic_label
            )
            
            # Update label selector choices
            label_choices = list(labeler.cluster_labels.keys())
            
            return (
                assignments_text,
                cleared_selection,  # Clear cluster selection
                gr.update(choices=label_choices)  # Update label selector
            )
        
        def select_by_label_handler(semantic_label):
            """Handle selection by existing label."""
            if semantic_label:
                cluster_indices = labeler.select_clusters_by_label(semantic_label)
                cluster_names = [f"Cluster {i}" for i in cluster_indices]
                return cluster_names
            return []
        
        def export_handler():
            """Handle label export."""
            return labeler.export_labels()
        
        def show_load_file_handler():
            """Show file selector when load button is clicked."""
            return gr.update(visible=True)
        
        def load_handler(file_obj):
            """Handle label loading from file."""
            if file_obj is None:
                return "No file selected", gr.update(), gr.update(), gr.update(visible=False)
            
            # Load the labels
            status = labeler.load_labels(file_obj.name)
            
            # Update interface elements
            assignments_text = labeler.format_assignments_display()
            label_choices = list(labeler.cluster_labels.keys())
            
            return (
                status,
                assignments_text,
                gr.update(choices=label_choices),
                gr.update(visible=False)  # Hide file selector after loading
            )
        
        # Wire up events - separate for each view
        
        # Load data
        load_btn.click(
            load_data_handler,
            outputs=[status_text, main_interface, cluster_checkboxes, image_slider, component_selection]
        )
        
        # Original image - only updates when image index changes
        image_slider.change(
            update_original_image,
            inputs=[image_slider],
            outputs=[original_image]
        )
        
        # Highlighted clusters - updates when clusters or highlight params change
        for component in [image_slider, cluster_checkboxes, alpha, smooth]:
            component.change(
                update_highlighted_clusters,
                inputs=[image_slider, cluster_checkboxes, alpha, smooth],
                outputs=[highlighted_image]
            )
        
        # Third view - updates when relevant params change
        for component in [image_slider, third_view_type, normalization, temperature, alpha, smooth,
                         uncertainty_method, uncertainty_colormap, component_selection,
                         morphological_cleanup, morph_operations, morph_kernel_size, morph_kernel_shape]:
            component.change(
                update_third_view,
                inputs=[image_slider, third_view_type, normalization, temperature, alpha, smooth,
                       uncertainty_method, uncertainty_colormap, component_selection,
                       morphological_cleanup, morph_operations, morph_kernel_size, morph_kernel_shape],
                outputs=[third_view_image]
            )
        
        # Assignment handlers
        assign_btn.click(
            assign_handler,
            inputs=[cluster_checkboxes, semantic_dropdown],
            outputs=[assignments_display, cluster_checkboxes, label_selector]
        )
        
        select_btn.click(
            select_by_label_handler,
            inputs=[label_selector],
            outputs=[cluster_checkboxes]
        )
        
        export_btn.click(
            export_handler,
            outputs=[export_status]
        )
        
        load_btn_labels.click(
            show_load_file_handler,
            outputs=[load_file]
        )
        
        load_file.change(
            load_handler,
            inputs=[load_file],
            outputs=[export_status, assignments_display, label_selector, load_file]
        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )