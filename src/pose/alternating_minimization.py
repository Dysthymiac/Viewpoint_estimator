"""
Alternating minimization algorithm for animal pose estimation.

Following the specification for learning canonical 2D anatomy and per-image pose parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from pathlib import Path
import json
import time
from tqdm import tqdm

from .batch_detection_dataset import BatchDetectionDataset
from .pose_model import CanonicalPoseModel
from .pose_estimation import create_patch_assignments, visualize_pose_estimates
from .pose_utils import solve_transformation_unconstrained, procrustes_decomposition


def rotation_matrix_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert 3-parameter axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: 3D vector where magnitude is angle, direction is axis
        
    Returns:
        3x3 rotation matrix
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    
    axis = axis_angle / angle
    
    # Rodrigues' formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def compute_robust_centers(animal_patches: List[Dict]) -> Dict[str, Tuple[float, float, float]]:
    """
    Convert multiple patches per body part into single representative point.
    Handle outliers robustly using median.
    
    Args:
        animal_patches: List of patch dictionaries with keys: x, y, d, label
        
    Returns:
        Dictionary mapping body part name to (x, y, d) center coordinates
    """
    # Group patches by label
    patches_by_label = {}
    for patch in animal_patches:
        label = patch['label']
        if label not in patches_by_label:
            patches_by_label[label] = []
        patches_by_label[label].append((patch['x'], patch['y'], patch['d']))
    
    centers = {}
    for body_part, patches in patches_by_label.items():
        if len(patches) == 1:
            centers[body_part] = patches[0]
        else:
            # Use median for robustness against outliers
            coords = np.array(patches)
            centers[body_part] = tuple(np.median(coords, axis=0))
    
    return centers


def center_points(points_dict: Dict[str, Tuple]) -> Tuple[Dict[str, Tuple], Tuple]:
    """
    Center a set of points at their median.
    
    Args:
        points_dict: Dictionary mapping names to coordinate tuples
        
    Returns:
        centered_points: Dictionary with centered coordinates
        median_center: The center point that was subtracted
    """
    if not points_dict:
        return {}, (0, 0, 0)
    
    all_coords = np.array(list(points_dict.values()))
    median_center = tuple(np.median(all_coords, axis=0))
    
    centered = {}
    for part, coord in points_dict.items():
        centered[part] = tuple(np.array(coord) - np.array(median_center))
    
    return centered, median_center


def estimate_image_parameters(centers: Dict[str, Tuple], 
                            canonical_2d: Dict[str, Tuple[float, float]]) -> Optional[Dict]:
    """
    Jointly estimate rotation matrix R and scales (s_xy, s_d) for one image.
    
    Args:
        centers: Observed body part centers (x, y, d)
        canonical_2d: Canonical template coordinates (X, Y)
        
    Returns:
        Dictionary with 'R', 's_xy', 's_d' or None if estimation failed
    """
    detected_parts = [part for part in centers.keys() if part in canonical_2d]
    
    if len(detected_parts) < 3:
        return None  # Too few constraints
    
    # Center both observed and canonical points
    obs_centered, obs_center = center_points(centers)
    canon_points = {part: canonical_2d[part] for part in detected_parts}
    canon_centered, canon_center = center_points({part: (x, y, 0) for part, (x, y) in canon_points.items()})
    
    # Use analytical Procrustes solution instead of iterative optimization
    # Prepare point correspondences for Procrustes
    canonical_points = []
    observed_points = []
    
    for part in detected_parts:
        canonical_points.append([canon_centered[part][0], canon_centered[part][1], 0])
        observed_points.append(obs_centered[part])
    
    canonical_points = np.array(canonical_points)
    observed_points = np.array(observed_points)
    
    # Solve for transformation matrix analytically
    A = solve_transformation_unconstrained(observed_points, canonical_points)
    
    # Decompose into rotation + scaling
    R, s_xy, s_d = procrustes_decomposition(A)
    
    # Check if solution is valid
    if s_xy > 0 and s_d > 0 and np.isfinite(s_xy) and np.isfinite(s_d):
        # Compute error for validation
        error = 0
        for i, part in enumerate(detected_parts):
            canon_3d = canonical_points[i]
            projected = R @ canon_3d
            projected_scaled = np.array([s_xy * projected[0], s_xy * projected[1], s_d * projected[2]])
            observed = observed_points[i]
            error += np.sum((projected_scaled - observed)**2)
        
        return {
            'R': R, 
            's_xy': s_xy, 
            's_d': s_d,
            'error': error,
            'obs_center': obs_center,
            'canon_center': canon_center
        }
    
    return None


def update_canonical_template(dataset: BatchDetectionDataset,
                            all_image_params: Dict,
                            canonical_model: CanonicalPoseModel,
                            cluster_labels: Dict,
                            min_observations: int = 5) -> Dict[str, Tuple[float, float]]:
    """
    Update canonical template coordinates using all valid image parameter estimates.
    
    Args:
        dataset: Batch detection dataset
        all_image_params: Dictionary of estimated parameters per image
        canonical_model: Canonical pose model
        cluster_labels: Dictionary mapping semantic labels to cluster IDs
        min_observations: Minimum detections needed to update a body part
        
    Returns:
        new_canonical: Updated canonical 2D template
    """
    # Collect all backprojections
    backprojected_points = {}
    for part in canonical_model.body_parts:
        backprojected_points[part] = []
    
    processed = 0
    found_params = 0
    total_params = len(all_image_params)
    print(f"  Looking for {total_params} successful parameter estimates...")
    
    for detection_batch in dataset.iter_batches(100):  # Process in batches
        for detection in detection_batch:
            detection_id = f"{processed}"  # Simple ID for tracking
            
            if detection_id not in all_image_params:
                processed += 1
                continue
            
            found_params += 1
            
            # Extract patch data and convert to centers
            image_info = detection['image_info']
            patch_indices = detection['patch_indices']
            
            all_patch_coords = np.array(image_info['patch_coordinates'])
            all_patch_components = np.array(image_info['patch_components'])
            all_patch_depths = np.array(image_info['patch_depths'])
            
            patch_coords = all_patch_coords[patch_indices]
            patch_components = all_patch_components[patch_indices]
            patch_depths = all_patch_depths[patch_indices]
            
            # Use same approach as E-step: create_patch_assignments
            positions_3d = np.column_stack([patch_coords, patch_depths])
            cluster_assignments = np.argmax(patch_components, axis=1)
            
            patch_assignments = create_patch_assignments(
                cluster_assignments,
                cluster_labels,
                canonical_model,
                positions_3d
            )
            
            # Convert to centers format
            centers = {}
            for body_part, assignments in patch_assignments.items():
                if np.sum(assignments) > 0:
                    center_3d = np.average(positions_3d, axis=0, weights=assignments)
                    centers[body_part] = (float(center_3d[0]), float(center_3d[1]), float(center_3d[2]))
            
            if not centers:
                processed += 1
                continue
            
            # Debug: print what centers contains for first few detections
            if found_params <= 3:
                print(f"    Detection {detection_id} centers: {list(centers.keys())}")
            
            params = all_image_params[detection_id]
            R, s_xy, s_d = params['R'], params['s_xy'], params['s_d']
            obs_center, canon_center = params['obs_center'], params['canon_center']
            
            # Backproject each detected canonical body part
            for body_part, (x, y, d) in centers.items():
                if body_part in backprojected_points:
                    # Center the observation
                    centered_obs = np.array([x, y, d]) - np.array(obs_center)
                    
                    # Inverse scaling and rotation
                    scaled_obs = np.array([centered_obs[0]/s_xy, centered_obs[1]/s_xy, centered_obs[2]/s_d])
                    canonical_3d = R.T @ scaled_obs  # Inverse rotation
                    
                    # Extract 2D canonical coordinates (ignore z-component) 
                    canonical_2d_point = (canonical_3d[0], canonical_3d[1])
                    
                    # Add to backprojected points for this body part
                    backprojected_points[body_part].append(canonical_2d_point)
                    
                    # Debug: print first few backprojections
                    if found_params <= 3:
                        print(f"      Backprojected {body_part}: {canonical_2d_point}")
            
            processed += 1
    
    print(f"  Found {found_params}/{total_params} parameter estimates during M-step processing")
    
    # Update template: median of backprojections
    new_canonical = {}
    print(f"  M-step statistics:")
    for part in canonical_model.body_parts:
        observations_count = len(backprojected_points[part])
        print(f"    {part}: {observations_count} observations (min required: {min_observations})")
        
        if observations_count >= min_observations:
            coords = np.array(backprojected_points[part])
            new_canonical[part] = tuple(np.median(coords, axis=0))
    
    return new_canonical


def compute_template_change(old_template: Dict, new_template: Dict) -> float:
    """
    Compute L2 change in template for convergence check.
    
    Args:
        old_template: Previous canonical template
        new_template: New canonical template
        
    Returns:
        RMS change in template coordinates
    """
    change = 0
    count = 0
    for part in old_template:
        if part in new_template:
            change += np.sum((np.array(old_template[part]) - np.array(new_template[part]))**2)
            count += 1
    return np.sqrt(change / count) if count > 0 else float('inf')


def run_alternating_minimization(detections_dir: Path,
                                cluster_labels: Dict,
                                canonical_model_path: Path,
                                output_path: Path,
                                batch_size: int = 100,
                                max_epochs: int = 50,
                                convergence_threshold: float = 1e-4) -> None:
    """
    Main alternating minimization algorithm for animal pose estimation.
    
    Args:
        detections_dir: Directory containing detection pickle files
        cluster_labels: Dictionary mapping semantic labels to cluster IDs
        canonical_model_path: Path to canonical model config
        output_path: Path to save final canonical model
        batch_size: Batch size for processing
        max_epochs: Maximum number of iterations
        convergence_threshold: Convergence threshold for template change
    """
    print(f"Starting alternating minimization algorithm...")
    
    # Load batch dataset - no fallback to individual files
    dataset = BatchDetectionDataset(detections_dir)
    print(f"Dataset: {dataset.get_stats()}")
    
    # Load canonical model
    canonical_model = CanonicalPoseModel(str(canonical_model_path))
    
    # Initialize 2D canonical template from 3D means (X, Z coordinates, Y=0)
    canonical_2d = {}
    for body_part in canonical_model.body_parts:
        mean_3d = canonical_model.get_canonical_mean(body_part)
        canonical_2d[body_part] = (float(mean_3d[0]), float(mean_3d[2]))  # X, Z coordinates
    
    # Create reverse cluster lookup for performance
    cluster_to_semantic = {}
    for semantic_label, cluster_list in cluster_labels.items():
        for cluster_id in cluster_list:
            cluster_to_semantic[cluster_id] = semantic_label
    
    print(f"Initial canonical template:")
    for part, coords in canonical_2d.items():
        print(f"  {part}: {coords}")
    
    # Create output directory for visualizations
    output_dir = output_path.parent / "pose_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Main optimization loop
    for iteration in range(max_epochs):
        print(f"\nEpoch {iteration + 1}/{max_epochs}")
        start_time = time.time()
        
        # E-STEP: Estimate projection parameters for each image
        print("  E-step: Estimating image parameters...")
        all_image_params = {}
        successful_estimates = 0
        total_processed = 0
        
        # Get total number of batches for proper progress tracking
        total_detections = dataset.total_detections
        total_batches = (total_detections + batch_size - 1) // batch_size  # Ceiling division
        
        batch_count = 0
        pbar = tqdm(dataset.iter_batches(batch_size), desc="  Processing batches", total=total_batches)
        
        # Timing diagnostics
        total_data_extract_time = 0
        total_assignment_time = 0
        total_estimation_time = 0
        total_batch_overhead_time = 0
        
        for detection_batch in pbar:
            batch_overhead_start = time.time()
            batch_successful = 0
            batch_processed = 0
            
            for detection in detection_batch:
                detection_id = f"{total_processed}"
                
                # Extract and process detection data
                data_start = time.time()
                image_info = detection['image_info']
                patch_indices = detection['patch_indices']
                
                all_patch_coords = np.array(image_info['patch_coordinates'])
                all_patch_components = np.array(image_info['patch_components'])
                all_patch_depths = np.array(image_info['patch_depths'])
                
                patch_coords = all_patch_coords[patch_indices]
                patch_components = all_patch_components[patch_indices]
                patch_depths = all_patch_depths[patch_indices]
                
                # Create 3D positions (x, y, d)
                positions_3d = np.column_stack([patch_coords, patch_depths])
                cluster_assignments = np.argmax(patch_components, axis=1)
                total_data_extract_time += time.time() - data_start
                
                # Use existing patch assignment function
                assignment_start = time.time()
                patch_assignments = create_patch_assignments(
                    cluster_assignments, 
                    cluster_labels, 
                    canonical_model, 
                    positions_3d
                )
                
                # Convert to centers format for parameter estimation
                centers = {}
                for body_part, assignments in patch_assignments.items():
                    if np.sum(assignments) > 0:
                        # Compute weighted center
                        center_3d = np.average(positions_3d, axis=0, weights=assignments)
                        centers[body_part] = (float(center_3d[0]), float(center_3d[1]), float(center_3d[2]))
                total_assignment_time += time.time() - assignment_start
                
                if len(centers) >= 3:  # Need at least 3 points for parameter estimation
                    estimation_start = time.time()
                    params = estimate_image_parameters(centers, canonical_2d)
                    total_estimation_time += time.time() - estimation_start
                    
                    if params is not None:
                        all_image_params[detection_id] = params
                        successful_estimates += 1
                        batch_successful += 1
                
                total_processed += 1
                batch_processed += 1
            
            # Update progress bar with meaningful statistics
            current_success_rate = successful_estimates / total_processed if total_processed > 0 else 0
            batch_success_rate = batch_successful / batch_processed if batch_processed > 0 else 0
            
            batch_count += 1
            pbar.set_description(
                f"  Batch {batch_count}: {current_success_rate:.1%} success "
                f"({successful_estimates}/{total_processed}), "
                f"batch: {batch_success_rate:.1%}"
            )
            
            # Track batch overhead time
            total_batch_overhead_time += time.time() - batch_overhead_start
        
        iteration_time = time.time() - start_time
        success_rate = successful_estimates / total_processed if total_processed > 0 else 0
        
        print(f"  E-step completed in {iteration_time:.1f}s")
        print(f"  Success rate: {success_rate:.1%} ({successful_estimates}/{total_processed})")
        print(f"  Timing breakdown:")
        print(f"    Data extraction: {total_data_extract_time:.1f}s ({total_data_extract_time/iteration_time*100:.1f}%)")
        print(f"    Patch assignments: {total_assignment_time:.1f}s ({total_assignment_time/iteration_time*100:.1f}%)")
        print(f"    Parameter estimation: {total_estimation_time:.1f}s ({total_estimation_time/iteration_time*100:.1f}%)")
        print(f"    Batch overhead: {total_batch_overhead_time:.1f}s ({total_batch_overhead_time/iteration_time*100:.1f}%)")
        dataset_io_time = iteration_time - total_data_extract_time - total_assignment_time - total_estimation_time - total_batch_overhead_time
        print(f"    Dataset I/O: {dataset_io_time:.1f}s ({dataset_io_time/iteration_time*100:.1f}%)")
        
        if successful_estimates < 10:
            print("  Too few successful parameter estimates, stopping.")
            break
        
        # M-STEP: Update canonical template
        print("  M-step: Updating canonical template...")
        new_canonical_2d = update_canonical_template(dataset, all_image_params, canonical_model, cluster_labels)
        
        if not new_canonical_2d:
            print("  No template updates possible, stopping.")
            break
        
        # Check convergence
        template_change = compute_template_change(canonical_2d, new_canonical_2d)
        canonical_2d.update(new_canonical_2d)
        
        print(f"  Template change: {template_change:.6f}")
        # Template updated
        
        # Visualize progress after every epoch
        print(f"  Saving visualization for epoch {iteration + 1}...")
        visualize_pose_estimates(
            detections_dir, 
            cluster_labels, 
            canonical_model,
            output_dir, 
            iteration + 1
        )
        
        if template_change < convergence_threshold:
            print(f"  Converged after {iteration + 1} iterations")
            break
    
    # Update the canonical model with final 2D template
    for body_part, (x_2d, z_2d) in canonical_2d.items():
        if body_part in canonical_model.body_parts:
            # Update 3D mean with new X,Z coordinates (keep Y=0)
            old_mean = canonical_model.get_canonical_mean(body_part)
            new_mean = np.array([x_2d, 0.0, z_2d], dtype=np.float32)
            canonical_model.means[body_part] = new_mean
    
    # Save final canonical model in existing format
    output_config = {
        'body_parts': {},
        'connectivity': canonical_model.connectivity,
        'label_mapping': canonical_model.label_mapping
    }
    
    for body_part in canonical_model.body_parts:
        output_config['body_parts'][body_part] = {
            'mean_position': canonical_model.get_canonical_mean(body_part).tolist(),
            'covariance': canonical_model.get_canonical_covariance(body_part).tolist()
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_config, f, indent=2)
    
    print(f"\nAlternating minimization completed!")
    print(f"Final canonical model saved to {output_path}")
    print(f"Final template:")
    for part, coords in canonical_2d.items():
        print(f"  {part}: ({coords[0]:.3f}, {coords[1]:.3f})")