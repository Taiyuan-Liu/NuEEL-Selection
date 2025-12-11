from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import numpy as np
import numba as nb
import sys
sys.path.append('/sdf/data/neutrino/software/spine/src')
import spine.math as sm


def compute_shower_dqdx(
        interaction, 
        r=3, 
        min_segment_size=.9):
    """Caculate shower dedx.

    Args:
        interaction: reco or truth
        r:radius
        min_segment_size:
    """

    interaction_dedx = 0
        
    num_particles_effective = 0
    for i in range(len(interaction.particles)): 
        if interaction.particles[i].is_primary == False:
            continue
        if interaction.particles[i].is_valid == False:
            continue
        if(interaction.particles[i].pid > 1):
            continue
        
        interaction_startpoint = interaction.particles[i].start_point.reshape(1, -1)

        interaction_points_vertex_dist = cdist(interaction.particles[i].points, interaction_startpoint)
        interaction_points_mask = interaction_points_vertex_dist.squeeze() < r       
        interaction_selected_points = interaction.particles[i].points[interaction_points_mask]
        if interaction_selected_points.shape[0] < 2:
            continue
        dx = np.max(interaction_points_vertex_dist[interaction_points_mask])
        if dx < min_segment_size:
            continue
        
        dq = np.sum(interaction.particles[i].depositions[interaction_points_mask])
        
        interaction_dedx += dq/dx
        num_particles_effective += 1

    if num_particles_effective > 1:
        pass
        #print("Warning: dedx calculation - more than one primary & valid particle found in interaction, return 0")
        interaction_dedx = 0.0
    
    return interaction_dedx



def cluster_dedx(
    voxels: nb.float64[:, :],
    values: nb.float64[:],
    start: nb.float64[:],
    max_dist: nb.float64 = 5.0,
    anchor: nb.boolean = True,
) -> nb.float64[:]:
    """Computes the initial local dE/dx of a cluster.

    Parameters
    ----------
    voxels : np.ndarray
        (N, 3) Voxel coordinates
    values : np.ndarray
        (N) Voxel values
    start : np.ndarray
        (3) Start point w.r.t. which to compute the local dE/dx
    max_dist : float, default 5.0
        Neighborhood radius around the point used to compute the dE/dx
    anchor : bool, default False
        If true, anchor the start point to the closest cluster point

    Returns
    -------
    float
        Local dE/dx value around the start point
    """
    # Sanity check
    assert (
        voxels.shape[1] == 3
    ), "The shape of the input is not compatible with voxel coordinates."

    # If necessary, anchor start point to the closest cluster point
    if anchor:
        dists = cdist(start.reshape(1, -1), voxels).flatten()
        start = voxels[np.argmin(dists)].astype(start.dtype)  # Dirty

    # If max_dist is set, limit the set of voxels to those within a sphere of
    # radius max_dist around the start point
    dists = cdist(start.reshape(1, -1), voxels).flatten()
    if max_dist > 0.0:
        index = np.where(dists <= max_dist)[0]
        if len(index) < 2:
            return 0.0

        values, dists = values[index], dists[index]

    # Compute the total energy in the neighborhood and the max distance, return ratio
    if np.max(dists) == 0.0:
        return 0.0

    return np.sum(values) / np.max(dists)



def extract_particle_features(particle):
    """extract particle attributes. Return a dict.

    Args:
        particle: Particle object e.g. reco.particles[0]
    
    Returns:
        dict: particle features
    """
    start_dedx = 0.0
    if len(particle.points) > 0 and len(particle.depositions) > 0:
        start_dedx = cluster_dedx(
            particle.points,
            particle.depositions,
            particle.start_point,
            max_dist=5.0,
            anchor=False
        )
    
    # beam direction deviation
    th = -0.101  # radians
    beam_direction = np.array([0, np.sin(th), np.cos(th)])
    
    particle_direction = particle.start_dir
    
    angle_wrt_beam = 0.0
    if particle_direction is not None:
        cos_angle = np.dot(particle_direction, beam_direction)
        angle_wrt_beam = np.arccos(cos_angle) * 180.0 / np.pi
    
    return {
        "primary": particle.is_primary,
        "valid": particle.is_valid,
        "particle_id": particle.id,
        "particle_pid": particle.pid,
        "reco_ke": particle.ke,  
        "angle_wrt_beam": angle_wrt_beam,  
        "start_dedx": start_dedx,
        "start_point": particle.start_point,
        "start_dir": particle.start_dir,
        #"points": particle.points,
        #"depositions": particle.depositions # Too much RAM consumption
    }

def calculate_angular_spread(start_point, points, lambda_r=14.0):
    """Calculate angular spread of an interaction.
    
    Args:
        start_point: interaction start point
        points: all points of the interaction
        lambda_r: attenuation length for weighting
    
    Returns:
        angular spread value
    """
    if len(points) == 0:
        return -1 
    
    # 1. Calculate distance x_i from each point to start point
    distances = np.linalg.norm(points - start_point, axis=1)  # shape: (N,)
    
    # 2. Calculate weights w_i = exp(-x_i / λ_r)
    #weights = np.exp(-distances / lambda_r)  # shape: (N,)
    weights = np.ones(len(distances))
    
    # 3. Calculate unit direction vector n̂_i from start point to each point
    direction_vectors = points - start_point  # shape: (N, 3)
    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)  # shape: (N, 1)
    # Avoid division by zero
    valid_mask = (norms.flatten() > 0)
    if not valid_mask.any():
        return -1
    
    unit_direction_vectors = np.zeros_like(direction_vectors)
    unit_direction_vectors[valid_mask] = direction_vectors[valid_mask] / norms[valid_mask]
    
    # 4. Calculate weighted average direction vector v̄
    weighted_sum = np.sum(weights[:, np.newaxis] * unit_direction_vectors, axis=0)  # shape: (3,)
    weighted_sum_norm = np.linalg.norm(weighted_sum)
    if weighted_sum_norm == 0:
        return -1
    #mean_direction = weighted_sum / weighted_sum_norm  # shape: (3,)
    th = -0.101
    mean_direction = np.array([0, np.sin(th), np.cos(th)])
    
    # 5. Calculate angular spread δ_d
    # δ_d = Σ[w_i * (1 - n̂_i · v̄)] / Σ[w_i]
    dot_products = np.sum(unit_direction_vectors * mean_direction, axis=1)  # shape: (N,)
    numerator = np.sum(weights * (1 - dot_products))
    denominator = np.sum(weights)
    
    if denominator == 0:
        return -1
    
    delta_d = numerator / denominator
    return delta_d

def pca_analysis(points):
    """Calculate 3D PCA eigenvalues and eigenvectors.
    
    Args:
        points: np.array, shape (N, 3), all 3D point coordinates [x, y, z]
    
    Returns:
        eigenvalues: np.array, shape (3,), three eigenvalues [λ1, λ2, λ3], sorted in descending order
        eigenvectors: np.array, shape (3, 3), three eigenvectors, each column is one eigenvector
        centroid: np.array, shape (3,), centroid of the points
    """
    if len(points) == 0:
        return np.array([-1, -1, -1]), np.zeros((3, 3)), np.zeros(3)
    
    # Step 1: Calculate centroid
    centroid = np.mean(points, axis=0)  # shape: (3,)
    
    # Step 2: Center the point cloud (relative to centroid)
    centered_points = points - centroid  # shape: (N, 3)
    
    # Step 3: Build covariance matrix (3x3)
    cov_matrix = (centered_points.T @ centered_points) / len(points)  # shape: (3, 3)
    
    # Step 4: Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 5: Sort by eigenvalue in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 6: Calculate PC1-Beam angle
    th = -0.101  # radians
    beam_direction = np.array([0, np.sin(th), np.cos(th)])
    pc1 = eigenvectors[:, 0]
    angle = np.degrees(np.arccos(np.abs(np.dot(pc1, beam_direction))))
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'angle_wrt_beam': angle,
    }


def check_proximity_contact(
    indices_to_check,
    event_ids,
    reco_ids,
    distance_threshold=5.0
):
    """Check if interactions have point cloud proximity contact with other interactions in the same event.
    
    Args:
        indices_to_check: list of interaction indices that passed previous cuts
        event_ids: list of event IDs for all interactions
        reco_ids: list of reco IDs for all interactions
        distance_threshold: distance threshold in cm for contact detection
    
    Returns:
        indices_pass_proximity: list of interaction indices that pass proximity cut (isolated)
        proximity_results: dict with detailed proximity check results for each interaction
    """
    import yaml
    from spine.driver import Driver
    
    print("=" * 80)
    print(f"Cut: Point Cloud Proximity Check")
    print(f"Excluding interactions with point cloud distance < {distance_threshold} cm")
    print(f"to any other interaction in the same event")
    print("=" * 80)
    
    # Base configuration template
    cfg_template = '''
base:
  verbosity: info
build:
  mode: both
  fragments: false
  particles: true
  interactions: true
  units: cm
io:
  reader:
    file_keys: DATA_PATH
    name: hdf5
'''
    
    indices_pass_proximity = []
    proximity_results = {}
    
    print(f"\nProcessing {len(indices_to_check)} interactions...\n")
    
    # Process each interaction that needs to check
    for idx in indices_to_check:
        
        evt_id = event_ids[idx]
        reco_id = reco_ids[idx]
        file_number = str(evt_id)[:4]
        
        # Create driver for this file
        file_path = f'/sdf/data/neutrino/ndlar/spine/prod/microprod_n4p1_v2/output_spine/MicroProdN4p1_NDComplex_FHC.flow2supera.full.000{file_number}.LARCV_spine.h5'
        
        cfg = cfg_template.replace('DATA_PATH', file_path)
        file_driver = Driver(yaml.safe_load(cfg))
        
        # Find the target event and collect all interactions in it
        event_data = None
        for data in file_driver:
            if data['run_info'].event == evt_id:
                event_data = {}
                
                for reco, truth in data['interaction_matches_r2t']:
                    # Collect all points from all particles
                    all_points = []
                    for particle in reco.particles:
                        if len(particle.points) > 0:
                            all_points.append(particle.points)
                    
                    if len(all_points) > 0:
                        event_data[reco.id] = np.vstack(all_points)
                    else:
                        event_data[reco.id] = np.array([]).reshape(0, 3)
                
                break

        current_points = event_data[reco_id]
        
        # Build KDTree for current interaction
        current_tree = cKDTree(current_points)
        
        # Check contact with other interactions in same event
        has_contact = False
        contact_interactions = []
        min_distance = np.inf
        
        for other_reco_id, other_points in event_data.items():
            if other_reco_id == reco_id:
                continue
            
            if len(other_points) == 0:
                continue
            
            # Query nearest neighbor distance
            distances, _ = current_tree.query(other_points, k=1)
            min_dist_to_other = np.min(distances)
            
            # Update minimum distance
            if min_dist_to_other < min_distance:
                min_distance = min_dist_to_other
            
            # Check if contact (distance below threshold)
            if min_dist_to_other < distance_threshold:
                has_contact = True
                contact_interactions.append((other_reco_id, min_dist_to_other))
        
        # Save results
        proximity_results[idx] = {
            'has_contact': has_contact,
            'contact_interactions': contact_interactions,
            'min_distance': min_distance,
            'reason': f'Contact with {len(contact_interactions)} interactions' if has_contact else 'Isolated'
        }
        
        # Pass if no contact
        if not has_contact:
            indices_pass_proximity.append(idx)
    
    # Print summary
    print(f"\n✓ Proximity check complete!")
    print(f"\nPass proximity cut: {len(indices_pass_proximity)} / {len(indices_to_check)}")
    print(f"Pass rate: {len(indices_pass_proximity)/len(indices_to_check)*100:.2f}%")
    print(f"Excluded: {len(indices_to_check) - len(indices_pass_proximity)} interactions")
    print("=" * 80)
    
    return indices_pass_proximity, proximity_results