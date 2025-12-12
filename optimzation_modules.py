"""
Optimization modules for cut parameter tuning using F1 score.
This module contains shared functions for applying cuts and calculating metrics.
"""

import numpy as np
from scipy.spatial import cKDTree
from spine.utils.geo import Geometry


def apply_topology_cut(reco_topology, reco_particle_info, allowed_topologies=['1e', '1g']):
    """Apply topology cut.
    
    Args:
        reco_topology: list of reconstructed topologies
        reco_particle_info: list of particle information for each interaction
        allowed_topologies: list of allowed topology types
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in range(len(reco_topology)):
        if reco_topology[i] in allowed_topologies:
            # Check if all particles are EM particles (e or gamma)
            flag_non_em_particle_exist = False
            for j, info in enumerate(reco_particle_info[i]):
                if info['particle_pid'] > 100:  # not e or g
                    flag_non_em_particle_exist = True
                    break
            if not flag_non_em_particle_exist:
                indices_after_cut.append(i)
    
    return indices_after_cut


def apply_fiducial_cut(indices_prev_cut, reco_vertex, margin_module=1.5, margin_detector=5.0):
    """Apply fiducial volume cut.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_vertex: list of reconstructed vertices
        margin_module: margin for module containment (cm)
        margin_detector: margin for detector containment (cm)
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    geo = Geometry('ndlar')
    geo.define_containment_volumes(margin=margin_module, mode='module')
    geo1 = Geometry('ndlar')
    geo1.define_containment_volumes(margin=margin_detector, mode='detector')
    
    cathode_x = [-300, -200, -100, 0, 100, 200, 300]
    
    for i in indices_prev_cut:
        # Check if vertex is within fiducial volume
        if not geo.check_containment(reco_vertex[i]):
            continue
        if not geo1.check_containment(reco_vertex[i]):
            continue
        
        # Check if vertex x is too close to cathode surfaces
        vertex_x = reco_vertex[i][0]
        too_close_to_cathode = False
        for boundary_x in cathode_x:
            if abs(vertex_x - boundary_x) < margin_module:
                too_close_to_cathode = True
                break
        
        if not too_close_to_cathode:
            indices_after_cut.append(i)
    
    return indices_after_cut


def apply_angle_cut(indices_prev_cut, reco_particle_info, angle_max=7.0):
    """Apply angle cut on primary particles.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_particle_info: list of particle information
        angle_max: maximum angle wrt beam direction (degrees)
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in indices_prev_cut:
        pass_angle_cut = True
        for particle_info in reco_particle_info[i]:
            if not particle_info['primary']:
                continue
            if not particle_info['valid']:
                continue
            
            angle = particle_info['angle_wrt_beam']
            if angle > angle_max:
                pass_angle_cut = False
                break
        
        if pass_angle_cut:
            indices_after_cut.append(i)
    
    return indices_after_cut


def apply_dedx_cut(indices_prev_cut, reco_particle_info, dedx_min=-1.2, dedx_max=2.5):
    """Apply dE/dx cut.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_particle_info: list of particle information dictionaries
        dedx_min: minimum dE/dx (MeV/cm)
        dedx_max: maximum dE/dx (MeV/cm)
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in indices_prev_cut:
        particle_list = reco_particle_info[i]
        
        # Count primary and valid particles
        num_primary_valid_particles = sum(1 for p in particle_list if p['primary'] and p['valid'])
        
        # If not exactly 1 primary valid particle, pass through
        if num_primary_valid_particles == 0:
            indices_after_cut.append(i)
            continue
        elif num_primary_valid_particles > 1:
            indices_after_cut.append(i)
            continue
        
        # Find the primary and valid particle and check start_dedx
        for particle_dict in particle_list:
            if particle_dict['primary'] and particle_dict['valid']:
                dedx = particle_dict['start_dedx']
                
                # Apply dE/dx cut
                if dedx_min < dedx < dedx_max:
                    indices_after_cut.append(i)
                break
    
    return indices_after_cut


def apply_etheta2_cut(indices_prev_cut, reco_particle_info, etheta2_max=3.0):
    """Apply E*theta^2 cut.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_particle_info: list of particle information
        etheta2_max: maximum E*theta^2 (MeV*rad^2)
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in indices_prev_cut:
        pass_etheta2_cut = True
        for particle_info in reco_particle_info[i]:
            if not particle_info['primary']:
                continue
            if not particle_info['valid']:
                continue
            
            ke = particle_info['reco_ke']
            angle_rad = np.radians(particle_info['angle_wrt_beam'])
            etheta2 = ke * angle_rad**2
            
            if etheta2 > etheta2_max:
                pass_etheta2_cut = False
                break
        
        if pass_etheta2_cut:
            indices_after_cut.append(i)
    
    return indices_after_cut


def apply_ke_cut(indices_prev_cut, reco_particle_info, ke_min=150.0):
    """Apply kinetic energy cut.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_particle_info: list of particle information
        ke_min: minimum kinetic energy (MeV)
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in indices_prev_cut:
        pass_ke_cut = False
        for particle_info in reco_particle_info[i]:
            if not particle_info['primary']:
                continue
            if not particle_info['valid']:
                continue
            
            ke = particle_info['reco_ke']
            if ke > ke_min:
                pass_ke_cut = True
                break
        
        if pass_ke_cut:
            indices_after_cut.append(i)
    
    return indices_after_cut


def apply_pca_cut(indices_prev_cut, reco_pca_variables, 
                  pca_longitudinal_min=-2, pca_longitudinal_max=99960,
                  pca_transverse_major_min=0.04, pca_transverse_major_max=9990.4,
                  pca_transverse_minor_min=0.02, pca_transverse_minor_max=9990.2,
                  pca_pc1_angle_max=99999999999):
    """Apply PCA cuts.
    
    Args:
        indices_prev_cut: list of indices from previous cut
        reco_pca_variables: list of PCA variables
        pca_longitudinal_min: minimum longitudinal spread
        pca_longitudinal_max: maximum longitudinal spread
        pca_transverse_major_min: minimum major transverse spread
        pca_transverse_major_max: maximum major transverse spread
        pca_transverse_minor_min: minimum minor transverse spread
        pca_transverse_minor_max: maximum minor transverse spread
        pca_pc1_angle_max: maximum PC1 angle wrt beam
    
    Returns:
        indices_after_cut: list of indices that pass the cut
    """
    indices_after_cut = []
    
    for i in indices_prev_cut:
        pca_vars = reco_pca_variables[i]
        
        # Check if pca_vars is a dictionary (new format)
        if isinstance(pca_vars, dict):
            eigenvalues = pca_vars['eigenvalues']
            pc1_angle = pca_vars['angle_wrt_beam']
        else:
            # Old format: tuple (eigenvalues, eigenvectors, centroid)
            eigenvalues, eigenvectors, centroid = pca_vars
            # Calculate angle manually
            th = -0.101  # radians
            beam_direction = np.array([0, np.sin(th), np.cos(th)])
            pc1 = eigenvectors[:, 0]
            cos_angle = np.abs(np.dot(pc1, beam_direction))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            pc1_angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        # Calculate PCA features
        longitudinal_spread = np.sqrt(eigenvalues[0])
        transverse_major_spread = np.sqrt(eigenvalues[1]) / np.sqrt(eigenvalues[0])
        transverse_minor_spread = np.sqrt(eigenvalues[2]) / np.sqrt(eigenvalues[0])
        
        # Apply all PCA cuts
        pass_all_cuts = True
        
        if not (pca_longitudinal_min < longitudinal_spread < pca_longitudinal_max):
            pass_all_cuts = False
        
        if not (pca_transverse_major_min < transverse_major_spread < pca_transverse_major_max):
            pass_all_cuts = False
        
        if not (pca_transverse_minor_min < transverse_minor_spread < pca_transverse_minor_max):
            pass_all_cuts = False
        
        if pca_pc1_angle_max < 90 and pc1_angle > pca_pc1_angle_max:
            pass_all_cuts = False
        
        if pass_all_cuts:
            indices_after_cut.append(i)
    
    return indices_after_cut


def apply_proximity_cut(indices_prev_cut, file_indices, event_ids, reco_ids, 
                       reco_particle_info, distance_threshold=0.5):
    
    import yaml
    from spine.driver import Driver
    import pickle
    import os
    
    # Load or initialize cache
    cache_dir = '/sdf/home/l/liuty/NuEEL/upload/data'
    cache_file = os.path.join(cache_dir, 'proximity_cache.pkl')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            proximity_cache = pickle.load(f)
    else:
        proximity_cache = {}
    
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
    
    # Process each interaction that needs to check
    for idx in indices_prev_cut:
        evt_id = event_ids[idx]
        reco_id = reco_ids[idx]
        
        # Create cache key WITHOUT threshold: (event_id, reco_id)
        cache_key = (evt_id, reco_id)
        
        # Check cache first for the minimum distance and closest reco_id
        if cache_key in proximity_cache:
            min_distance, closest_reco_id = proximity_cache[cache_key]
            #print(evt_id, reco_id, "Cache hit: ", min_distance, closest_reco_id)
            # Use cached distance to determine if passes threshold
            if min_distance >= distance_threshold:
                indices_pass_proximity.append(idx)
            continue
        
        # Cache miss - need to compute the minimum distance
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
        
        # Find the MINIMUM distance to any other interaction in same event
        min_distance = float('inf')
        closest_reco_id = None
        
        for other_reco_id, other_points in event_data.items():
            if other_reco_id == reco_id:
                continue
            
            if len(other_points) == 0:
                continue
            
            # Query nearest neighbor distance
            distances, _ = current_tree.query(other_points, k=1)
            min_dist_to_other = np.min(distances)
            
            # Track the overall minimum distance and closest reco_id
            if min_dist_to_other < min_distance:
                min_distance = min_dist_to_other
                closest_reco_id = other_reco_id
        
        # Save the MINIMUM DISTANCE and CLOSEST RECO_ID to cache
        proximity_cache[cache_key] = (min_distance, closest_reco_id)
        
        # Pass if distance is above threshold
        if min_distance >= distance_threshold:
            indices_pass_proximity.append(idx)
    
    # Save updated cache
    with open(cache_file, 'wb') as f:
        pickle.dump(proximity_cache, f)
    
    return indices_pass_proximity


def apply_all_cuts(reco_topology, reco_vertex, reco_particle_info, reco_dedx_list, 
                   reco_pca_variables, params, enable_flags, 
                   file_indices=None, event_ids=None, reco_ids=None):
    """Apply all cuts sequentially based on parameters and enable flags.
    
    Args:
        reco_topology: list of reconstructed topologies
        reco_vertex: list of reconstructed vertices
        reco_particle_info: list of particle information
        reco_dedx_list: list of dE/dx values
        reco_pca_variables: list of PCA variables
        params: dict of cut parameters
        enable_flags: dict of enable/disable flags for each cut
        file_indices: list of file indices (required for proximity cut)
        event_ids: list of event IDs (required for proximity cut)
        reco_ids: list of reco IDs (required for proximity cut)
    
    Returns:
        indices_after_all_cuts: list of indices that pass all cuts
    """
    # DEBUG: Set to True to print intermediate values
    DEBUG_PRINT = False
    
    total_events = len(reco_topology)
    if DEBUG_PRINT:
        print(f"\n{'='*80}")
        print(f"DEBUG: apply_all_cuts - Total events: {total_events}")
        print(f"{'='*80}")
    
    # Cut 1: Topology
    if enable_flags.get('ENABLE_TOPOLOGY_CUT', True):
        indices = apply_topology_cut(reco_topology, reco_particle_info, 
                                    params.get('RECO_TOPOLOGY_ALLOWED', ['1e', '1g']))
        if DEBUG_PRINT:
            print(f"After TOPOLOGY cut: {len(indices)}/{total_events} events ({len(indices)/total_events*100:.2f}%)")
    else:
        indices = list(range(len(reco_topology)))
        if DEBUG_PRINT:
            print(f"TOPOLOGY cut DISABLED: {len(indices)}/{total_events} events")
    
    # Cut 2: Fiducial Volume
    prev_count = len(indices)
    if enable_flags.get('ENABLE_FIDUCIAL_CUT', True):
        indices = apply_fiducial_cut(indices, reco_vertex, 
                                    params.get('MARGIN_MODULE', 1.5),
                                    params.get('MARGIN_DETECTOR', 5.0))
        if DEBUG_PRINT:
            print(f"After FIDUCIAL cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"FIDUCIAL cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 3: Angle
    prev_count = len(indices)
    if enable_flags.get('ENABLE_ANGLE_CUT', True):
        indices = apply_angle_cut(indices, reco_particle_info, 
                                 params.get('ANGLE_CUT_MAX', 7.0))
        if DEBUG_PRINT:
            print(f"After ANGLE cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"ANGLE cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 4: dE/dx
    prev_count = len(indices)
    if enable_flags.get('ENABLE_DEDX_CUT', True):
        indices = apply_dedx_cut(indices, reco_particle_info,
                                params.get('DEDX_CUT_MIN', -1.2),
                                params.get('DEDX_CUT_MAX', 2.5))
        if DEBUG_PRINT:
            print(f"After DEDX cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"DEDX cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 5: E*theta^2
    prev_count = len(indices)
    if enable_flags.get('ENABLE_ETHETA2_CUT', True):
        indices = apply_etheta2_cut(indices, reco_particle_info,
                                   params.get('ETHETA2_CUT_MAX', 3.0))
        if DEBUG_PRINT:
            print(f"After ETHETA2 cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"ETHETA2 cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 6: KE
    prev_count = len(indices)
    if enable_flags.get('ENABLE_KE_CUT', True):
        indices = apply_ke_cut(indices, reco_particle_info,
                              params.get('KE_CUT_MIN', 150.0))
        if DEBUG_PRINT:
            print(f"After KE cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"KE cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 7: PCA
    prev_count = len(indices)
    if enable_flags.get('ENABLE_PCA_CUT', False):
        indices = apply_pca_cut(indices, reco_pca_variables,
                               params.get('PCA_LONGITUDINAL_MIN', -2),
                               params.get('PCA_LONGITUDINAL_MAX', 99960),
                               params.get('PCA_TRANSVERSE_MAJOR_MIN', 0.04),
                               params.get('PCA_TRANSVERSE_MAJOR_MAX', 9990.4),
                               params.get('PCA_TRANSVERSE_MINOR_MIN', 0.02),
                               params.get('PCA_TRANSVERSE_MINOR_MAX', 9990.2),
                               params.get('PCA_PC1_ANGLE_MAX', 99999999999))
        if DEBUG_PRINT:
            print(f"After PCA cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
    elif DEBUG_PRINT:
        print(f"PCA cut DISABLED: {len(indices)}/{prev_count} events")
    
    # Cut 8: Proximity
    prev_count = len(indices)
    print(prev_count)
    if enable_flags.get('ENABLE_PROXIMITY_CUT', False):
        # Check if required data is available
        if file_indices is not None and event_ids is not None and reco_ids is not None:
            indices = apply_proximity_cut(indices, file_indices, event_ids, reco_ids,
                                        reco_particle_info,
                                        params.get('PROXIMITY_DISTANCE_THRESHOLD', 0.5))
            if DEBUG_PRINT:
                print(f"After PROXIMITY cut: {len(indices)}/{prev_count} events ({len(indices)/prev_count*100:.2f}%) [cumulative: {len(indices)}/{total_events}]")
        else:
            if DEBUG_PRINT:
                print(f"PROXIMITY cut SKIPPED (missing required data): {len(indices)}/{prev_count} events")
    elif DEBUG_PRINT:
        print(f"PROXIMITY cut DISABLED: {len(indices)}/{prev_count} events")
    
    if DEBUG_PRINT:
        print(f"{'='*80}")
        print(f"FINAL: {len(indices)}/{total_events} events pass all cuts ({len(indices)/total_events*100:.2f}%)")
        print(f"{'='*80}\n")
    
    return indices


def calculate_efficiency_purity_f1(nuone_data, full_spill_data, params, enable_flags):
    """Calculate efficiency, purity, and F1 score for given cut parameters.
    
    Args:
        nuone_data: dict containing nuone dataset arrays
        full_spill_data: dict containing full_spill dataset arrays
        params: dict of cut parameters
        enable_flags: dict of enable/disable flags
    
    Returns:
        efficiency: fraction of true 1e events selected
        purity: fraction of selected events that are true 1e
        f1_score: harmonic mean of efficiency and purity
        nuone_counts: tuple (passed, total) for nuone dataset
        full_spill_counts: tuple (true_1e_selected, selected) for full_spill dataset
    """
    # Apply cuts to nuone data (for efficiency)
    # Disable proximity cut for nuone (efficiency calculation) due to dataset limitations
    nuone_enable_flags = enable_flags.copy()
    nuone_enable_flags['ENABLE_PROXIMITY_CUT'] = False
    
    nuone_indices = apply_all_cuts(
        nuone_data['reco_topology'],
        nuone_data['reco_vertex'],
        nuone_data['reco_particle_info'],
        nuone_data['reco_dedx_list'],
        nuone_data['reco_pca_variables'],
        params,
        nuone_enable_flags,
        file_indices=nuone_data.get('file_index'),
        event_ids=nuone_data.get('event_id'),
        reco_ids=nuone_data.get('reco_id')
    )
    
    # Count true 1e events in nuone (should be all of them for pure sample)
    total_true_1e = sum(1 for t in nuone_data['true_topology'] if t == '1e')
    
    # Count how many true 1e events passed all cuts
    true_positive = sum(1 for i in nuone_indices if nuone_data['true_topology'][i] == '1e')
    
    # Calculate efficiency
    efficiency = true_positive / total_true_1e if total_true_1e > 0 else 0
    
    # Apply cuts to full_spill data (for purity)
    full_spill_indices = apply_all_cuts(
        full_spill_data['reco_topology'],
        full_spill_data['reco_vertex'],
        full_spill_data['reco_particle_info'],
        full_spill_data['reco_dedx_list'],
        full_spill_data['reco_pca_variables'],
        params,
        enable_flags,
        file_indices=full_spill_data.get('file_index'),
        event_ids=full_spill_data.get('event_id'),
        reco_ids=full_spill_data.get('reco_id')
    )
    
    # Count selected events and how many are true 1e with correct interaction mode
    selected = len(full_spill_indices)
    true_1e_selected = sum(1 for i in full_spill_indices 
                          if full_spill_data['true_topology'][i] == '1e' 
                          and full_spill_data['interaction_mode'][i] == 7)
    
    # Calculate purity using efficiency and expected signal rate
    background_count = selected - true_1e_selected
    signal_expectation = efficiency * 0.015 * 0.01 * 20483
    
    if background_count + signal_expectation > 0:
        purity = signal_expectation / (background_count + signal_expectation)
    else:
        purity = 0
    
    # Calculate F1 score
    if efficiency + purity > 0:
        f1_score = 2 * (efficiency * purity) / (efficiency + purity)
    else:
        f1_score = 0
    
    return efficiency, purity, f1_score, (true_positive, total_true_1e), (true_1e_selected, selected)
