import logging
import geopandas as gpd

def apply_drift_correction(detectors: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, tolerance: float):
    """
    Applies London Drift Correction (Relaxed Spatial Search).
    Ensures the output column name matches the expected 'overture_segment_id'.
    """
    logging.info("Applying London Drift Correction (Relaxed Spatial Search)...")
    
    # Increase tolerance by 50% for drifting sensors
    snapped = gpd.sjoin_nearest(detectors, roads, how="inner", max_distance=tolerance * 1.5)
    
    # RENAME the ID column so the pivot table can find it
    return snapped[['detector_id', 'id']].rename(columns={'id': 'overture_segment_id'})
