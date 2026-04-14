import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

def load_and_project_data(roads_path: Path, detectors_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads raw data and projects it to a metric CRS (UTM Zone 30N)."""
    logging.info("Loading road network and detector metadata...")
    
    roads_gdf = gpd.read_parquet(roads_path)
    if roads_gdf.crs is None:
        roads_gdf.set_crs(epsg=4326, inplace=True)
        
    detectors_df = pd.read_csv(detectors_path)
    detectors_gdf = gpd.GeoDataFrame(
        detectors_df, 
        geometry=gpd.points_from_xy(detectors_df.longitude, detectors_df.latitude),
        crs="EPSG:4326"
    )

    logging.info("Projecting geometries to metric CRS (EPSG:32630)...")
    roads_metric = roads_gdf.to_crs(epsg=32630)
    detectors_metric = detectors_gdf.to_crs(epsg=32630)
    
    return roads_metric, detectors_metric

def snap_to_road(detectors: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, max_distance: float = 50.0) -> pd.DataFrame:
    """Snaps detector points to the nearest road segment within a distance threshold."""
    logging.info(f"Snapping {len(detectors)} detectors to the nearest road within {max_distance} meters...")
    
    snapped = gpd.sjoin_nearest(
        detectors, roads, how="inner", max_distance=max_distance, distance_col="snap_distance"
    )
    snapped = snapped.sort_values(by=['detector_id', 'snap_distance']).drop_duplicates(subset=['detector_id'])
    
    successful_snaps = len(snapped)
    discarded = len(detectors) - successful_snaps
    
    logging.info(f"Snapped {successful_snaps} detectors successfully.")
    if discarded > 0:
        logging.warning(f"{discarded} detectors discarded (distance > {max_distance}m).")
        
    return snapped[['detector_id', 'id']].rename(columns={'id': 'overture_segment_id'})

def build_time_series_matrix(mapping_df: pd.DataFrame, flow_path: Path, output_path: Path) -> None:
    """Merges traffic flow with the spatial mapping and pivots into a time-series matrix."""
    logging.info("Loading traffic flow data...")
    flow_df = pd.read_csv(flow_path)
    merged = flow_df.merge(mapping_df, on="detector_id", how="inner")
    
    logging.info("Pivoting data into the 'Universal Twin' time-series matrix...")
    matrix = merged.pivot_table(
        index="timestamp", columns="overture_segment_id", values="flow", aggfunc="sum", fill_value=0
    )
    
    matrix.columns = matrix.columns.astype(str)
    matrix.to_parquet(output_path)
    logging.info(f"Matrix saved to: {output_path}")
