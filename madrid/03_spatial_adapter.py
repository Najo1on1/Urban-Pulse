import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Configure explainable logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_and_project_data(roads_path: Path, detectors_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads raw data and projects it to a metric CRS (UTM Zone 30N for Madrid) for accurate distance calculation."""
    logging.info("Loading Overture road network and UTD19 detector metadata...")
    
    # Load Roads (Parquet)
    roads_gdf = gpd.read_parquet(roads_path)
    if roads_gdf.crs is None:
        roads_gdf.set_crs(epsg=4326, inplace=True)
        
    # Load Detectors (CSV)
    detectors_df = pd.read_csv(detectors_path)
    detectors_gdf = gpd.GeoDataFrame(
        detectors_df, 
        geometry=gpd.points_from_xy(detectors_df.longitude, detectors_df.latitude),
        crs="EPSG:4326"
    )

    logging.info("Projecting geometries to metric CRS (EPSG:32630) to calculate distance in meters...")
    roads_metric = roads_gdf.to_crs(epsg=32630)
    detectors_metric = detectors_gdf.to_crs(epsg=32630)
    
    return roads_metric, detectors_metric

def snap_to_road(detectors: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, max_distance: float = 50.0) -> pd.DataFrame:
    """Snaps detector points to the nearest road segment within a maximum distance threshold."""
    logging.info(f"Snapping {len(detectors)} detectors to the nearest road within {max_distance} meters...")
    
    # Use spatial join nearest with a max distance threshold
    snapped = gpd.sjoin_nearest(
        detectors, 
        roads, 
        how="inner", 
        max_distance=max_distance,
        distance_col="snap_distance"
    )
    
    # Keep only the closest road segment if a point snaps to multiple overlapping segments
    snapped = snapped.sort_values(by=['detector_id', 'snap_distance']).drop_duplicates(subset=['detector_id'])
    
    successful_snaps = len(snapped)
    discarded = len(detectors) - successful_snaps
    
    logging.info(f"Snapped {successful_snaps} detectors successfully.")
    if discarded > 0:
        logging.warning(f"{discarded} detectors discarded (distance > {max_distance}m).")
        
    return snapped[['detector_id', 'id']].rename(columns={'id': 'overture_segment_id'})

def build_time_series_matrix(mapping_df: pd.DataFrame, flow_path: Path, output_path: Path) -> None:
    """Merges traffic flow with the spatial mapping and pivots into a time-series matrix."""
    logging.info("Loading 24-hour traffic flow data...")
    flow_df = pd.read_csv(flow_path)
    
    # Merge the flow data with our snapped Overture segment IDs
    merged = flow_df.merge(mapping_df, on="detector_id", how="inner")
    
    logging.info("Pivoting data into the final 'Universal Twin' time-series matrix...")
    # Rows: Timestamp, Columns: Overture Segment IDs, Values: Flow
    matrix = merged.pivot_table(
        index="timestamp", 
        columns="overture_segment_id", 
        values="flow", 
        aggfunc="sum",
        fill_value=0
    )
    
    # Save the final pristine matrix to Parquet
    matrix.columns = matrix.columns.astype(str) # Ensure string column names for Parquet
    matrix.to_parquet(output_path)
    
    logging.info(f"Phase 1 Complete! Matrix dimensions: {matrix.shape[0]} timesteps x {matrix.shape[1]} road segments.")
    logging.info(f"Saved final processed twin to: {output_path}")

def main():
    # Define paths
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    roads_file = raw_dir / "overture_spain" / "madrid_roads.parquet"
    detectors_file = raw_dir / "madrid_utd19" / "madrid_detectors.csv"
    flow_file = raw_dir / "madrid_utd19" / "madrid_flow.csv"
    output_matrix = processed_dir / "madrid_universal_twin.parquet"
    
    # Execute Pipeline
    roads_gdf, detectors_gdf = load_and_project_data(roads_file, detectors_file)
    mapping = snap_to_road(detectors_gdf, roads_gdf, max_distance=50.0)
    build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    main()
