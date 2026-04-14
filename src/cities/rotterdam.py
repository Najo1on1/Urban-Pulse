import logging
import geopandas as gpd
import pandas as pd
from src.base_ingestor import CityDataIngestor
from src.spatial_utils import load_and_project_data, build_time_series_matrix

class RotterdamIngestor(CityDataIngestor):
    def __init__(self):
        # Rotterdam bounding box
        super().__init__(city_name="rotterdam", bbox=[4.3875, 51.8700, 4.5690, 51.9650])

    def fetch_road_network(self):
        """Standard Overture fetch for Rotterdam."""
        conn = self._get_duckdb_connection()
        latest_release = self._get_latest_overture_release(conn)
        overture_url = f"s3://overturemaps-us-west-2/release/{latest_release}/theme=transportation/type=segment/*"
        
        query = f"""
        COPY (
            SELECT id, geometry, class AS road_class
            FROM read_parquet('{overture_url}', filename=true, hive_partitioning=1)
            WHERE bbox.xmin >= {self.bbox[0]} AND bbox.xmax <= {self.bbox[2]}
              AND bbox.ymin >= {self.bbox[1]} AND bbox.ymax <= {self.bbox[3]}
        ) TO '{self.roads_path}' (FORMAT PARQUET);
        """
        logging.info(f"Fetching Overture roads for {self.city_name.capitalize()}...")
        conn.execute(query)

    def two_pass_spatial_snap(self, detectors: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        ADVANCED LOGIC: Filters out cycleways and footways to ensure car traffic 
        only snaps to motorized infrastructure.
        """
        logging.info("Executing Two-Pass Filter for Rotterdam's cycling infrastructure...")
        
        # Define 'forbidden' road classes for car traffic
        non_motorized = ['cycleway', 'footway', 'pedestrian', 'path', 'sidewalk']
        
        # 1. Find the 3 nearest neighbors (instead of just 1) to give us options
        snapped = gpd.sjoin_nearest(
            detectors, 
            roads, 
            how="inner", 
            max_distance=50.0, 
            distance_col="snap_distance",
            exclusive=False
        )
        
        # 2. Filter logic: Sort by distance, then drop rows where the nearest is a bike path
        # but keep the next closest if it's a real road.
        valid_snaps = snapped[~snapped['road_class'].isin(non_motorized)]
        
        # 3. Final Selection: Keep only the single closest VALID road per detector
        final_mapping = valid_snaps.sort_values(by=['detector_id', 'snap_distance']).drop_duplicates(subset=['detector_id'])
        
        logging.info(f"Successfully mapped {len(final_mapping)} detectors to motorized roads.")
        return final_mapping[['detector_id', 'id']].rename(columns={'id': 'overture_segment_id'})

    def run_pipeline(self):
        logging.info(f"--- Starting Pipeline for {self.city_name.capitalize()} ---")
        self.fetch_road_network()
        
        detectors_file = self.utd19_dir / f"{self.city_name}_detectors.csv"
        flow_file = self.utd19_dir / f"{self.city_name}_flow.csv"
        output_matrix = self.processed_dir / f"{self.city_name}_universal_twin.parquet"
        
        # Load and project using our core utils
        roads_gdf, detectors_gdf = load_and_project_data(self.roads_path, detectors_file)
        
        # Execute the specialized Rotterdam snapping
        mapping = self.two_pass_spatial_snap(detectors_gdf, roads_gdf)
        
        # Build matrix
        build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    ingestor = RotterdamIngestor()
    ingestor.run_pipeline()
