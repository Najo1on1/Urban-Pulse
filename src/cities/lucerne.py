import logging
import geopandas as gpd
import pandas as pd
from src.base_ingestor import CityDataIngestor
from src.spatial_utils import load_and_project_data, build_time_series_matrix

class LucerneIngestor(CityDataIngestor):
    def __init__(self):
        # Lucerne bounding box
        super().__init__(city_name="lucerne", bbox=[8.2435, 47.0180, 8.3585, 47.0850])

    def fetch_road_network(self):
        """Fetch Overture roads for Lucerne, including bridge/tunnel metadata."""
        conn = self._get_duckdb_connection()
        latest_release = self._get_latest_overture_release(conn)
        overture_url = f"s3://overturemaps-us-west-2/release/{latest_release}/theme=transportation/type=segment/*"
        
        # We explicitly select 'level' and 'layer' to handle 3D positioning
        query = f"""
        COPY (
            SELECT id, geometry, class AS road_class, level, layer
            FROM read_parquet('{overture_url}', filename=true, hive_partitioning=1)
            WHERE bbox.xmin >= {self.bbox[0]} AND bbox.xmax <= {self.bbox[2]}
              AND bbox.ymin >= {self.bbox[1]} AND bbox.ymax <= {self.bbox[3]}
        ) TO '{self.roads_path}' (FORMAT PARQUET);
        """
        logging.info(f"Fetching Overture roads for {self.city_name.capitalize()} with 3D Metadata...")
        conn.execute(query)

    def spatial_join_3d(self, detectors: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        3D LOGIC: Prioritizes bridges/tunnels for detectors tagged as such.
        """
        logging.info("Executing 3D-Aware Spatial Join for Lucerne...")
        
        # Perform initial join to get nearest candidates
        snapped = gpd.sjoin_nearest(
            detectors, roads, how="inner", max_distance=50.0, distance_col="snap_distance"
        )

        # 1. Identify 3D sensors vs Surface sensors
        is_3d = snapped['location_type'] == 'tunnel_or_bridge'
        
        # 2. For 3D sensors, prefer roads where level != 0 or layer is present
        # For surface sensors, prefer level == 0
        snapped['is_3d_road'] = snapped['level'].fillna(0).astype(int) != 0
        
        # We apply a 'penalty' distance to non-matching types to push them down the priority list
        snapped.loc[is_3d & ~snapped['is_3d_road'], 'snap_distance'] += 100
        snapped.loc[~is_3d & snapped['is_3d_road'], 'snap_distance'] += 100

        # 3. Final selection: pick the 'closest' after penalty
        final_mapping = snapped.sort_values(by=['detector_id', 'snap_distance']).drop_duplicates(subset=['detector_id'])
        
        logging.info(f"Successfully mapped {len(final_mapping)} detectors with 3D priority.")
        return final_mapping[['detector_id', 'id']].rename(columns={'id': 'overture_segment_id'})

    def run_pipeline(self):
        logging.info(f"--- Starting Pipeline for {self.city_name.capitalize()} ---")
        self.fetch_road_network()
        
        detectors_file = self.utd19_dir / f"{self.city_name}_detectors.csv"
        flow_file = self.utd19_dir / f"{self.city_name}_flow.csv"
        output_matrix = self.processed_dir / f"{self.city_name}_universal_twin.parquet"
        
        roads_gdf, detectors_gdf = load_and_project_data(self.roads_path, detectors_file)
        mapping = self.spatial_join_3d(detectors_gdf, roads_gdf)
        build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    ingestor = LucerneIngestor()
    ingestor.run_pipeline()
