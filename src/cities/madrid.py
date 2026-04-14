import logging
from src.base_ingestor import CityDataIngestor
from src.spatial_utils import load_and_project_data, snap_to_road, build_time_series_matrix

class MadridIngestor(CityDataIngestor):
    def __init__(self):
        # Pass the city name and specific bounding box to the parent class
        super().__init__(city_name="madrid", bbox=[-3.7492, 40.3612, -3.5736, 40.5136])

    def fetch_road_network(self):
        """DuckDB query specific to Madrid's standard ingestion."""
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
        logging.info(f"Saved road network to {self.roads_path}")

    def run_pipeline(self):
        """Executes the full pipeline for Madrid."""
        logging.info(f"--- Starting Pipeline for {self.city_name.capitalize()} ---")
        
        # 1. Fetch Roads
        self.fetch_road_network()
        
        # 2. Define dynamic paths based on the base class structure
        detectors_file = self.utd19_dir / f"{self.city_name}_detectors.csv"
        flow_file = self.utd19_dir / f"{self.city_name}_flow.csv"
        output_matrix = self.processed_dir / f"{self.city_name}_universal_twin.parquet"
        
        # 3. Execute Spatial Logic
        roads_gdf, detectors_gdf = load_and_project_data(self.roads_path, detectors_file)
        mapping = snap_to_road(detectors_gdf, roads_gdf, max_distance=50.0)
        build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    ingestor = MadridIngestor()
    ingestor.run_pipeline()
