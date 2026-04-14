import logging
from src.base_ingestor import CityDataIngestor
from src.spatial_utils import load_and_project_data, snap_to_road, build_time_series_matrix

class ZurichIngestor(CityDataIngestor):
    def __init__(self):
        # Zurich bounding box
        super().__init__(city_name="zurich", bbox=[8.4480, 47.3202, 8.6254, 47.4346])

    def fetch_road_network(self):
        """
        OVERRIDE: Zurich has extensive tram and bus networks. 
        We must filter these out at the DuckDB level so cars don't snap to them.
        """
        conn = self._get_duckdb_connection()
        latest_release = self._get_latest_overture_release(conn)
        overture_url = f"s3://overturemaps-us-west-2/release/{latest_release}/theme=transportation/type=segment/*"
        
        # Notice the strict WHERE clause filtering out trams and bus lanes
        query = f"""
        COPY (
            SELECT id, geometry, class AS road_class, subtype
            FROM read_parquet('{overture_url}', filename=true, hive_partitioning=1)
            WHERE bbox.xmin >= {self.bbox[0]} AND bbox.xmax <= {self.bbox[2]}
              AND bbox.ymin >= {self.bbox[1]} AND bbox.ymax <= {self.bbox[3]}
              AND road_class != 'tram'
              AND (subtype IS NULL OR subtype != 'bus_lane')
        ) TO '{self.roads_path}' (FORMAT PARQUET);
        """
        logging.info(f"Fetching Overture roads for {self.city_name.capitalize()} with Multi-Modal Filter...")
        conn.execute(query)
        logging.info(f"Saved filtered road network to {self.roads_path}")

    def run_pipeline(self):
        """Executes the full pipeline for Zurich."""
        logging.info(f"--- Starting Pipeline for {self.city_name.capitalize()} ---")
        
        self.fetch_road_network()
        
        detectors_file = self.utd19_dir / f"{self.city_name}_detectors.csv"
        flow_file = self.utd19_dir / f"{self.city_name}_flow.csv"
        output_matrix = self.processed_dir / f"{self.city_name}_universal_twin.parquet"
        
        roads_gdf, detectors_gdf = load_and_project_data(self.roads_path, detectors_file)
        mapping = snap_to_road(detectors_gdf, roads_gdf, max_distance=50.0)
        build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    ingestor = ZurichIngestor()
    ingestor.run_pipeline()
