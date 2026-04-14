import yaml
import argparse
import logging
from pathlib import Path
from src.spatial_utils import load_and_project_data, snap_to_road, build_time_series_matrix

class UniversalEngine:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logging.info(f"Factory Engine loaded for: {self.config['city']}")

    def run(self):
        city = self.config['city'].lower()
        # We simulate the subclass behavior dynamically
        roads_path = Path(f"data/raw/overture_{city}/{city}_roads.parquet")
        detectors_file = Path(self.config['dataset']['utd19_path']) / f"{city}_detectors.csv"
        flow_file = Path(self.config['dataset']['utd19_path']) / f"{city}_flow.csv"
        output_matrix = Path(self.config['output']['path'])

        # 1. Spatial Processing
        roads_gdf, detectors_gdf = load_and_project_data(roads_path, detectors_file)
        
        # 2. Dynamic Strategy Selection
        if "drift_handler" in self.config['processing']['spatial_strategies']:
            from src.strategies.drift_handler import apply_drift_correction
            mapping = apply_drift_correction(detectors_gdf, roads_gdf, self.config['processing']['max_tolerance_meters'])
        else:
            mapping = snap_to_road(detectors_gdf, roads_gdf, self.config['processing']['max_tolerance_meters'])

        # 3. Build Matrix
        build_time_series_matrix(mapping, flow_file, output_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    engine = UniversalEngine(args.config)
    engine.run()
