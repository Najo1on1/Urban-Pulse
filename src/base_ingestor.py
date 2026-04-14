import duckdb
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CityDataIngestor(ABC):
    """
    Abstract Base Class for urban data ingestion.
    Handles common infrastructure (DuckDB, cloud connections) while 
    forcing subclasses to implement city-specific spatial quirks.
    """
    def __init__(self, city_name: str, bbox: list[float]):
        self.city_name = city_name.lower()
        self.bbox = bbox
        
        # Standardized paths
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        
        self.overture_dir = self.raw_dir / f"overture_{self.city_name}"
        self.utd19_dir = self.raw_dir / f"{self.city_name}_utd19"
        
        self.overture_dir.mkdir(parents=True, exist_ok=True)
        self.utd19_dir.mkdir(parents=True, exist_ok=True)
        
        self.roads_path = self.overture_dir / f"{self.city_name}_roads.parquet"

    def _get_duckdb_connection(self):
        """Spins up a memory-mapped DuckDB instance with spatial/cloud extensions."""
        logging.info("Initializing DuckDB spatial/cloud engines...")
        conn = duckdb.connect(':memory:')
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("SET s3_region='us-west-2';")
        return conn

    def _get_latest_overture_release(self, conn) -> str:
        """Dynamically fetches the latest STAC catalog release."""
        release = conn.execute("SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'").fetchone()[0]
        return release

    @abstractmethod
    def fetch_road_network(self):
        """
        Must be implemented by the subclass. 
        Different cities have different filtering needs (e.g., Zurich excludes bus lanes).
        """
        pass
        
    @abstractmethod
    def run_pipeline(self):
        """Executes the full end-to-end ingestion and spatial snapping pipeline."""
        pass
