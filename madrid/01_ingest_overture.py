import duckdb
import logging
import time
from pathlib import Path

# 1. Configure explainable logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_madrid_network():
    # 2. Define the geographic bounds for central Madrid
    madrid_bbox = [-3.7492, 40.3612, -3.5736, 40.5136] 
    
    # Ensure the raw data directory exists
    output_dir = Path("data/raw/overture_spain")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "madrid_roads.parquet"

    logging.info("Spinning up DuckDB and loading cloud/spatial engines...")
    conn = duckdb.connect(':memory:')
    
    # 3. Install and load extensions
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    
    # Set AWS region to avoid cross-region routing latency
    conn.execute("SET s3_region='us-west-2';")

    # 4. FUTURE-PROOFING: Dynamically fetch the latest release version from Overture's STAC catalog
    logging.info("Checking STAC catalog for the latest Overture Maps release...")
    latest_release = conn.execute("SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'").fetchone()[0]
    logging.info(f"Found latest release: {latest_release}")

    overture_url = f"s3://overturemaps-us-west-2/release/{latest_release}/theme=transportation/type=segment/*"

    # 5. Construct the massive parallel query (Updated to xmin/xmax schema)
    query = f"""
    COPY (
        SELECT 
            id,
            geometry,
            class AS road_class
        FROM read_parquet('{overture_url}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin >= {madrid_bbox[0]} AND bbox.xmax <= {madrid_bbox[2]}
          AND bbox.ymin >= {madrid_bbox[1]} AND bbox.ymax <= {madrid_bbox[3]}
    ) TO '{output_path}' (FORMAT PARQUET);
    """

    logging.info(f"Querying Overture S3 buckets for Madrid bounding box: {madrid_bbox}")
    logging.info("This will utilize your 32-thread CPU for parallel fetching. Please wait...")
    
    start_time = time.time()
    
    # Execute the query
    conn.execute(query)
    
    elapsed = time.time() - start_time
    logging.info(f"Success! Saved Madrid road skeleton to {output_path} in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    fetch_madrid_network()