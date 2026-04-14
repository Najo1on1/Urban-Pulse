import duckdb
from pathlib import Path

def fetch():
    bbox = [-0.510, 51.286, 0.334, 51.691]
    out_path = Path("data/raw/overture_london/london_roads.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(':memory:')
    conn.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    latest = conn.execute("SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'").fetchone()[0]
    
    url = f"s3://overturemaps-us-west-2/release/{latest}/theme=transportation/type=segment/*"
    query = f"""
    COPY (
        SELECT id, geometry, class AS road_class
        FROM read_parquet('{url}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin >= {bbox[0]} AND bbox.xmax <= {bbox[2]}
          AND bbox.ymin >= {bbox[1]} AND bbox.ymax <= {bbox[3]}
    ) TO '{out_path}' (FORMAT PARQUET);
    """
    print(f"Downloading London roads from Overture {latest}...")
    conn.execute(query)
    print("Done!")

if __name__ == "__main__":
    fetch()
