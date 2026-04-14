import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate():
    # London Bounding Box
    min_lon, min_lat = -0.510, 51.286
    max_lon, max_lat = 0.334, 51.691
    
    out_dir = Path("data/raw/london_utd19")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "london_detectors.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['detector_id', 'longitude', 'latitude', 'location_type'])
        for i in range(1, 201):
            # 10% of sensors are "tagged" as being in tunnels or on bridges
            loc = "surface" if i > 20 else "tunnel_or_bridge"
            writer.writerow([f"LDN_{i:04d}", random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat), loc])
            
    with open(out_dir / "london_flow.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'detector_id', 'flow'])
        start = datetime(2026, 5, 1, 0, 0, 0)
        for step in range(96 * 14):
            time_str = (start + timedelta(minutes=15 * step)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, 201):
                writer.writerow([time_str, f"LDN_{i:04d}", random.randint(5, 40)])
    logging.info("Generated synthetic London data with 3D tags.")

if __name__ == "__main__":
    generate()
