import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate():
    # Lucerne Bounding Box
    min_lon, min_lat = 8.2435, 47.0180
    max_lon, max_lat = 8.3585, 47.0850
    
    out_dir = Path("data/raw/lucerne_utd19")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "lucerne_detectors.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['detector_id', 'longitude', 'latitude', 'location_type'])
        for i in range(1, 201):
            # 10% of sensors are "tagged" as being in tunnels or on bridges
            loc = "surface" if i > 20 else "tunnel_or_bridge"
            writer.writerow([f"LCR_{i:04d}", random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat), loc])
            
    with open(out_dir / "lucerne_flow.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'detector_id', 'flow'])
        start = datetime(2026, 5, 1, 0, 0, 0)
        for step in range(96):
            time_str = (start + timedelta(minutes=15 * step)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, 201):
                writer.writerow([time_str, f"LCR_{i:04d}", random.randint(5, 40)])
    logging.info("Generated synthetic Lucerne data with 3D tags.")

if __name__ == "__main__":
    generate()
