import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate():
    # Rotterdam Bounding Box
    min_lon, min_lat = 4.3875, 51.8700
    max_lon, max_lat = 4.5690, 51.9650
    
    out_dir = Path("data/raw/rotterdam_utd19")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "rotterdam_detectors.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['detector_id', 'longitude', 'latitude'])
        for i in range(1, 351): # 350 detectors for Rotterdam
            writer.writerow([f"RTM_{i:04d}", random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)])
            
    with open(out_dir / "rotterdam_flow.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'detector_id', 'flow'])
        start = datetime(2026, 5, 1, 0, 0, 0)
        
        for step in range(96 * 14):
            time_str = (start + timedelta(minutes=15 * step)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, 351):
                writer.writerow([time_str, f"RTM_{i:04d}", random.randint(15, 75)])
                
    logging.info("Generated synthetic Rotterdam data.")

if __name__ == "__main__":
    generate()
