import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate():
    # Madrid Bounding Box (Inner M-30/M-40 ring)
    min_lon, min_lat = -3.7492, 40.3323
    max_lon, max_lat = -3.5286, 40.5104
    
    out_dir = Path("data/raw/madrid_utd19")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "madrid_detectors.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['detector_id', 'longitude', 'latitude'])
        for i in range(1, 301): # 300 detectors for Madrid
            writer.writerow([f"MDR_{i:04d}", random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)])
            
    with open(out_dir / "madrid_flow.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'detector_id', 'flow'])
        start = datetime(2026, 5, 1, 0, 0, 0)
        
        # 14 days of data (96 intervals/day * 14)
        for step in range(96 * 14):
            time_str = (start + timedelta(minutes=15 * step)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, 301):
                # Madrid gets slightly higher traffic noise baseline
                writer.writerow([time_str, f"MDR_{i:04d}", random.randint(20, 85)])
                
    logging.info("Generated synthetic Madrid data for 14 days.")

if __name__ == "__main__":
    generate()
