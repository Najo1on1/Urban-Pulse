import csv
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure explainable logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_synthetic_utd19():
    # Madrid Bounding Box (must match Overture data)
    min_lon, min_lat = -3.7492, 40.3612
    max_lon, max_lat = -3.5736, 40.5136
    
    num_detectors = 500
    
    output_dir = Path("data/raw/madrid_utd19")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detectors_path = output_dir / "madrid_detectors.csv"
    flow_path = output_dir / "madrid_flow.csv"

    logging.info(f"Generating {num_detectors} synthetic loop detectors for Madrid...")
    
    # 1. Generate Detectors (Metadata)
    detectors = []
    with open(detectors_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['detector_id', 'longitude', 'latitude'])
        for i in range(1, num_detectors + 1):
            det_id = f"MAD_{i:04d}"
            lon = random.uniform(min_lon, max_lon)
            lat = random.uniform(min_lat, max_lat)
            detectors.append(det_id)
            writer.writerow([det_id, round(lon, 6), round(lat, 6)])
            
    logging.info(f"Saved detector metadata to {detectors_path}")

    # 2. Generate Traffic Flow (Time Series)
    logging.info("Generating 24 hours of time-series flow data (15-min intervals)...")
    start_time = datetime(2026, 5, 1, 0, 0, 0)
    intervals = 4 * 24 # 96 intervals for 24 hours
    
    with open(flow_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'detector_id', 'flow'])
        
        for step in range(intervals):
            current_time = start_time + timedelta(minutes=15 * step)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Simulate basic urban physics: higher flow during rush hours (8 AM and 5 PM)
            hour = current_time.hour
            is_rush_hour = (7 <= hour <= 9) or (16 <= hour <= 19)
            
            for det_id in detectors:
                base_flow = random.randint(10, 50)
                flow = base_flow * 3 if is_rush_hour else base_flow
                writer.writerow([time_str, det_id, flow])

    logging.info(f"Saved traffic flow data to {flow_path}")
    logging.info("UTD19 synthetic ingestion complete.")

if __name__ == "__main__":
    generate_synthetic_utd19()
