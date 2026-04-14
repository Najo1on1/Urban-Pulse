import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from src.spatial_utils import snap_to_road

def test_distance_calculation_and_snapping():
    # 1. Create a fake metric coordinate system
    metric_crs = "EPSG:32630"
    
    # 2. Draw a straight vertical road at X=0, from Y=0 to Y=100
    roads_df = pd.DataFrame({'id': ['ROAD_1']})
    roads_gdf = gpd.GeoDataFrame(
        roads_df, 
        geometry=[LineString([(0, 0), (0, 100)])], 
        crs=metric_crs
    )
    
    # 3. Place a detector exactly 10 meters to the right (X=10, Y=50)
    detectors_df = pd.DataFrame({'detector_id': ['DET_01']})
    detectors_gdf = gpd.GeoDataFrame(
        detectors_df, 
        geometry=[Point(10, 50)], 
        crs=metric_crs
    )
    
    # 4. Run your snapping algorithm with a tight 15m threshold
    snapped_df = snap_to_road(detectors_gdf, roads_gdf, max_distance=15.0)
    
    # 5. Assertions (The Test)
    assert len(snapped_df) == 1, "The detector should have snapped successfully."
    assert snapped_df.iloc[0]['overture_segment_id'] == 'ROAD_1', "It mapped to the wrong road."
    
def test_max_distance_rejection():
    metric_crs = "EPSG:32630"
    roads_gdf = gpd.GeoDataFrame({'id': ['ROAD_1']}, geometry=[LineString([(0, 0), (0, 100)])], crs=metric_crs)
    
    # Place a detector 60 meters away
    detectors_gdf = gpd.GeoDataFrame({'detector_id': ['DET_02']}, geometry=[Point(60, 50)], crs=metric_crs)
    
    # Run algorithm with a 50m threshold
    snapped_df = snap_to_road(detectors_gdf, roads_gdf, max_distance=50.0)
    
    # The detector should be discarded
    assert len(snapped_df) == 0, "The detector was 60m away but was not discarded by the 50m threshold!"
