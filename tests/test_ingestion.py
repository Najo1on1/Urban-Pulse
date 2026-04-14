import pandas as pd
from pathlib import Path

def test_madrid_matrix_structure():
    matrix_path = Path("data/processed/madrid_universal_twin.parquet")
    
    # Ensure the file was actually created
    assert matrix_path.exists(), "Madrid matrix Parquet file does not exist."
    
    # Load the matrix
    df = pd.read_parquet(matrix_path)
    
    # The index should be the timestamp (time-series data)
    assert df.index.name == "timestamp", "Matrix index is not set to timestamp."
    
    # The matrix should contain data (at least 1 timestamp and 1 road segment)
    assert df.shape[0] > 0, "Matrix has no rows (no time intervals)."
    assert df.shape[1] > 0, "Matrix has no columns (no road segments)."
    
def test_zurich_matrix_structure():
    matrix_path = Path("data/processed/zurich_universal_twin.parquet")
    
    assert matrix_path.exists(), "Zurich matrix Parquet file does not exist."
    
    df = pd.read_parquet(matrix_path)
    assert df.index.name == "timestamp", "Matrix index is not set to timestamp."
    assert df.shape[0] > 0, "Matrix has no rows."
    assert df.shape[1] > 0, "Matrix has no columns."
