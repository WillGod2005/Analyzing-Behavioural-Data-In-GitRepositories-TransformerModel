from pathlib import Path
import pandas as pd

def get_csv(filename):
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir.parent.parent / "data" / "raw" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    return data_path


try:
    csv_path = get_csv("01org_opa-ff.csv")
    df = pd.read_csv(csv_path)
    sorted_df = df.sort_values(by=["created_at"], ascending = False)
    print(f"{sorted_df}")
except FileNotFoundError as e:
    print(e)