from pathlib import Path
import numpy as np
import pandas as pd

#Retrieves the csvs from the data/raw folder 
def get_csvs():
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir.parent.parent / "data" / "raw" 
    
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    csvs = data_path.glob("01org_opa-ff.csv")

    dataframes = {}

    #Loads and sorts data sets in ascending order ready to be tokenized
    for csv_file in csvs:
        df_name = csv_file.stem
        print(f"loading: {df_name}")

        dataframes[df_name] = (
            pd.read_csv(csv_file, parse_dates=["created_at"])
            .sort_values(by=["created_at"], ascending = True)
            .reset_index(drop = True)
        )

    return dataframes


