import pandas as pd
import numpy as np


def calculate_rel_strength_acceleration_20_60():
    # Read the candidate panel data
    panel = pd.read_hdf('candidate_panel.h5', key='data')
    
    # Calculate the relative strength acceleration factor
    # Formula: (rel_spy_20 - rel_spy_60) / 2
    panel['Rel_Strength_Acceleration_20_60'] = (panel['rel_spy_20'] - panel['rel_spy_60']) / 2
    
    # Create the result DataFrame with MultiIndex (datetime, instrument)
    result = panel[['Rel_Strength_Acceleration_20_60']].copy()
    
    # Ensure the index is properly formatted
    result.index.names = ['datetime', 'instrument']
    
    return result


if __name__ == "__main__":
    result_df = calculate_rel_strength_acceleration_20_60()
    result_df.to_hdf('result.h5', key='data', mode='w')
    print("Factor calculated and saved to result.h5")
    print(result_df.head())
    print(f"Shape: {result_df.shape}")
    print(f"Dtypes:\n{result_df.dtypes}")
