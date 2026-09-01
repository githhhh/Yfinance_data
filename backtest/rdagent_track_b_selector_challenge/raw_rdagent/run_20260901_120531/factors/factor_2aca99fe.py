import pandas as pd
import numpy as np

def calculate_Volume_Confirmed_RiskAdj_Mom_20():
    # Read the candidate panel data
    panel = pd.read_hdf('candidate_panel.h5', key='data')
    
    # Verify required columns are present
    if 'agent_factor_risk_adj_mom_20' not in panel.columns or 'volume_ratio' not in panel.columns:
        raise ValueError("Required columns are missing in the panel.")
    
    # Calculate the volume-confirmed risk-adjusted momentum factor
    # Formula: VRM_20 = agent_factor_risk_adj_mom_20 * log1p(volume_ratio)
    panel['Volume_Confirmed_RiskAdj_Mom_20'] = panel['agent_factor_risk_adj_mom_20'] * np.log1p(panel['volume_ratio'])
    
    # Create the result DataFrame with MultiIndex (datetime, instrument)
    result = panel[['Volume_Confirmed_RiskAdj_Mom_20']].copy()
    
    # Ensure the index is properly formatted
    result.index.names = ['datetime', 'instrument']
    
    return result

if __name__ == "__main__":
    result_df = calculate_Volume_Confirmed_RiskAdj_Mom_20()
    result_df.to_hdf('result.h5', key='data', mode='w')
    print("Factor calculated and saved to result.h5")
    print(result_df.head())
    print(f"Shape: {result_df.shape}")
    print(f"Dtypes:\n{result_df.dtypes}")
