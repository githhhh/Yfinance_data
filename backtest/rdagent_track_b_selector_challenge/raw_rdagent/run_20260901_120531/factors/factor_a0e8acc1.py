import pandas as pd

def calculate_base_tightness_score():
    # Read the source data from candidate_panel.h5
    df = pd.read_hdf('candidate_panel.h5', key='data')
    
    # Extract the required variables
    base_depth_pct = df['base_depth_pct']
    base_duration_weeks = df['base_duration_weeks']
    
    # Calculate the Base Tightness Score
    # BT = (1 / (1 + base_depth_pct)) * (1 / (1 + base_duration_weeks / 10))
    factor_values = (1.0 / (1.0 + base_depth_pct)) * (1.0 / (1.0 + base_duration_weeks / 10.0))
    
    # Create result dataframe with MultiIndex (datetime, instrument)
    result = pd.DataFrame({'Base_Tightness_Score': factor_values}, index=df.index)
    
    # Save to result.h5
    result.to_hdf('result.h5', key='data')
    
    return result

if __name__ == '__main__':
    calculate_base_tightness_score()
