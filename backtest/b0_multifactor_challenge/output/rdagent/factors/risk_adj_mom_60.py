import pandas as pd
import numpy as np

def calculate_risk_adj_mom_60():
    df = pd.read_hdf('daily_pv.h5', key='data')
    mom_60 = df['mom_60']
    rv_20 = df['rv_20']
    factor_value = mom_60 / (rv_20 + 1e-5)
    factor_value = factor_value.replace([np.inf, -np.inf], np.nan)
    result = pd.DataFrame({'risk_adj_mom_60': factor_value}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w')

if __name__ == '__main__':
    calculate_risk_adj_mom_60()
