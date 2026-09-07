import pandas as pd

def calculate_vol_confirmed_mom_20():
    df = pd.read_hdf('daily_pv.h5', key='data')
    df['vol_confirmed_mom_20'] = df['mom_20'] * df['vol_ratio_5_20']
    factor_df = df[['vol_confirmed_mom_20']]
    factor_df.to_hdf('result.h5', key='data', mode='w')

if __name__ == '__main__':
    calculate_vol_confirmed_mom_20()
