import pandas as pd

def calculate_prox_52w_high():
    df = pd.read_hdf('daily_pv.h5', key='data')
    prox_52w_high = -df['dist_to_52w_high_pct']
    factor_df = prox_52w_high.to_frame(name='prox_52w_high')
    factor_df.to_hdf('result.h5', key='data', mode='w')

if __name__ == '__main__':
    calculate_prox_52w_high()
