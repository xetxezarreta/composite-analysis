# python -W ignore .\script1-tsfresh.py
import numpy as np
import pandas as pd
import os
from glob import glob
import multiprocessing
from ts_featurizer.base import TimeSeriesFeaturizer

def main():
    files = glob('data/*.csv')
    totals = ['Total_PorosityQuantity', 'Total_PorosityQuality', 'Total_UnfilledZones', 'Total_FillingQuality', 'TOTAL_QUALITY']

    jobs = multiprocessing.cpu_count()
    print("CPU count: "+ str(jobs))

    output_file = 'tmp/extracted_data_tsfeaturizer.csv'

    df_list = list()
    target = list()

    for i, file in enumerate(files):
        print("Reading file: " + str(i) + " (" + file + ")")
        df = pd.read_csv(file)
        target.append(df.TOTAL_QUALITY.unique()[0])
        df = df.drop(axis=1, columns=totals)

        series = [col for col in df if col.endswith(('Time', 'id', 'Flow rate', 'Pressure'))]    
        df_list.append(df[series])
    
    tseries = TimeSeriesFeaturizer()
    extracted_features = tseries.featurize(df_list, n_jobs=jobs)
    extracted_features['target'] = target  

    extracted_features.to_csv(output_file)
    print("tsfeaturizer finished")

if __name__ == "__main__":
    main()