# python -W ignore .\script-tsfeaturizer.py
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

    for i, file in enumerate(files):
        if not file.endswith('.csv'):
            continue

        print("Extracting file: " + str(i) + " (" + file + ")")
        df = pd.read_csv(file)
        df['id'] = i
        target = df.TOTAL_QUALITY.unique()[0]
        df = df.drop(axis=1, columns=totals)

        series = [col for col in df if col.endswith(('Time', 'id', 'Flow rate', 'Pressure'))]    
        
        tseries = TimeSeriesFeaturizer()
        extracted_features = tseries.featurize(df[series], n_jobs=jobs)
        
        extracted_features['target'] = target    
            
        if not os.path.isfile(output_file):
            extracted_features.to_csv(output_file)
        else:
            extracted_features.to_csv(output_file, mode='a', header=False)

    print('Tsfresh succesfully finished')


if __name__ == "__main__":
    main()