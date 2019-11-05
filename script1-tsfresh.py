# Run 'nohup python bgservice.py &' to get the script to ignore the hangup signal and keep running. Output will be put in nohup.out.
# python -W ignore .\script1-tsfresh.py
# nohup python -W ignore .\script1-tsfresh.py &
import os
import pandas as pd
from glob import glob
import multiprocessing
from tsfresh import extract_features

def main():
    files = glob('data/*.csv')
    totals = ['Total_PorosityQuantity', 'Total_PorosityQuality', 'Total_UnfilledZones', 'Total_FillingQuality', 'TOTAL_QUALITY']

    jobs = multiprocessing.cpu_count()
    print("CPU count: "+ str(jobs))

    output_file = 'tmp/extracted_data_test.csv'

    for i, file in enumerate(files[:5]):
        print("Extracting file: " + str(i) + " (" + file + ")")
        df = pd.read_csv(file)
        df['id'] = i
        target = df.TOTAL_QUALITY.unique()[0]
        df = df.drop(axis=1, columns=totals)

        consta = [col for col in df if col.endswith(('K1', 'K2', 'K3'))]
        series = [col for col in df if col.endswith(('Time', 'id', 'Flow rate', 'Pressure'))]

        extracted_features = extract_features(df[series], disable_progressbar=True, column_id='id', column_sort='Time', n_jobs=jobs)

        for j in consta:
            extracted_features[j] = df[j].unique()[0]    
        
        extracted_features['target'] = target    
        
        if not os.path.isfile(output_file):
            extracted_features.to_csv(output_file)
        else:
            extracted_features.to_csv(output_file, mode='a', header=False)

    print('Tsfresh succesfully finished')

if __name__ == "__main__":
    main()