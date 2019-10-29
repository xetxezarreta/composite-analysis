# python -W ignore .\script1-tsfresh.py
import pandas as pd
from glob import glob
import multiprocessing
from tsfresh import extract_features

def main():
    files = glob('data/*.csv')
    totals = ['Total_PorosityQuantity', 'Total_PorosityQuality', 'Total_UnfilledZones', 'Total_FillingQuality', 'TOTAL_QUALITY']

    df_list = list()
    target = list()

    jobs = multiprocessing.cpu_count()
    print("CPU count: "+ str(jobs))

    for i, file in enumerate(files):
        print("Extracting features from: " + file)
        df = pd.read_csv(file)
        df['id'] = i
        target.append(df.TOTAL_QUALITY.unique()[0])
        df = df.drop(axis=1, columns=totals)

        consta = [col for col in df if col.endswith(('K1', 'K2', 'K3'))]
        series = [col for col in df if col.endswith(('Time', 'id', 'Flow rate', 'Pressure'))]

        extracted_features = extract_features(df[series], disable_progressbar=True, column_id='id', column_sort='Time', n_jobs=jobs)

        for j in consta:
            extracted_features[j] = df[j].unique()[0] 
        
        df_list.append(extracted_features)        

    df = pd.concat(df_list)
    df['target'] = target

    df.to_csv('tmp/extracted_data.csv')
    print('Tsfresh succesfully finished')

if __name__ == "__main__":
    main()