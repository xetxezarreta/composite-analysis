import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

total = ['Total_PorosityQuantity', 'Total_PorosityQuality', 'Total_UnfilledZones', 'Total_FillingQuality', 'TOTAL_QUALITY']

path = r'D:/master/LIP/composite-analysis/data'
all_files = glob(path + '/*.csv')
li = []

for id, filename in enumerate(all_files[:10]):
    df = pd.read_csv(filename, index_col=None, header=0)
    df = df.drop(axis=1, columns=total)
    df['id'] = id
    li.append(df)    

data = pd.concat(li, axis=0, ignore_index=True)

import tsfresh

# extracted_features = extract_features(data, column_id='id', column_sort='Time')