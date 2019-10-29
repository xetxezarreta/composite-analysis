# python -W ignore .\script1-tpot.py
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import multiprocessing

def main():
    df = pd.read_csv('tmp/extracted_data_processed.csv')
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    jobs = multiprocessing.cpu_count()
    print("CPU count: "+ str(jobs))

    tpot = TPOTClassifier(generations=50, population_size=50, n_jobs=jobs, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

    tpot.export('tmp/tpot_composite_pipeline.py')

if __name__ == "__main__":
    main()