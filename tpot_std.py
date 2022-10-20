import tpot
from tpot import TPOTClassifier
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys



data_dir = sys.argv[1]
target_col = sys.argv[2]


loaded_data = pd.read_csv(data_dir)


for col in loaded_data.columns:  # Iterate over chosen columns
    loaded_data[col] = pd.to_numeric(loaded_data[col], errors='coerce')

#drop nulls
loaded_data.dropna(inplace=True)

X = loaded_data.drop(target_col, axis=1).values
y = loaded_data[target_col].values


X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y)


tpot_class = TPOTClassifier(generations=10, scoring='accuracy', n_jobs=4,
                         random_state=1,
                         verbosity=2)
tpot_class.fit(X_train, y_train)
tpot_class.export('tpot_pipeline.py')

score = tpot_class.score(X_test, y_test)

print(score)



