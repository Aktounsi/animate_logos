from src.preprocessing.sm_label_transformer import encode_classes
import numpy as np
import pandas as pd
import random


random.seed(96)


X_train = np.random.normal(size=[5000, 36])

y_train = np.random.choice([0, 1, 2, 3, 4], size=5000).reshape(-1, 1)
y_train = encode_classes(y_train)

train_data = pd.DataFrame(np.concatenate((X_train, y_train), axis=1))

train_data.to_csv('../../data/fitness_function/train_ff.csv', index=False)


X_test = np.random.normal(size=[1000, 36])

y_test = np.random.choice([0, 1, 2, 3, 4], size=1000).reshape(-1,1)
y_test = encode_classes(y_test)

test_data = pd.DataFrame(np.concatenate((X_test, y_test),axis=1))

test_data.to_csv('../../data/fitness_function/test_ff.csv', index=False)