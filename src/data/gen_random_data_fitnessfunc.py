import numpy as np
import pandas as pd
import random


random.seed(96)


X_train = np.random.normal(size=[5000, 360])

y_train = np.random.choice([0, 1, 2, 3, 4], size=5000).reshape(-1, 1)

train_data = pd.DataFrame(np.concatenate((X_train, y_train), axis=1))

train_data.to_csv('../../data/fitness_function/train_ff.csv', index=False)


X_test = np.random.normal(size=[1000, 360])

y_test = np.random.choice([0, 1, 2, 3, 4], size=1000).reshape(-1,1)

test_data = pd.DataFrame(np.concatenate((X_test, y_test),axis=1))

test_data.to_csv('../../data/fitness_function/test_ff.csv', index=False)