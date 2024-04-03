# import pandas as pd
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('sign_mnist/sign_mnist.csv')

# train_data, temp_data = train_test_split(df, test_size=0.5, random_state=42)
# test_data, validation_data = train_test_split(temp_data, test_size=0.6, random_state=42)

# train_data.to_csv('sign_mnist/sign_mnist_train/sign_mnist_train.csv', index=False)
# test_data.to_csv('sign_mnist/sign_mnist_test/sign_mnist_test.csv', index=False)
# validation_data.to_csv('sign_mnist/sign_mnist_validation/sign_mnist_validation.csv', index=False)

###############################################################
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import torch

data = pd.read_csv('sign_mnist/sign_mnist.csv')

print("Общая информация")
print(data.info())
print("______________________________")
print("Первые 5 строк таблицы:")
print(data.head())
print("______________________________")
print("Количество классов")
print(len(data['label'].unique()))
print("______________________________")
print("Баланс классов")
print(data['label'].value_counts().sort_values())

X = data.drop('label', axis=1)
y = data['label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

X_train.insert(0, "label", y_train.values)
X_val.insert(0, "label", y_val.values)
X_test.insert(0, "label", y_test.values)

os.makedirs('sign_mnist_train', exist_ok=True)  
X_train.to_csv('sign_mnist_train/sign_mnist_train.csv', index=False)

os.makedirs('sign_mnist_val', exist_ok=True)  
X_val.to_csv('sign_mnist_val/sign_mnist_val.csv', index=False)

os.makedirs('sign_mnist_test', exist_ok=True)  
X_test.to_csv('sign_mnist_test/sign_mnist_test.csv', index=False)

print("Выборки успешно размещены по папкам")

print(torch.cuda.device_count())
print(torch.cuda.is_available())