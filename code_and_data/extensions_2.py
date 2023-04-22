"""
Extension 2

I have Build networks for other data sets. The Dataset set i have used is 
wine quality dataset. Name of the csv file is 'winequality-red.csv'. It is
kept along with the data and code folder. file to donwload it is also mentioned in 
the read me file.

I have tried to compare the different model accuracy with different epoches

Submitted by : Gmon Kuzhiyanikkal
NU ID: 002724506
Date: 14/03/2023
"""
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv('winequality-red.csv', sep=';')

# Convert the target variable to binary (1 for quality >= 7, 0 for quality < 7)
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Split the dataset into independent and dependent variables
X = df.drop(['quality'], axis=1)
y = df['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

print("\nmodels with 100 iterations\n")

# Model 1
model1 = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=100)
model1.fit(X_train, y_train)
accuracy = model1.score(X_test, y_test)
print('Model 1 Test accuracy:', accuracy)

# Model 2
model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), activation='relu', max_iter=100)
model2.fit(X_train, y_train)
accuracy = model2.score(X_test, y_test)
print('Model 2 Test accuracy:', accuracy)

print("\nmodels with 50 iterations\n")

# Model 1
model1 = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=50)
model1.fit(X_train, y_train)
accuracy = model1.score(X_test, y_test)
print('Model 1 Test accuracy:', accuracy)

# Model 2
model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), activation='relu', max_iter=50)
model2.fit(X_train, y_train)
accuracy = model2.score(X_test, y_test)
print('Model 2 Test accuracy:', accuracy)

print("\nmodels with 10 iterations\n")

# Model 1
model1 = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=10)
model1.fit(X_train, y_train)
accuracy = model1.score(X_test, y_test)
print('Model 1 Test accuracy:', accuracy)

# Model 2
model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8), activation='relu', max_iter=10)
model2.fit(X_train, y_train)
accuracy = model2.score(X_test, y_test)
print('Model 2 Test accuracy:', accuracy)
