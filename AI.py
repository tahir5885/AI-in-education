# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Reading the dataset
data = pd.read_csv('student_data.csv')

# Splitting the dataset into training and testing sets
X = data.drop('Grade', axis=1)
y = data['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predicting grades for the test set
y_pred = clf.predict(X_test)

# Evaluating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)