from sklearn import svm
import sklearn
import numpy as np
from sklearn import datasets
import pickle

# a supervised machine learning program to detect cancer as benign or malignant using SVM

cancer = datasets.load_breast_cancer()
'''print(cancer.feature_names)
print(cancer.target_names)'''

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
best = 0

''' # trained the model for best accuracy so commented out.
values = ["0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]
for i in values:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    print(i)
    model = svm.SVC(kernel='linear', C = float(i))  # using the Support Vector Machine # C should not be less than or equal to zero
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("cancerPred.pickle", "wb") as f:
            pickle.dump(model, f)

print(best)
'''
pickle_in = open("cancerPred.pickle", "rb")
model = pickle.load(pickle_in)
predictions = model.predict(x_test)

name = ["benign", "malignant"]

for i in range(predictions.shape[0]):
    print("Predicted: ", name[predictions[i]], "Actual: ", name[y_test[i]])
