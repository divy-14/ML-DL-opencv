import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import  pickle
import matplotlib.pyplot as plt

dt = pd.read_csv("heart.csv")
prediction = "target"

X = np.array(dt.drop([prediction], axis=1))
y = np.array(dt[prediction])

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1)

best = 0

'''
for i in range(3,11, 2):
    print(i)
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    #print(acc)
    if acc > best:
        best = acc
        with open("prediction1.pickle" , "wb") as f:
            pickle.dump(model, f)

print(best) # 81.6 the best accuracy with knn

pickle_in = open("prediction1.pickle", "rb")
model = pickle.load(pickle_in)

list = ["no presence", "present"]
predict = model.predict(x_test)
for i in range(predict.shape[0]):
    print("Predicted: "+ list[predict[i]], "Actual: "+ list[y_test[i]])

'''
float_val = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

'''for i in float_val:
    model1 = svm.SVC(kernel="linear", C=i)
    model1.fit(x_train, y_train)
    acc= model1.score(x_test, y_test)
    print(acc)

    if acc>best:
        best = acc
        with open("prediction2.pickle", "wb") as f:
            pickle.dump(model1, f)

print(best)
'''
pickle_in = open("prediction2.pickle", "rb")
model1 = pickle.load(pickle_in)
y_pred = model1.predict(x_test)

list = ["no presence", "present"]
predict = model1.predict(x_test)
for i in range(predict.shape[0]):
    print("Predicted: "+ list[predict[i]], "Actual: "+ list[y_test[i]])


