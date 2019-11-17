import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
import pickle      # to save our model as pickle file

# A simple linear prediction model

dt = pd.read_csv('student-mat.csv', sep=";")  # as the sperator used in the csv file is a semi colon we write it here.
dt = dt[["G1", "G2", "G3", "absences", "failures", "studytime"]] # to use the following columns to predict the data

prediction = 'G3'

X = np.array(dt.drop([prediction], 1)) # the attributes we need for their prediction
y = np.array(dt[prediction])    # the thing we need to predict, the whole column is sent

'''
print(X)
print(y)
'''

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # splitting the # data into testing and training

best = 0

'''  # already found the best model so stopping
for i in range(100):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # we need to change the data every time
    linear = linear_model.LinearRegression()  # to call the linear regression model
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("student.pickle", "wb") as f:
            pickle.dump(linear, f)     # to save our model into student.pickle file only when accuracy is better then best


print(best)

'''
pickle_in = open("student.pickle", "rb")  # to open our saved model in read mode
linear = pickle.load(pickle_in)    # save our model into linear model

# print(linear.coef_)  there will be 5 coefficients because it is in 5-D space
# print(linear.intercept_)

predict = linear.predict(x_test)  # returns the predicted value for the set of inputs

# print(predict.shape) --> (40,)

for i in range( predict.shape[0]):
    print(predict[i], x_test[i], y_test[i])

# plotting our data

p = 'G1'
plt.scatter(dt[p], dt["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()




