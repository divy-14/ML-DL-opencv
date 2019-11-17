from sklearn import linear_model
import sklearn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, \
    preprocessing  # preprocessing is the library used to generate labels for character strings
import pickle

# classification problem

dt = pd.read_csv("car.data")

l = preprocessing.LabelEncoder()

# converting all the string values into their respective numerical values via a list
# the label encoder sorts the list
# in form of alphabetical order where each word is unique and then assign the corresponding values to them.

buying = l.fit_transform(list(dt["buying"]))  # fit_transform returns an array of encoded values
mai = l.fit_transform(list(dt["maint"]))
door = l.fit_transform(list(dt["door"]))
persons = l.fit_transform(list(dt["persons"]))
lug_boot = l.fit_transform(list(dt["lug_boot"]))
safety = l.fit_transform(list(dt["safety"]))
cls = l.fit_transform(list(dt["class"]))

predict = "class"

X = list(zip(buying, mai, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
best = 0
x = 3

'''  # saved our model, if running for first time uncomment and then run 
for i in range(30):

    model = KNeighborsClassifier(n_neighbors=x)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    if acc > best:
        x += 2
        best = acc
        with open("carModel.pickle", "wb") as f:
            pickle.dump(model, f)

print(best)
'''

pickle_in = open("carModel.pickle", "rb")  # to open our saved model in read mode
model = pickle.load(pickle_in)    # save our model into linear model

prediction = model.predict(x_test)
# print(prediction.shape) numpy array

class_names = ["acc", "good", "unacc", "vgood"]
for i in range(prediction.shape[0]):
    print(class_names[prediction[i]], x_test[i], class_names[y_test[i]])
