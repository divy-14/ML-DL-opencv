from tensorflow import keras
from keras import datasets as dt
from matplotlib import pyplot as plt
import numpy as np
import cv2

data = dt.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_test.shape)
list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
var = [5, 10]
best = 0

'''  # train the model with the whole dataset for more accuracy
for i in var:
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=i)
    loss, acc = model.evaluate(x_test, y_test)

    if acc > best:
        best = acc
        model.save("mnist.h5")

'''
model = keras.models.load_model("mnist.h5")


def prepare(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img, (28,28))
    return new_img.reshape(-1, 28, 28, 1)


prediction = model.predict([prepare('7_1.jpg')])
print(prediction)


'''for i in range(40, 46):
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title("Predicted: " + str(list[np.argmax(prediction[i])]))
    plt.xlabel("Actual: " + str(list[y_test[i]]))
    plt.show()'''
