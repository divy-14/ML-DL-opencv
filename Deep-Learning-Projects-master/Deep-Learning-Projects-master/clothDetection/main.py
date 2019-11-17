import numpy as np
from tensorflow import keras
from keras import datasets as dt
from matplotlib import pyplot as plt


data = dt.fashion_mnist

(x_train, y_train), (x_test, y_test) = data.load_data()
# print(x_train.shape) (60000, 28,28)
# print(y_train.shape)

cloth_names = ["T-shirt/top", "trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Snicker", "Bag",
               "Ankle-Boots"]
x_train = x_train / 255.0  # this data cannot be directly given to the neural network
x_test = x_test / 255.0

'''plt.imshow(x_train[2], cmap=plt.cm.binary)  # binary changes the colour scheme to black and white
print(cloth_names[y_train[2]])
plt.show()'''

'''var = [5, 10]    # To train the model for the first time uncomment this and save it in a new .h5 file.
best =0    

for i in var:
    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train = x_train / 255.0  # this data cannot be directly given to the neural network
    x_test = x_test / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # first layer is the input layer that needs to flattened
        keras.layers.Dense(128, activation="relu"),
        # the hidden layer will have 128 neurons and the activation function will be relu
        keras.layers.Dense(10, activation="softmax")  # calculates the probability density for each output
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=i)  # epochs denote the number of iterations the system will go through in hope to increase the accuracy

    loss, acc = model.evaluate(x_test,
                               y_test)  # model.evaluate returns the loss value and the metrics we mentioned in the compile method

    if acc > best:
        best = acc
        model.save("clothDetection.h5") # to save a model in keras

print(best)'''

model = keras.models.load_model("clothDetection.h5")

predictions = model.predict(x_test)
for i in range(20, 26):
    plt.imshow(x_test[i], cmap= plt.cm.binary)
    plt.xlabel("Predicted: " + cloth_names[np.argmax(predictions[i])])
    plt.title("Actual: " + cloth_names[y_test[i]])
    plt.show()
