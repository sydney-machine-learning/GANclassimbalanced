import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from statistics import stdev

class test_model():        #parent class
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self):
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
        ])
    
        model.compile(optimizer='adam',               #Configures the model for training.
                    loss='mean_absolute_error',
                    metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=30)

        test_loss, test_accuracy = model.evaluate(self.X_test,  self.y_test, verbose=2)
        train_loss, train_accuracy = model.evaluate(self.X_train,  self.y_train, verbose=2)
        y_preds = model.predict(self.X_test)
        y_preds = np.ravel((y_preds>0.5)*1)
        F1_score = f1_score(self.y_test, y_preds, average='micro')    #For class imbalanced multiclass dataset, use micro as average.

        print('\nTest accuracy:', test_accuracy)

        return test_accuracy, train_accuracy, F1_score   


def test_model_lists(X_train, y_train, X_test, y_test, no_of_trainings):
    test_accuracy_array = []
    train_accuracy_array = []
    f1_score_array = []
    test_model_object = test_model(X_train, y_train.ravel(), X_test, y_test.ravel())

    for i in range(no_of_trainings):
        test_accuracy, train_accuracy, F1_score = test_model_object()
        test_accuracy_array.append(test_accuracy)
        train_accuracy_array.append(train_accuracy)
        f1_score_array.append(F1_score) 

    return test_accuracy_array, train_accuracy_array, f1_score_array



