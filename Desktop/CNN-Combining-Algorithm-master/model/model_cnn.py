import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import load_model
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def get_model(shape_1, num_classes):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(1, 1), activation='relu', input_shape=(shape_1, num_classes, 1)))
    model.add(Conv2D(256, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(384, kernel_size=(1, 1), activation='relu'))
    # model.add(Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(384, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(256, kernel_size=(1, 1), activation='relu'))
    # model.add(Conv2D(512, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    # model.add(Conv2D(512, kernel_size=(1, 1), activation='relu'))
    # model.add(Conv2D(512, kernel_size=(1, 1), activation='relu'))
    # model.add(Conv2D(512, kernel_size=(1, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1)))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    # model.add(Dense(4096, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dense(4096, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dense(1000, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0005),
                  metrics=['accuracy'])
    # print(model.summary())
    return model


def calc_mf1(model, X, y):
    pred_y = model.predict_classes(X)
    pred_y_array = [i + 1 for i in pred_y]
    mf1 = metrics.f1_score(y, pred_y_array, average="macro")
    return mf1
