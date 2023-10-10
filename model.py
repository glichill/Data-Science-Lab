from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def build_and_train_model(X, Y, test_size=0.2, epochs=70, batch_size=32, verbose=1):
    # Fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # Split development set in training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=True, shuffle=True,
                                                        stratify=Y)

    # Oversampling to treat imbalanced classes
    oversample = RandomOverSampler(sampling_strategy='not majority')
    oversample.fit_resample(X_train[:, :, 0], y_train)
    X_train = X_train[oversample.sample_indices_]
    y_train = y_train[oversample.sample_indices_]

    # CNN parameters
    feature_dim_1 = 39
    feature_dim_2 = 94
    channel = 1
    num_classes = 7

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
    X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

    # CNN model
    model = Sequential()

    model.add(Conv2D(24, kernel_size=(3, 3), input_shape=(feature_dim_1, feature_dim_2, channel), kernel_regularizer=l2(0.1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(feature_dim_1, feature_dim_2, channel), kernel_regularizer=l2(0.1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(feature_dim_1, feature_dim_2, channel), kernel_regularizer=l2(0.1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(feature_dim_1, feature_dim_2, channel), kernel_regularizer=l2(0.2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=0.0001)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=verbose)

    return model, history, X_test, y_test
