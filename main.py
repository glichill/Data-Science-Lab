import pandas as pd
import numpy as np
import audio_visualization as av
import data_preprocessing as dp
import model as mdl
import evaluation as evl


# Configuration settings
BASE_PATH='./project/'
AUDIO_DATASET_PATH= './project/dsl_data/audio/speakers/'

# Get a random audio file from the directory
random_audio_file = av.get_random_audio_file(AUDIO_DATASET_PATH)

# Analyze the randomly chosen audio file
av.analyze_audio(random_audio_file)

# Load DataSet
metadata=pd.read_csv(BASE_PATH+'dsl_data/development.csv')
metadata.head()
metadata['class']=metadata['action'].astype(str) +""+ metadata["object"]

# Explore Dataset
dp.explore_dataset(metadata, BASE_PATH)

# Preprocess Data obtaining our X and Y
X, Y = dp.preprocess_data(metadata, BASE_PATH)

# Build and Train the Model
model, history,X_test, y_test = mdl.build_and_train_model(X, Y, test_size=0.2, epochs=70, batch_size=32, verbose=1)


# Evaluate the Model
loss, accuracy = evl.evaluate_model(model, X_test, y_test, verbose=2)
print(f'Model Loss: {loss}, Accuracy: {accuracy}')

# Plot Training History
evl.plot_history(history)


# Class Predictions
classes = ['activatemusic','change languagenone','deactivatelights','decreaseheat','decreasevolume','increaseheat','increasevolume']
y_pred_class = evl.predict_classes(model, X_test, classes)

# Get True Class Labels
y_true = np.argmax(y_test, axis=1)
y_test_class = [classes[i] for i in y_true]

# Plot Confusion Matrix
evl.plot_confusion_matrix(y_test_class, y_pred_class, classes)