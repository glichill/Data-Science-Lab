import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test, verbose=2):
    """
    Evaluate the model on the test data.
    """
    return model.evaluate(X_test, y_test, verbose=verbose)

def plot_history(history):
    """
    Plot training history of the model - Accuracy and Loss
    """
    # Plot accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def predict_classes(model, X_test, classes):
    """
    Make predictions over the testing set and convert integer predictions to class labels.
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_class = [classes[i] for i in y_pred]
    return y_pred_class


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots the confusion matrix.
    """
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 8.5))
    plt.imshow(matrix)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i, true_label in enumerate(matrix):
        for j, predicted_label in enumerate(true_label):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()