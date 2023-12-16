import itertools
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


def confusion_matrix(y_test, y_pred):
    if len(y_test.shape) != 2:
        raise IOError('y_test must be a 2D array (Matrix)')
    elif len(y_pred.shape) != 2:
        raise IOError('y_pred must be a 2D array (Matrix)')
    
    # Initialize the confusion matrix with zeros
    cm = np.zeros((y_test.shape[1], y_test.shape[1]))
    
    # Iterate over each observation
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            # For each label, increment the appropriate cell in the confusion matrix
            if y_test[i, j] == 1:
                # True positive and false negatives
                if y_pred[i, j] == 1:
                    cm[j, j] += 1  # True positive
                else:
                    # False negative for each other label that was predicted
                    for k in range(y_pred.shape[1]):
                        if y_pred[i, k] == 1:
                            cm[j, k] += 1
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print("Accuracy on the test-set: " + str(accuracy))

    return cm



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Reds):
    plt.ion()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if np.isnan(cm).any():
            np.nan_to_num(cm, copy=False)

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))

    fmt = '.0%' if normalize else '.0f'
    
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, format(cm[i, j], fmt),
                #  horizontalalignment='center',
                #  color='white' if cm[i, j] > thresh else 'black')
    plt.figure(figsize=(20, 15))

    ax = sns.heatmap(cm, annot=True, fmt=fmt, xticklabels=classes, yticklabels=classes)

    plt.title(title)
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    # plt.ioff()


def draw_cm(y_test, y_pred, classes, normalize=False):
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes, normalize)
    return cm