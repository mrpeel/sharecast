import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def score_accuracy(class_labels, class_actuals, bin_names, generate_images=True):
    print(accuracy_score(class_actuals, class_labels))
    print('Scoring validation')

    score = accuracy_score(class_actuals, class_labels)
    print('Accuracy:', score)

    f1 = f1_score(
        class_actuals, class_labels, average='weighted')
    print('F1 score:', f1)

    cm = confusion_matrix(class_actuals, class_labels)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    if generate_images:
        plt.figure()
        plot_confusion_matrix(cm, bin_names, title='Confusion matrix')

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    if generate_images:
        plt.figure()
        plot_confusion_matrix(cm_normalized, bin_names,
                              title='Normalized confusion matrix')

    return score, f1
