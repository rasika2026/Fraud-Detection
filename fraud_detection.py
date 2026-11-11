# Credit card fraud detection using SVM (linear kernel)

# Importing libraries
import itertools
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

# Create a Tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Prompt the user to select the file path
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

# Reading the .csv file
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

print(df.describe())  # Descriptive statistics of the features
print(df.columns)

# Filter the fraud data
df_fraud = df[df['Class'] == 1]
plt.figure(figsize=(15, 10))
plt.scatter(df_fraud['Time'], df_fraud['Amount'])
plt.title('Scatter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0, 175000])
plt.ylim([0, 2500])
plt.show()

# Number of frauds where the amount was bigger than 1000
nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0]
print(f'There are only {nb_big_fraud} frauds where the amount was bigger than 1000 '
      f'out of {df_fraud.shape[0]} frauds')

# Count of fraud and non-fraud cases
number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print(f'There are only {number_fraud} frauds and {number_no_fraud} non-frauds in the dataset.')

# Estimated accuracy of a classifier that always predicts "non-fraud"
print("The accuracy of the classifier would be: "
      + str(number_no_fraud / (number_fraud + number_no_fraud)))

# Correlation matrix
df_corr = df.corr()
plt.figure(figsize=(15, 10))
plt.imshow(df_corr, cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=90)
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
plt.title('Heatmap Correlation')
plt.show()

# Rank features based on correlation with the target class
rank = df_corr['Class']
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class', ascending=False)
df_rank.dropna(inplace=True)

# Build the train dataset (first 150000 rows)
df_train_all = df[0:150000]
df_train_1 = df_train_all[df_train_all['Class'] == 1]
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print(f'In this dataset, there are {len(df_train_1)} frauds, '
      f'so we need a similar number of non-frauds')

df_sample = df_train_0.sample(300, random_state=42)
df_train = pd.concat([df_sample, df_train_1], ignore_index=True)

# Shuffle the balanced training set
df_train = df_train.sample(frac=1, random_state=42)

# Build the test dataset (remaining rows)
df_test_all = df[150000:]

# Select the top ten ranked features (index 0 is 'Class' itself, so skip it)
top_features = df_rank.index[1:11]

X_train = np.asarray(df_train[top_features])
y_train = np.asarray(df_train['Class'])

X_test_all = np.asarray(df_test_all[top_features])
y_test_all = np.asarray(df_test_all['Class'])

# Binary label: class=1 (fraud) and class=0 (non-fraud)
class_names = np.array(['0', '1'])


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Use SVM classifier with linear kernel
classifier = svm.SVC(kernel='linear')

# Train the model with the balanced train data
classifier.fit(X_train, y_train)

# Predict using the test data
prediction_svm_all = classifier.predict(X_test_all)

cm = confusion_matrix(y_test_all, prediction_svm_all)
plot_confusion_matrix(cm, class_names)
plt.show()

print('Our criterion gives a result of ' +
      str((cm[0][0] / sum(cm[0]) + 4 * cm[1][1] / (cm[1][0] + cm[1][1])) / 5))
print('We have detected ' + str(cm[1][1]) + ' frauds out of ' +
      str(cm[1][1] + cm[1][0]) + ' total frauds.')
print('The probability to detect a fraud is ' +
      str(cm[1][1] / (cm[1][1] + cm[1][0])))
