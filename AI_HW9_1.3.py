#   Matthew Mabrey
#   Artificial Intelligence CSC-380
#   Dr. Yoon
#   Homework 9 Problem 1.3
#   Program Description: This program attempts to classify human activity recognition data set. To find the values
#   for kernel and C we use a train-test split with all different types of hyperparameters (C=1,..,100) and
#   (linear, poly, rbf, sigmoid).
#   We then predict the data using the best hyperparameters found and plot it using Principle Component Analysis.

#   We use sklearn packages as the foundation of this program to fit, score, and predict our values as well as
#   using a 2-D scatter plot of data using Principal Component Analysis to project the data to a lower dimension
#   the dimensional space courtesy of scipy-lectures at
#   https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_digits_simple_classif.html

from sklearn import svm
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf)

X_train_file = open("X_train.txt","r")
X_final_test_file = open("X_test.txt","r")
y_train_file_file = open("y_train.txt","r")
y_final_test_file = open("y_test.txt","r")
activity_labels_file = open("activity_labels.txt","r")

X_train = np.loadtxt(X_train_file)
X_test = np.loadtxt(X_final_test_file)
y_train = np.loadtxt(y_train_file_file)
y_test = np.loadtxt(y_final_test_file)
activity_labels = np.loadtxt(activity_labels_file, dtype=str)

X_train_file.close()
X_final_test_file.close()
y_train_file_file.close()
y_final_test_file.close()
activity_labels_file.close()

kf = KFold(n_splits=5)
kf.get_n_splits(X_train)
print(kf, "\n")

max_score = 0               # save the max 5-fold CV score
best_C_value = 0            # save the C value used to get the max 5-fold CV score
best_kernel = ""            # save the kernel used to get the max 5-fold CV score

kernels = ["linear", "poly" , "rbf", "sigmoid" ] #

for kernel in kernels:

    print("\nUsing kernel=", kernel, "\n")

    for c in range(1, 101):
        score = 0

        clf = svm.SVC(kernel=kernel, C=c).fit(X_train, y_train)

        score = clf.score(X_test, y_test)

        print("Kernel: ", kernel, " Score with C=", c, ": ", score)
        if score > max_score:
            max_score = score
            best_C_value = c
            best_kernel = kernel



print("Using a training and test set to determine the best parameters for the SVM \n")

clf = svm.SVC(kernel=best_kernel, C=best_C_value).fit(X_train, y_train)
final_test_score = clf.score(X_test, y_test)

print("The best score comes from using Kernel=", best_kernel, " and C=", best_C_value, ": ", final_test_score, "\n")

print("Predicted activity for the test data \n")
print(clf.predict(X_test), "\n")
print("Actual activity for the test data \n")
print(y_test)


formatter = plt.FuncFormatter(lambda i, *args: activity_labels[int(i)])

plt.figure(figsize=(8, 6))
pca = PCA(n_components=2)

proj = pca.fit_transform(X_test)
plt.title("Real activity for data")
plt.scatter(proj[:, 0], proj[:, 1], c=y_test, cmap="Paired")
plt.colorbar(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], format=formatter)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')

plt.figure(figsize=(8, 6))
plt.title("Predicted activity for data")
plt.scatter(proj[:, 0], proj[:, 1], c=clf.predict(X_test), cmap="Paired")
plt.colorbar(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], format=formatter)
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.show()


