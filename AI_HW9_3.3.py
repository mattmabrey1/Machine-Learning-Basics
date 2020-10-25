#   Matthew Mabrey
#   Artificial Intelligence CSC-380
#   Dr. Yoon
#   Homework 9 Problem 3.3
#   Program Description: This program uses Gaussian Mixture Method using the 'tied' covariance type to try to
#   separate the input data in X_train into 6 groups and plots the data in a 2D graph with the ground truth, as well
#   as plots when using 2 components determined by Principle Component Analysis. It also prints the Adjusted Rand
#   Index (ARI) performance and the Normalized Mutual Information (NMI) performance.

#   We use sklearn packages as the foundation of this program to fit and score our values as well as
#   using a 2-D scatter plot courtesy of scikit-learn at
#   https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_digits_simple_classif.html

from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf)

X_train_file = open("X_train.txt","r")
X_final_test_file = open("X_test.txt","r")
y_train_file_file = open("y_train.txt","r")
y_final_test_file = open("y_test.txt","r")
activity_labels_file = open("activity_labels.txt","r")

X_train = np.loadtxt(X_train_file)
X_final_test = np.loadtxt(X_final_test_file)
y_train = np.loadtxt(y_train_file_file)
y_final_test = np.loadtxt(y_final_test_file)
activity_labels = np.loadtxt(activity_labels_file, dtype=str)

X_train_file.close()
X_final_test_file.close()
y_train_file_file.close()
y_final_test_file.close()
activity_labels_file.close()


gauss_mix = mixture.GaussianMixture(n_components=6, covariance_type="tied")

gm = gauss_mix.fit(X_train)

labels = gm.predict(X_final_test)


print("\n The Adjusted Rand Index (ARI) performance for a Gaussian Mixture Model with 6 components and \
covariance type = 'tied' : ", metrics.adjusted_rand_score(y_final_test, labels), " \n")
print("\n The Normalized Mutual Information (NMI) performance for a Gaussian Mixture Model with 6 components and \
covariance type = 'tied' : ", metrics.normalized_mutual_info_score(y_final_test, labels), " \n")
print("Real activity label for the data\n", y_final_test)
print("Predicted Gaussian Mixture label\n", labels)



formatter = plt.FuncFormatter(lambda i, *args: activity_labels[int(i)])

fig1 = plt.figure(figsize=(8, 6))


plt.title("Real activity label for data")
plt.scatter(X_final_test[:, 00], X_final_test[:, 40], c=y_final_test, cmap="Paired", edgecolor='k')
plt.xlabel('tBodyAcc-XYZ')
plt.ylabel('tGravityAcc-XYZ')
plt.colorbar(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], format=formatter)

fig2 = plt.figure(figsize=(8, 6))
plt.title("Predicted Gaussian Mixture label")
plt.xlabel('tBodyAcc-XYZ')
plt.ylabel('tGravityAcc-XYZ')
plt.scatter(X_final_test[:, 0], X_final_test[:, 40], c=labels, edgecolor='k')


pca = PCA(n_components=2)
proj = pca.fit_transform(X_final_test)

fig3 = plt.figure(figsize=(8, 6))
plt.title("Actual activity label for data using PCA")
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.scatter(proj[:, 0], proj[:, 1], c=y_final_test, cmap="Paired", edgecolor='k')
plt.colorbar(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], format=formatter)


fig3 = plt.figure(figsize=(8, 6))
plt.title("Predicted Gaussian Mixture label using PCA")
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.scatter(proj[:, 0], proj[:, 1], c=labels, edgecolor='k')

plt.show()

