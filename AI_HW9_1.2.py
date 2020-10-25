#   Matthew Mabrey
#   Artificial Intelligence CSC-380
#   Dr. Yoon
#   Homework 9 Problem 1.2
#   Program Description: This program attempts to classify Digits data set. To find the values for kernel and C we use
#   five-fold cross validation to try out different values (linear, poly, rbf, sigmoid) and (C=1,..,100) and use
#   the values which give the best score. We then use those values to fit the test set (60%) to the final validation
#   set (40%) and output the score as well as the predicted species (class) for each of the data points as well
#   as their actual species (class)

#   We use sklearn packages as the foundation of this program to fit, score, and predict our values as well as
#   using a 2-D scatter plot of Digit data using Principal Component Analysis to project the data to a lower dimension
#   the dimensional space courtesy of scipy-lectures at
#   https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_digits_simple_classif.html

from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


X, y = load_digits(return_X_y = True)

X_train, X_final_test, y_train, y_final_test = train_test_split(X, y, test_size=0.40, random_state=0)

kf = KFold(n_splits=5)
kf.get_n_splits(X_train)
print(kf, "\n")

max_score = 0               # save the max 5-fold CV score
best_C_value = 0            # save the C value used to get the max 5-fold CV score
best_kernel = ""            # save the kernel used to get the max 5-fold CV score

kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:

    print("\nUsing kernel=", kernel, "\n")

    for c in range(1, 101):
        score = 0

        for train_index, test_index in kf.split(X_train):
            # Uncomment to print out the indices of the training and test sets for this fold
            # print("Train Indices: ", train_index )
            # print("Test Indices: ", test_index, "\n")

            X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]

            # Uncomment to print out the data values of the training and test sets for this fold
            # print("X-fold Train: ", X_fold_train)
            # print("X-fold Test:", X_fold_test, "\n")

            y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

            # Uncomment to print out the class values of the training and test sets for this fold
            # print("y-fold Train: ", y_fold_train)
            # print("y-fold Test:", y_fold_test,"\n")

            clf = svm.SVC(kernel=kernel, C=c).fit(X_fold_train, y_fold_train)

            score += clf.score(X_fold_test, y_fold_test)

        score = score / 5
        print("5-fold CV Score with C=", c, ": ", score)
        if score > max_score:
            max_score = score
            best_C_value = c
            best_kernel = kernel


print("\nBest 5-fold cross validation score (average of the 5 scores), with kernel=", best_kernel,"and C=",
      best_C_value,", is:", max_score)
print("Using 60% as training set for 5-fold CV and saving 40% for the validation set \n")

clf = svm.SVC(kernel=best_kernel, C=best_C_value).fit(X_train, y_train)
final_test_score = clf.score(X_final_test, y_final_test)

print("Final test score based on using Kernel=", best_kernel, " and C=", best_C_value, ": ", final_test_score, "\n")

print("Predicted digit for the final test data \n")
print(clf.predict(X_final_test), "\n")
print("Actual digit for the final test data \n")
print(y_final_test)



from sklearn.decomposition import PCA

plt.figure()
pca = PCA(n_components=2)
proj = pca.fit_transform(X_final_test)
plt.title("Real digit for hand drawn digit picture data")
plt.scatter(proj[:, 0], proj[:, 1], c=y_final_test, cmap="Paired")
plt.colorbar()
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')

plt.figure()
plt.title("Predicted digit for hand drawn digit picture data")
plt.scatter(proj[:, 0], proj[:, 1], c=clf.predict(X_final_test), cmap="Paired")
plt.colorbar()
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')

plt.show()


