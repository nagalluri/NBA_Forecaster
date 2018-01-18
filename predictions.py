import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
import pandas as pd

adv12_13 = np.array(list(csv.reader(open("12-13adv.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))
adv13_14 = np.array(list(csv.reader(open("13-14adv.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))
adv14_15 = np.array(list(csv.reader(open("14-15adv.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))
adv15_16 = np.array(list(csv.reader(open("15-16adv.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))

trad13_14 = np.array(list(csv.reader(open("13-14trad.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))
trad14_15 = np.array(list(csv.reader(open("14-15trad.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))
trad15_16 = np.array(list(csv.reader(open("15-16trad.csv", "rb"), dialect=csv.excel_tab, delimiter=",")))

#Removing top row
adv12_13 = adv12_13[1:]
adv13_14 = adv13_14[1:]
adv14_15 = adv14_15[1:]
adv15_16 = adv15_16[1:]

trad13_14 = trad13_14[1:]
trad14_15 = trad14_15[1:]
trad15_16 = trad15_16[1:]


#-------Testing data-------------
test = adv15_16
# test = trad15_16

#shuffle data 
np.random.shuffle(test)
#grab the rank column
X_test_ranks = test.T[0]
#Grab columns between the ranks column and labels column
X_test = test.T[1:].T
#convert to float
test = test.astype(np.float)
#Grab true labels
# true_vals = test.T[-1:].ravel()
# print true_vals
#Normalize the data
X_test = preprocessing.scale(X_test)



#---------Training Data---------
#Concatenate multiple datasets and remove the rank column

data = np.concatenate((adv13_14, adv14_15), axis=0)

# data = adv14_15
# data = trad14_15
data = data.T[1:].T

#convert to float
data = data.astype(np.float)

#shuffle and remove labels column
np.random.shuffle(data)
X_train = data.T[:-1].T
y_train = data.T[-1:].ravel()
print X_train.shape
print y_train.shape
#normalize train data
X_train = preprocessing.scale(X_train)

#Split data into a validation and train set
# X_val = X_train.T[-50:]
# y_val = y_train.T[-50:].ravel()

# X_train = X_train.T[:-50]
# y_train = y_train.T[:-50].ravel()


#--------Train the Model-----------

clf = SVC()
clf.fit(X_train, y_train)


#---------Save Predictions----------

predictions = clf.predict(X_test)

d = {
    'Rank' : X_test_ranks,
    'All NBA' : predictions,
}
df = pd.DataFrame(data=d, columns=['Rank', 'All NBA'])
df.to_csv("my_predictions_adv.csv", index=False)
print "Saved predictions!" 













# # mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
# #                     solver='sgd', verbose=10, tol=1e-4, random_state=1)

# svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = svm.SVR(kernel='linear', C=1e3)
# svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)

# y_rbf = svr_rbf.fit(X_train, y_train)
# # y_lin = svr_lin.fit(X_train, y_train)
# # y_poly = svr_poly.fit(X_train, y_train)


# # print("Training set score: %f" % svr_poly.score(X_train, y_train))
# # print("Test set score: %f" % svr_poly.score(X_val, y_val))

# # print("Training set score: %f" % svr_lin.score(X_train, y_train))
# # print("Test set score: %f" % svr_lin.score(X_val, y_val))

# print("Training set score: %f" % svr_rbf.score(X_train, y_train))
# print("Val set score: %f" % svr_rbf.score(X_val, y_val))
# print("Test set score: %f" % svr_rbf.score(X_test, true_vals))
# # print X_train.shape
# # print X_test.shape
# # print X_test_ranks.shape
# predictions = svr_rbf.predict(X_test)
# # X_test_ranks = X_test_ranks.reshape((603, 1))
# # predictions = predictions.reshape((603, 1))
# # predictions = np.concatenate((X_test_ranks, predictions), axis=1)
# # predictions = np.concatenate((X_test_ranks, predictions), axis=1)

# d = {
#     'Rank' : X_test_ranks,
#     'Win Shares' : predictions,
# }
# df = pd.DataFrame(data=d, columns=['Rank', 'Win Shares'])
# df.to_csv("my_predictions.csv", index=False)
# print "Saved predictions!" 
# # print predictions
# # np.savetxt("predictions.csv", predictions, delimiter=" ")
# lw = 2


# print X_test.shape
# print true_vals.shape



# plt.scatter(X_test_ranks, true_vals, color='darkorange', label='data')
# plt.scatter(X_test_ranks, predictions, color='navy', label='RBF model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# # plt.show()


