import numpy as np 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report

p1test = np.load('p1_bottleneck_features_test.npy')
p2test = np.load('p2_bottleneck_features_test.npy')
p3test = np.load('p3_bottleneck_features_test.npy')
assert p1test.shape == p2test.shape and p1test.shape == p3test.shape

p1tl = np.load('p1_test_labels.npy')
p2tl = np.load('p2_test_labels.npy')
p3tl = np.load('p3_test_labels.npy')
assert p1tl.shape == p2tl.shape and p1tl.shape == p3tl.shape


p1train = np.load('p1_bottleneck_features_train.npy')
p2train = np.load('p2_bottleneck_features_train.npy')
p3train = np.load('p3_bottleneck_features_train.npy')
assert p1train.shape == p2train.shape and p1train.shape == p3train.shape

p1trl = np.load('p1_train_labels.npy')
p2trl = np.load('p2_train_labels.npy')
p3trl = np.load('p3_train_labels.npy')
assert p1trl.shape == p2trl.shape and p1trl.shape == p3trl.shape
assert np.array_equal(p1tl, p2tl) and np.array_equal(p1tl, p3tl) 
assert np.array_equal(p1trl, p2trl) and np.array_equal(p1trl, p3trl) 

def plot_confusion_matrix(cm):
    
    plt.imshow(cm, interpolation='nearest', cmap='hot')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix4.svg')



X_train = np.hstack([p1train, p2train, p3train])
X_test = np.hstack([p1test, p2test, p3test])

y_train = p1trl 
y_test = p1tl


param = [{
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        }, {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]


svm = SVC()

# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
clf = grid_search.GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)

clf.fit(X_train, y_train)
print(clf.best_params_)

print clf.score(X_train, y_train)
print clf.score(X_test, y_test)

y_test_predict = clf.predict(X_test)
labels = range(1, 201)
cm = confusion_matrix(y_test, y_test_predict, labels=labels)
plot_confusion_matrix()