import numpy as np 
from sklearn import linear_model

p1test = np.load('p1_bottleneck_features_test.npy')
p2test = np.load('p2_bottleneck_features_test.npy')
p3test = np.load('p3_bottleneck_features_test.npy')

assert p1test.shape == p2test.shape and p1test.shape == p3test.shape
print p1test.shape, p2test.shape, p3test.shape

p1tl = np.load('p1_test_labels.npy')
p2tl = np.load('p2_test_labels.npy')
p3tl = np.load('p3_test_labels.npy')
assert p1tl.shape == p2tl.shape and p1tl.shape == p3tl.shape
print p1tl.shape, p2tl.shape, p3tl.shape


p1train = np.load('p1_bottleneck_features_train.npy')
p2train = np.load('p2_bottleneck_features_train.npy')
p3train = np.load('p3_bottleneck_features_train.npy')

assert p1train.shape == p2train.shape and p1train.shape == p3train.shape
print p1train.shape, p2train.shape, p3train.shape

p1trl = np.load('p1_train_labels.npy')
p2trl = np.load('p2_train_labels.npy')
p3trl = np.load('p3_train_labels.npy')

assert p1trl.shape == p2trl.shape and p1trl.shape == p3trl.shape
print p1trl.shape, p2trl.shape, p3trl.shape

train_data = np.hstack([p1train, p2train, p3train])
test_data = np.hstack([p1test, p2test, p3test])

train_labels = p1trl 
test_labels = p1tl

print train_data.shape, test_data.shape

assert np.array_equal(p1tl, p2tl) and np.array_equal(p1tl, p3tl) 
assert np.array_equal(p1trl, p2trl) and np.array_equal(p1trl, p3trl) 



clf = linear_model.SGDClassifier(verbose=1)
clf.fit(train_data, train_labels)
print clf.score(test_data, test_labels)

