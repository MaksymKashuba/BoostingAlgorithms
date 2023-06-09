import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from splitter2 import X_train, X_test, Y_train, Y_test
start_time = time.time()
clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, Y_train)
duration = time.time() - start_time
predicted = clf.predict(X_test)
print('Accuracy of the result is:')
print(np.mean(predicted == Y_test))
print("Execution time: {:.2f} seconds".format(duration))
