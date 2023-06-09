import time
import numpy as np
import lightgbm as lgb
from splitter2 import X_train, X_test, Y_train, Y_test, cat_df

start_time = time.time()
clf = lgb.LGBMClassifier()
clf.fit(X=X_train, y=Y_train, feature_name=list(X_train.columns), categorical_feature=cat_df)
duration = time.time() - start_time
predicted = clf.predict(X_test)
print('Accuracy of the result is:')
print(np.mean(predicted == Y_test))
print("Execution time: {:.2f} seconds".format(duration))