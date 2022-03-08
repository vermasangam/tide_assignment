import pandas as pd
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import datetime
import gzip
import json
import pickle
test = pd.read_csv('test_data.csv')
feats = ['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch',
       'DifferentPredictedTime', 'TimeMappingMatch', 'PredictedNameMatch',
       'ShortNameMatch', 'DifferentPredictedDate', 'PredictedAmountMatch',
       'PredictedTimeCloseMatch']
lr = pickle.load(open('LR_receipt_match.sav', 'rb'))
test['pred'] = lr.predict_proba(test[feats])[:,1]
test["rank"] = tmp.groupby(['receipt_id','feature_transaction_id'])["pred"].rank("first", ascending=False)
test = test.sort_values(['receipt_id','rank'])
test[['receipt_id','feature_transaction_id','pred','rank']].to_csv('matched_receipts.csv')