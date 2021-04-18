# Dependencies
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Adding processed data
train = pd.read_csv('./input/proc_train.csv')
test = pd.read_csv('./input/proc_test.csv')

# Model
def new_model(estimators):
    model= XGBClassifier(n_estimators=estimators)
    model.fit(train[fea], target)
    preds = model.predict(test[fea])
    return preds, model

# spliting target and features
target = train['churn_risk_score']
fea=['gender', 'region_category', 'membership_category', 'joined_through_referral', 'referral_id', 'preferred_offer_types', 'medium_of_operation', 'internet_option', 'used_special_discount', 'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback', 'membership_by_refer_id_min', 'membership_by_refer_id_max']

# saving model
preds, model = new_model(estimators=575)
filename = 'XGBM_model'
pickle.dump(model, open(filename, 'wb'))

# submission file 
submission = pd.DataFrame()
submission['customer_id'] = test.customer_id
submission['churn_risk_score'] = preds
submission.to_csv('prediction.csv', index= False)
