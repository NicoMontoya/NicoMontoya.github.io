---
layout: post
title: Should we fix the wells?
subtitle: Predictive Model Exploration with Tanzania Water Wells
image: /img/WaterHole008A.jpg
published: True
---

## **Goal**

I participated in a competition using a dataset which contained information about water wells in Tanzania.  The competition was to build the best predictive model for classifying the water wells in Tanzania.  There were three classes of wells, functioning, non-functioning, and functioning but needs repairs.  Using some machine learning techniques, this post will go through my process of fine tuning a model to help with a real world problem.


**Quick Baseline**

The first thing I look for when I train a model is a quick dumb baseline. In other words, what accuracy should I be able to obtain if I predicted that every well would fit into the most likely case.  Essentially guess c on every answer on a test.  




    functional                 0.543081
    non functional             0.384242
    functional needs repair    0.072677
    Name: status_group, dtype: float64


It is simple enough to find this by counting the amount of each well from past data.  In this case the most likely scenario is that a well would be functioning. One can see from the chart above that the baseline is 54%.  If my model can be over 54% accurate, then I know I am heading in the right direction.

**Note**: I try my best to be cognizant of the actual problem at hand.  You can get 54% accuracy by guessing all wells will be functional, theoretically.  What is most useful though is if you can determine which wells are functional and just need to be repaired.  So guessing all wells will be functional will get you just the bare baseline, but it would not be useful at all.

# Beat the baseline


```python
y_train_target = y_train['status_group']
```


```python
y_train['status_group'].value_counts(normalize=True)
```




    functional                 0.543081
    non functional             0.384242
    functional needs repair    0.072677
    Name: status_group, dtype: float64




```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X_train_numbers = X_train[X_train.describe().columns]

basic_model = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42)
cross_val_score(basic_model, X_train_numbers, y_train_target, scoring='accuracy', cv=5).mean()
```

    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)





    0.551801290559174



Okay... that did not do too well. But a slight improvement nonetheless. Moving on

# Clean data, feature engineer, test different models


```python
## function that cleans the data, adds some features, ordinally encodes everything that is left

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def wrangle_wells(X):
    X = X.copy()
    
    # feature: get days passed since date recorded
    X['last_day'] = '2014-01-01'
    X['date_recorded'] = pd.to_datetime(X['date_recorded'], infer_datetime_format=True)
    X['last_day'] = pd.to_datetime(X['last_day'], infer_datetime_format=True)
    X['offset_days'] = X['last_day'] - X['date_recorded']
    X['offset_days'] = X['offset_days'].dt.days
    
    # month date was recorded
    X['month_recorded'] = X['date_recorded'].dt.month
    
    # years in service
    X['construction_year'].fillna(X.groupby(['region', 'district_code'])['construction_year'].transform('median'), inplace=True)
    X['construction_year'].fillna(X.groupby(['region'])['construction_year'].transform('median'), inplace=True)
    X['construction_year'].fillna(X.groupby(['district_code'])['construction_year'].transform('median'), inplace=True)
    X['construction_year'].fillna(X['construction_year'].median(), inplace=True)
    
    X['years_service'] = X.date_recorded.dt.year - X.construction_year
    
    # lower levels for funder categoricals
    X.loc[X['funder'].isin((X['funder'].value_counts()[X['funder'].value_counts() < 850]).index), 'funder'] = 'other'
    
    # lower levels for installer
    X.loc[X['installer'].isin((X['installer'].value_counts()[X['installer'].value_counts() < 620]).index), 'installer'] = 'other'
    
    # Try and make colum data a little more uniform
    X.waterpoint_type = X.waterpoint_type.str.lower()
    X.funder = X.funder.str.lower()
    X.basin = X.basin.str.lower()
    X.region = X.region.str.lower()
    X.source = X.source.str.lower()
    X.lga = X.lga.str.lower()
    X.management = X.management.str.lower()
    X.quantity = X.quantity.str.lower()
    X.water_quality = X.water_quality.str.lower()
    X.payment_type = X.payment_type.str.lower()
    X.extraction_type = X.extraction_type.str.lower()
    
    # Lower cardinality of extraction type by adding to 'other' category
    X['extraction_type'] = X['extraction_type'].replace({'other - mkulima/shinyanga': 'other'})
    
    
    X = X.drop(columns=['recorded_by', 'quantity_group', 'date_recorded', 'wpt_name', 'num_private', 'subvillage',
                       'region_code', 'management_group', 'extraction_type_group', 'extraction_type_class',
                       'scheme_name', 'payment', 'water_quality', 'source_type', 'source_class', 'waterpoint_type_group',
                       'ward', 'public_meeting', 'last_day', 'construction_year'])
    
    xyscaler = StandardScaler() 
    xyscaler.fit_transform(X[['latitude','longitude']])

    X["rot45X"] = .707* X['longitude'] + .707* X['latitude'] 
    X["rot45Y"] = .707* X['longitude'] - .707* X['latitude']

    X["rot30X"] = (1.732/2)* X['latitude'] + (1./2)* X['longitude'] 
    X["rot30Y"] = (1.732/2)* X['longitude'] - (1./2)* X['latitude']

    X["rot60X"] = (1./2)* X['latitude'] + (1.732/2)* X['longitude'] 
    X["rot60Y"] = (1./2)* X['longitude'] - (1.732/2)* X['latitude']

    X["radial_r"] = np.sqrt( np.power(X['longitude'],2) + np.power(X['latitude'],2) )
    
    
    return X
```


```python
## wrangle the test and train data

X_test1 = wrangle_wells(X_train)
X_test2 = wrangle_wells(X_test)
```


```python
## pre process train data to Ordinally Encode and then impute average values for all NaN's

from sklearn.pipeline import make_pipeline

features = ['id', 'amount_tsh', 'funder', 'gps_height',
       'installer', 'longitude', 'latitude',
       'basin', 'region', 'district_code', 'lga',
       'population',
       'scheme_management', 'permit',
       'extraction_type',
       'management', 'payment_type',
       'quality_group', 'quantity',
       'source', 'waterpoint_type',
       'offset_days', 'month_recorded', 'years_service',
       'rot45X', 'rot45Y', 'rot30X', 'rot30Y', 'rot60X', 'rot60Y', 'radial_r'
   ]

preprocessor = make_pipeline(ce.OrdinalEncoder(), SimpleImputer())
X_test1 = preprocessor.fit_transform(X_test1)
X_test1 = pd.DataFrame(X_test1, columns=features)
```


```python
# transform test data in the same way as the train data

X_teste2 = preprocessor.transform(X_test2)
X_teste2 = pd.DataFrame(X_teste2, columns=features)
```


```python
## test different models to see which will work best

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

models = [LogisticRegression(solver='lbfgs', max_iter=1000),
          DecisionTreeClassifier(max_depth=3),
          DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(max_depth=3, n_estimators=50, n_jobs=-1, random_state=42),
          RandomForestClassifier(max_depth=None, n_estimators=50, n_jobs=-1, random_state=42),
          XGBClassifier(max_depth=3, n_estimators=50, n_jobs=-1, random_state=42)]

for model in models:
  print(model, '\n')
  score = cross_val_score(model, X_test1, y_train_target, scoring='accuracy', cv=5).mean()
  print('Cross_Validation Accuracy:', score, '\n', '\n')
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=1000, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False) 
    


    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)
    /Users/lambda_school_loaner_95/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:758: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
      "of iterations.", ConvergenceWarning)


    Cross_Validation Accuracy: 0.6063973696515584 
     
    
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best') 
    
    Cross_Validation Accuracy: 0.6934514989614184 
     
    
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best') 
    
    Cross_Validation Accuracy: 0.7466330313376597 
     
    
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
                oob_score=False, random_state=42, verbose=0, warm_start=False) 
    
    Cross_Validation Accuracy: 0.6887042027413878 
     
    
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
                oob_score=False, random_state=42, verbose=0, warm_start=False) 
    
    Cross_Validation Accuracy: 0.8069023510130691 
     
    
    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=50,
           n_jobs=-1, nthread=None, objective='binary:logistic',
           random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
           seed=None, silent=True, subsample=1) 
    
    Cross_Validation Accuracy: 0.7344784863650176 
     
    


At quick glance, it is nice to see that the data cleaning did help, even with the logistic regression model, the scores improved.  It is nice to see some significant jumps in scores with other models as well.  Even something as simple as a single decision tree can get the score up in the 70's.  Maybe due to some leakage somewhere.. maybe there are some highly predictive features in the model.  Not sure yet.  Either way, the Random Forest Classifier did the best, so that is what I will stick with

# Model that Overfits


```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

searchRF = RandomizedSearchCV(
    estimator = RandomForestClassifier(n_jobs=-1, random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    verbose=10,
    return_train_score=True,
    random_state=42
)

searchRF.fit(X_test1, y_train_target)

```


```python
resultsRF = pd.DataFrame(searchRF.cv_results_)
resultsRF.sort_values(by='rank_test_score')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_n_estimators</th>
      <th>param_min_samples_split</th>
      <th>param_min_samples_leaf</th>
      <th>param_max_features</th>
      <th>param_max_depth</th>
      <th>param_bootstrap</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>720.509379</td>
      <td>5.798710</td>
      <td>13.033700</td>
      <td>0.503160</td>
      <td>2000</td>
      <td>2</td>
      <td>4</td>
      <td>auto</td>
      <td>90</td>
      <td>False</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 2,...</td>
      <td>0.819880</td>
      <td>0.811716</td>
      <td>0.815152</td>
      <td>0.817088</td>
      <td>0.816720</td>
      <td>0.816111</td>
      <td>0.002675</td>
      <td>1</td>
      <td>0.951893</td>
      <td>0.951598</td>
      <td>0.952041</td>
      <td>0.950526</td>
      <td>0.951707</td>
      <td>0.951553</td>
      <td>0.000536</td>
    </tr>
    <tr>
      <th>33</th>
      <td>729.220552</td>
      <td>16.964503</td>
      <td>12.409712</td>
      <td>1.579188</td>
      <td>2000</td>
      <td>2</td>
      <td>4</td>
      <td>auto</td>
      <td>70</td>
      <td>False</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 2,...</td>
      <td>0.819880</td>
      <td>0.811716</td>
      <td>0.815152</td>
      <td>0.817088</td>
      <td>0.816720</td>
      <td>0.816111</td>
      <td>0.002675</td>
      <td>1</td>
      <td>0.951893</td>
      <td>0.951598</td>
      <td>0.952041</td>
      <td>0.950526</td>
      <td>0.951707</td>
      <td>0.951553</td>
      <td>0.000536</td>
    </tr>
    <tr>
      <th>9</th>
      <td>349.453606</td>
      <td>1.907269</td>
      <td>5.406091</td>
      <td>0.230610</td>
      <td>1000</td>
      <td>5</td>
      <td>4</td>
      <td>sqrt</td>
      <td>90</td>
      <td>False</td>
      <td>{'n_estimators': 1000, 'min_samples_split': 5,...</td>
      <td>0.820638</td>
      <td>0.811295</td>
      <td>0.815067</td>
      <td>0.816414</td>
      <td>0.815794</td>
      <td>0.815842</td>
      <td>0.002987</td>
      <td>3</td>
      <td>0.951851</td>
      <td>0.951619</td>
      <td>0.951999</td>
      <td>0.950442</td>
      <td>0.951917</td>
      <td>0.951566</td>
      <td>0.000576</td>
    </tr>
    <tr>
      <th>42</th>
      <td>5063.786933</td>
      <td>2447.550383</td>
      <td>3.359759</td>
      <td>0.492994</td>
      <td>600</td>
      <td>5</td>
      <td>1</td>
      <td>auto</td>
      <td>60</td>
      <td>True</td>
      <td>{'n_estimators': 600, 'min_samples_split': 5, ...</td>
      <td>0.820554</td>
      <td>0.813736</td>
      <td>0.816835</td>
      <td>0.813636</td>
      <td>0.813605</td>
      <td>0.815673</td>
      <td>0.002733</td>
      <td>4</td>
      <td>0.981481</td>
      <td>0.981902</td>
      <td>0.981860</td>
      <td>0.981566</td>
      <td>0.981524</td>
      <td>0.981667</td>
      <td>0.000178</td>
    </tr>
    <tr>
      <th>5</th>
      <td>408.932301</td>
      <td>2.189365</td>
      <td>10.959561</td>
      <td>1.006485</td>
      <td>1600</td>
      <td>5</td>
      <td>2</td>
      <td>sqrt</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 1600, 'min_samples_split': 5,...</td>
      <td>0.821227</td>
      <td>0.811800</td>
      <td>0.815909</td>
      <td>0.815152</td>
      <td>0.814278</td>
      <td>0.815673</td>
      <td>0.003102</td>
      <td>4</td>
      <td>0.954671</td>
      <td>0.954797</td>
      <td>0.955471</td>
      <td>0.954482</td>
      <td>0.954926</td>
      <td>0.954870</td>
      <td>0.000335</td>
    </tr>
    <tr>
      <th>18</th>
      <td>493.976887</td>
      <td>6.552700</td>
      <td>7.843262</td>
      <td>0.624889</td>
      <td>1400</td>
      <td>5</td>
      <td>4</td>
      <td>sqrt</td>
      <td>30</td>
      <td>False</td>
      <td>{'n_estimators': 1400, 'min_samples_split': 5,...</td>
      <td>0.820217</td>
      <td>0.810959</td>
      <td>0.816077</td>
      <td>0.816498</td>
      <td>0.814531</td>
      <td>0.815657</td>
      <td>0.003001</td>
      <td>6</td>
      <td>0.950946</td>
      <td>0.950525</td>
      <td>0.951157</td>
      <td>0.950021</td>
      <td>0.950928</td>
      <td>0.950715</td>
      <td>0.000403</td>
    </tr>
    <tr>
      <th>19</th>
      <td>592.136366</td>
      <td>70.995140</td>
      <td>16.395862</td>
      <td>2.649887</td>
      <td>2000</td>
      <td>5</td>
      <td>1</td>
      <td>sqrt</td>
      <td>70</td>
      <td>True</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 5,...</td>
      <td>0.820217</td>
      <td>0.813147</td>
      <td>0.816498</td>
      <td>0.814731</td>
      <td>0.813605</td>
      <td>0.815640</td>
      <td>0.002564</td>
      <td>7</td>
      <td>0.982070</td>
      <td>0.982239</td>
      <td>0.982029</td>
      <td>0.981713</td>
      <td>0.981924</td>
      <td>0.981995</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>48</th>
      <td>511.176914</td>
      <td>5.466096</td>
      <td>13.474008</td>
      <td>1.147614</td>
      <td>2000</td>
      <td>5</td>
      <td>1</td>
      <td>sqrt</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 5,...</td>
      <td>0.820217</td>
      <td>0.813147</td>
      <td>0.816498</td>
      <td>0.814731</td>
      <td>0.813605</td>
      <td>0.815640</td>
      <td>0.002564</td>
      <td>7</td>
      <td>0.982028</td>
      <td>0.982239</td>
      <td>0.982050</td>
      <td>0.981734</td>
      <td>0.981924</td>
      <td>0.981995</td>
      <td>0.000165</td>
    </tr>
    <tr>
      <th>7</th>
      <td>452.435272</td>
      <td>7.082760</td>
      <td>10.639169</td>
      <td>0.572026</td>
      <td>1800</td>
      <td>5</td>
      <td>2</td>
      <td>auto</td>
      <td>30</td>
      <td>True</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 5,...</td>
      <td>0.821059</td>
      <td>0.811885</td>
      <td>0.815741</td>
      <td>0.814815</td>
      <td>0.814531</td>
      <td>0.815606</td>
      <td>0.003013</td>
      <td>9</td>
      <td>0.953976</td>
      <td>0.953598</td>
      <td>0.954524</td>
      <td>0.953556</td>
      <td>0.954105</td>
      <td>0.953952</td>
      <td>0.000356</td>
    </tr>
    <tr>
      <th>36</th>
      <td>145.618699</td>
      <td>0.554082</td>
      <td>2.863658</td>
      <td>0.061974</td>
      <td>600</td>
      <td>10</td>
      <td>1</td>
      <td>sqrt</td>
      <td>None</td>
      <td>True</td>
      <td>{'n_estimators': 600, 'min_samples_split': 10,...</td>
      <td>0.819712</td>
      <td>0.813484</td>
      <td>0.816667</td>
      <td>0.814646</td>
      <td>0.813184</td>
      <td>0.815539</td>
      <td>0.002419</td>
      <td>10</td>
      <td>0.935184</td>
      <td>0.936299</td>
      <td>0.935438</td>
      <td>0.934933</td>
      <td>0.935461</td>
      <td>0.935463</td>
      <td>0.000460</td>
    </tr>
    <tr>
      <th>12</th>
      <td>730.862590</td>
      <td>9.341759</td>
      <td>12.571742</td>
      <td>0.440390</td>
      <td>2000</td>
      <td>10</td>
      <td>2</td>
      <td>auto</td>
      <td>90</td>
      <td>False</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 10...</td>
      <td>0.819880</td>
      <td>0.812390</td>
      <td>0.814899</td>
      <td>0.814562</td>
      <td>0.815962</td>
      <td>0.815539</td>
      <td>0.002462</td>
      <td>10</td>
      <td>0.965319</td>
      <td>0.965530</td>
      <td>0.966014</td>
      <td>0.964141</td>
      <td>0.965279</td>
      <td>0.965257</td>
      <td>0.000616</td>
    </tr>
    <tr>
      <th>37</th>
      <td>71.089106</td>
      <td>0.828702</td>
      <td>0.907931</td>
      <td>0.032422</td>
      <td>200</td>
      <td>5</td>
      <td>4</td>
      <td>sqrt</td>
      <td>30</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 5, ...</td>
      <td>0.821816</td>
      <td>0.810875</td>
      <td>0.814141</td>
      <td>0.815825</td>
      <td>0.814952</td>
      <td>0.815522</td>
      <td>0.003565</td>
      <td>12</td>
      <td>0.950525</td>
      <td>0.950462</td>
      <td>0.950758</td>
      <td>0.949790</td>
      <td>0.950528</td>
      <td>0.950412</td>
      <td>0.000327</td>
    </tr>
    <tr>
      <th>21</th>
      <td>94.359531</td>
      <td>0.736202</td>
      <td>1.287312</td>
      <td>0.110311</td>
      <td>200</td>
      <td>2</td>
      <td>4</td>
      <td>auto</td>
      <td>30</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.821816</td>
      <td>0.810875</td>
      <td>0.814141</td>
      <td>0.815825</td>
      <td>0.814952</td>
      <td>0.815522</td>
      <td>0.003565</td>
      <td>12</td>
      <td>0.950525</td>
      <td>0.950462</td>
      <td>0.950758</td>
      <td>0.949790</td>
      <td>0.950528</td>
      <td>0.950412</td>
      <td>0.000327</td>
    </tr>
    <tr>
      <th>32</th>
      <td>105.976748</td>
      <td>7.346351</td>
      <td>1.921987</td>
      <td>0.130128</td>
      <td>400</td>
      <td>10</td>
      <td>1</td>
      <td>auto</td>
      <td>100</td>
      <td>True</td>
      <td>{'n_estimators': 400, 'min_samples_split': 10,...</td>
      <td>0.819460</td>
      <td>0.813063</td>
      <td>0.815488</td>
      <td>0.815152</td>
      <td>0.814278</td>
      <td>0.815488</td>
      <td>0.002156</td>
      <td>14</td>
      <td>0.935121</td>
      <td>0.935815</td>
      <td>0.935795</td>
      <td>0.934470</td>
      <td>0.935630</td>
      <td>0.935366</td>
      <td>0.000514</td>
    </tr>
    <tr>
      <th>28</th>
      <td>296.343008</td>
      <td>1.363506</td>
      <td>6.505740</td>
      <td>0.440909</td>
      <td>1200</td>
      <td>2</td>
      <td>2</td>
      <td>sqrt</td>
      <td>100</td>
      <td>True</td>
      <td>{'n_estimators': 1200, 'min_samples_split': 2,...</td>
      <td>0.820470</td>
      <td>0.811716</td>
      <td>0.815236</td>
      <td>0.814646</td>
      <td>0.815289</td>
      <td>0.815471</td>
      <td>0.002823</td>
      <td>15</td>
      <td>0.963215</td>
      <td>0.963720</td>
      <td>0.964794</td>
      <td>0.963131</td>
      <td>0.962943</td>
      <td>0.963561</td>
      <td>0.000668</td>
    </tr>
    <tr>
      <th>1</th>
      <td>381.245175</td>
      <td>0.959058</td>
      <td>11.912072</td>
      <td>2.487934</td>
      <td>1200</td>
      <td>2</td>
      <td>2</td>
      <td>sqrt</td>
      <td>60</td>
      <td>True</td>
      <td>{'n_estimators': 1200, 'min_samples_split': 2,...</td>
      <td>0.820470</td>
      <td>0.811716</td>
      <td>0.815236</td>
      <td>0.814646</td>
      <td>0.815289</td>
      <td>0.815471</td>
      <td>0.002823</td>
      <td>15</td>
      <td>0.963215</td>
      <td>0.963720</td>
      <td>0.964794</td>
      <td>0.963131</td>
      <td>0.962943</td>
      <td>0.963561</td>
      <td>0.000668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>388.391566</td>
      <td>43.832043</td>
      <td>8.529679</td>
      <td>1.078404</td>
      <td>1400</td>
      <td>5</td>
      <td>2</td>
      <td>auto</td>
      <td>None</td>
      <td>True</td>
      <td>{'n_estimators': 1400, 'min_samples_split': 5,...</td>
      <td>0.820638</td>
      <td>0.811969</td>
      <td>0.815320</td>
      <td>0.814983</td>
      <td>0.814278</td>
      <td>0.815438</td>
      <td>0.002851</td>
      <td>17</td>
      <td>0.954755</td>
      <td>0.954608</td>
      <td>0.955408</td>
      <td>0.954377</td>
      <td>0.955031</td>
      <td>0.954836</td>
      <td>0.000356</td>
    </tr>
    <tr>
      <th>46</th>
      <td>253.138390</td>
      <td>3.622253</td>
      <td>5.038906</td>
      <td>0.413614</td>
      <td>1000</td>
      <td>10</td>
      <td>1</td>
      <td>auto</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 1000, 'min_samples_split': 10...</td>
      <td>0.819207</td>
      <td>0.812895</td>
      <td>0.816582</td>
      <td>0.814815</td>
      <td>0.813521</td>
      <td>0.815404</td>
      <td>0.002282</td>
      <td>18</td>
      <td>0.934847</td>
      <td>0.936552</td>
      <td>0.935711</td>
      <td>0.935101</td>
      <td>0.935419</td>
      <td>0.935526</td>
      <td>0.000590</td>
    </tr>
    <tr>
      <th>10</th>
      <td>700.544952</td>
      <td>9.882919</td>
      <td>12.649523</td>
      <td>0.506787</td>
      <td>1800</td>
      <td>10</td>
      <td>1</td>
      <td>auto</td>
      <td>80</td>
      <td>False</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 10...</td>
      <td>0.818450</td>
      <td>0.812221</td>
      <td>0.814815</td>
      <td>0.815488</td>
      <td>0.813016</td>
      <td>0.814798</td>
      <td>0.002174</td>
      <td>19</td>
      <td>0.983922</td>
      <td>0.984680</td>
      <td>0.984848</td>
      <td>0.983817</td>
      <td>0.984512</td>
      <td>0.984356</td>
      <td>0.000412</td>
    </tr>
    <tr>
      <th>38</th>
      <td>680.305126</td>
      <td>24.599462</td>
      <td>12.119634</td>
      <td>1.635553</td>
      <td>1800</td>
      <td>5</td>
      <td>1</td>
      <td>sqrt</td>
      <td>20</td>
      <td>False</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 5,...</td>
      <td>0.819291</td>
      <td>0.812390</td>
      <td>0.816162</td>
      <td>0.813721</td>
      <td>0.811753</td>
      <td>0.814663</td>
      <td>0.002764</td>
      <td>20</td>
      <td>0.972137</td>
      <td>0.972537</td>
      <td>0.971907</td>
      <td>0.971633</td>
      <td>0.973276</td>
      <td>0.972298</td>
      <td>0.000572</td>
    </tr>
    <tr>
      <th>26</th>
      <td>682.136706</td>
      <td>25.348266</td>
      <td>11.776721</td>
      <td>1.152025</td>
      <td>1800</td>
      <td>5</td>
      <td>1</td>
      <td>auto</td>
      <td>20</td>
      <td>False</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 5,...</td>
      <td>0.819291</td>
      <td>0.812390</td>
      <td>0.816162</td>
      <td>0.813721</td>
      <td>0.811753</td>
      <td>0.814663</td>
      <td>0.002764</td>
      <td>20</td>
      <td>0.972137</td>
      <td>0.972537</td>
      <td>0.971907</td>
      <td>0.971633</td>
      <td>0.973276</td>
      <td>0.972298</td>
      <td>0.000572</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1112.538896</td>
      <td>1274.519431</td>
      <td>14.684893</td>
      <td>2.221643</td>
      <td>1800</td>
      <td>2</td>
      <td>1</td>
      <td>sqrt</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 2,...</td>
      <td>0.819123</td>
      <td>0.811632</td>
      <td>0.814478</td>
      <td>0.813973</td>
      <td>0.813352</td>
      <td>0.814512</td>
      <td>0.002498</td>
      <td>22</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>65.837109</td>
      <td>5.867067</td>
      <td>1.172589</td>
      <td>0.193521</td>
      <td>200</td>
      <td>2</td>
      <td>1</td>
      <td>sqrt</td>
      <td>20</td>
      <td>True</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.819544</td>
      <td>0.813568</td>
      <td>0.813973</td>
      <td>0.810606</td>
      <td>0.811921</td>
      <td>0.813923</td>
      <td>0.003057</td>
      <td>23</td>
      <td>0.969065</td>
      <td>0.968939</td>
      <td>0.967172</td>
      <td>0.968497</td>
      <td>0.969277</td>
      <td>0.968590</td>
      <td>0.000754</td>
    </tr>
    <tr>
      <th>30</th>
      <td>104.894440</td>
      <td>6.917692</td>
      <td>1.925417</td>
      <td>0.408957</td>
      <td>400</td>
      <td>2</td>
      <td>2</td>
      <td>auto</td>
      <td>20</td>
      <td>True</td>
      <td>{'n_estimators': 400, 'min_samples_split': 2, ...</td>
      <td>0.816514</td>
      <td>0.813063</td>
      <td>0.813721</td>
      <td>0.812626</td>
      <td>0.813100</td>
      <td>0.813805</td>
      <td>0.001399</td>
      <td>24</td>
      <td>0.931438</td>
      <td>0.931627</td>
      <td>0.930997</td>
      <td>0.930787</td>
      <td>0.932873</td>
      <td>0.931545</td>
      <td>0.000729</td>
    </tr>
    <tr>
      <th>49</th>
      <td>551.403973</td>
      <td>146.087543</td>
      <td>9.980960</td>
      <td>4.734288</td>
      <td>1800</td>
      <td>5</td>
      <td>2</td>
      <td>auto</td>
      <td>80</td>
      <td>False</td>
      <td>{'n_estimators': 1800, 'min_samples_split': 5,...</td>
      <td>0.816935</td>
      <td>0.810201</td>
      <td>0.813215</td>
      <td>0.813636</td>
      <td>0.813100</td>
      <td>0.813418</td>
      <td>0.002140</td>
      <td>25</td>
      <td>0.994213</td>
      <td>0.994423</td>
      <td>0.994529</td>
      <td>0.994423</td>
      <td>0.994297</td>
      <td>0.994377</td>
      <td>0.000110</td>
    </tr>
    <tr>
      <th>31</th>
      <td>610.815839</td>
      <td>9.270596</td>
      <td>12.973359</td>
      <td>0.845996</td>
      <td>1600</td>
      <td>5</td>
      <td>2</td>
      <td>sqrt</td>
      <td>80</td>
      <td>False</td>
      <td>{'n_estimators': 1600, 'min_samples_split': 5,...</td>
      <td>0.816850</td>
      <td>0.810369</td>
      <td>0.813215</td>
      <td>0.813384</td>
      <td>0.813184</td>
      <td>0.813401</td>
      <td>0.002058</td>
      <td>26</td>
      <td>0.994150</td>
      <td>0.994381</td>
      <td>0.994592</td>
      <td>0.994360</td>
      <td>0.994297</td>
      <td>0.994356</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>23</th>
      <td>827.512180</td>
      <td>51.275973</td>
      <td>18.382350</td>
      <td>2.061249</td>
      <td>1600</td>
      <td>5</td>
      <td>2</td>
      <td>sqrt</td>
      <td>90</td>
      <td>False</td>
      <td>{'n_estimators': 1600, 'min_samples_split': 5,...</td>
      <td>0.816850</td>
      <td>0.810369</td>
      <td>0.813215</td>
      <td>0.813384</td>
      <td>0.813184</td>
      <td>0.813401</td>
      <td>0.002058</td>
      <td>26</td>
      <td>0.994150</td>
      <td>0.994381</td>
      <td>0.994592</td>
      <td>0.994360</td>
      <td>0.994297</td>
      <td>0.994356</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52.588997</td>
      <td>0.340683</td>
      <td>1.027237</td>
      <td>0.056823</td>
      <td>200</td>
      <td>2</td>
      <td>1</td>
      <td>auto</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.818534</td>
      <td>0.810875</td>
      <td>0.813384</td>
      <td>0.811869</td>
      <td>0.811163</td>
      <td>0.813165</td>
      <td>0.002822</td>
      <td>28</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>445.006858</td>
      <td>5.532842</td>
      <td>9.274371</td>
      <td>1.408735</td>
      <td>1200</td>
      <td>2</td>
      <td>2</td>
      <td>auto</td>
      <td>70</td>
      <td>False</td>
      <td>{'n_estimators': 1200, 'min_samples_split': 2,...</td>
      <td>0.817187</td>
      <td>0.809275</td>
      <td>0.813468</td>
      <td>0.813552</td>
      <td>0.811753</td>
      <td>0.813047</td>
      <td>0.002588</td>
      <td>29</td>
      <td>0.997538</td>
      <td>0.997853</td>
      <td>0.997811</td>
      <td>0.997580</td>
      <td>0.997601</td>
      <td>0.997677</td>
      <td>0.000129</td>
    </tr>
    <tr>
      <th>16</th>
      <td>77.184565</td>
      <td>4.014259</td>
      <td>1.006591</td>
      <td>0.046729</td>
      <td>200</td>
      <td>5</td>
      <td>2</td>
      <td>sqrt</td>
      <td>40</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 5, ...</td>
      <td>0.816598</td>
      <td>0.810454</td>
      <td>0.811279</td>
      <td>0.813805</td>
      <td>0.812931</td>
      <td>0.813013</td>
      <td>0.002147</td>
      <td>30</td>
      <td>0.994150</td>
      <td>0.994192</td>
      <td>0.994360</td>
      <td>0.994276</td>
      <td>0.994045</td>
      <td>0.994205</td>
      <td>0.000108</td>
    </tr>
    <tr>
      <th>8</th>
      <td>220.967531</td>
      <td>1.620002</td>
      <td>3.380624</td>
      <td>0.068433</td>
      <td>600</td>
      <td>2</td>
      <td>2</td>
      <td>sqrt</td>
      <td>50</td>
      <td>False</td>
      <td>{'n_estimators': 600, 'min_samples_split': 2, ...</td>
      <td>0.817608</td>
      <td>0.809107</td>
      <td>0.812542</td>
      <td>0.812795</td>
      <td>0.812426</td>
      <td>0.812896</td>
      <td>0.002717</td>
      <td>31</td>
      <td>0.997622</td>
      <td>0.997727</td>
      <td>0.997896</td>
      <td>0.997538</td>
      <td>0.997622</td>
      <td>0.997681</td>
      <td>0.000123</td>
    </tr>
    <tr>
      <th>35</th>
      <td>73.356926</td>
      <td>0.466879</td>
      <td>1.013600</td>
      <td>0.053833</td>
      <td>200</td>
      <td>2</td>
      <td>2</td>
      <td>sqrt</td>
      <td>50</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.817355</td>
      <td>0.810622</td>
      <td>0.812121</td>
      <td>0.812205</td>
      <td>0.811079</td>
      <td>0.812677</td>
      <td>0.002416</td>
      <td>32</td>
      <td>0.997601</td>
      <td>0.997601</td>
      <td>0.997811</td>
      <td>0.997664</td>
      <td>0.997433</td>
      <td>0.997622</td>
      <td>0.000122</td>
    </tr>
    <tr>
      <th>20</th>
      <td>329.730996</td>
      <td>10.687908</td>
      <td>6.618619</td>
      <td>0.486682</td>
      <td>1000</td>
      <td>5</td>
      <td>4</td>
      <td>auto</td>
      <td>None</td>
      <td>True</td>
      <td>{'n_estimators': 1000, 'min_samples_split': 5,...</td>
      <td>0.815925</td>
      <td>0.809949</td>
      <td>0.813636</td>
      <td>0.811953</td>
      <td>0.810406</td>
      <td>0.812374</td>
      <td>0.002197</td>
      <td>33</td>
      <td>0.908163</td>
      <td>0.909173</td>
      <td>0.908123</td>
      <td>0.908628</td>
      <td>0.908905</td>
      <td>0.908598</td>
      <td>0.000410</td>
    </tr>
    <tr>
      <th>39</th>
      <td>148.392296</td>
      <td>1.916126</td>
      <td>2.166154</td>
      <td>0.062569</td>
      <td>400</td>
      <td>2</td>
      <td>2</td>
      <td>sqrt</td>
      <td>80</td>
      <td>False</td>
      <td>{'n_estimators': 400, 'min_samples_split': 2, ...</td>
      <td>0.816261</td>
      <td>0.809275</td>
      <td>0.812205</td>
      <td>0.812290</td>
      <td>0.811500</td>
      <td>0.812306</td>
      <td>0.002258</td>
      <td>34</td>
      <td>0.997643</td>
      <td>0.997790</td>
      <td>0.997917</td>
      <td>0.997601</td>
      <td>0.997538</td>
      <td>0.997698</td>
      <td>0.000137</td>
    </tr>
    <tr>
      <th>22</th>
      <td>127.287248</td>
      <td>1.548906</td>
      <td>2.565004</td>
      <td>0.202288</td>
      <td>400</td>
      <td>2</td>
      <td>4</td>
      <td>sqrt</td>
      <td>70</td>
      <td>True</td>
      <td>{'n_estimators': 400, 'min_samples_split': 2, ...</td>
      <td>0.815672</td>
      <td>0.808602</td>
      <td>0.813131</td>
      <td>0.811869</td>
      <td>0.811163</td>
      <td>0.812088</td>
      <td>0.002323</td>
      <td>35</td>
      <td>0.908205</td>
      <td>0.908458</td>
      <td>0.908018</td>
      <td>0.908586</td>
      <td>0.908842</td>
      <td>0.908422</td>
      <td>0.000288</td>
    </tr>
    <tr>
      <th>29</th>
      <td>283.505694</td>
      <td>6.881977</td>
      <td>5.466346</td>
      <td>0.221133</td>
      <td>1200</td>
      <td>10</td>
      <td>4</td>
      <td>sqrt</td>
      <td>100</td>
      <td>True</td>
      <td>{'n_estimators': 1200, 'min_samples_split': 10...</td>
      <td>0.815420</td>
      <td>0.809612</td>
      <td>0.813384</td>
      <td>0.811785</td>
      <td>0.809816</td>
      <td>0.812003</td>
      <td>0.002197</td>
      <td>36</td>
      <td>0.901513</td>
      <td>0.902123</td>
      <td>0.902104</td>
      <td>0.901747</td>
      <td>0.902087</td>
      <td>0.901915</td>
      <td>0.000245</td>
    </tr>
    <tr>
      <th>4</th>
      <td>298.373851</td>
      <td>8.223683</td>
      <td>5.832545</td>
      <td>0.164158</td>
      <td>1200</td>
      <td>10</td>
      <td>2</td>
      <td>sqrt</td>
      <td>20</td>
      <td>True</td>
      <td>{'n_estimators': 1200, 'min_samples_split': 10...</td>
      <td>0.814325</td>
      <td>0.810622</td>
      <td>0.812879</td>
      <td>0.810438</td>
      <td>0.811500</td>
      <td>0.811953</td>
      <td>0.001467</td>
      <td>37</td>
      <td>0.899619</td>
      <td>0.901576</td>
      <td>0.900905</td>
      <td>0.900715</td>
      <td>0.901582</td>
      <td>0.900880</td>
      <td>0.000721</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2298.658169</td>
      <td>1564.542364</td>
      <td>8.185188</td>
      <td>0.225765</td>
      <td>1600</td>
      <td>10</td>
      <td>4</td>
      <td>sqrt</td>
      <td>50</td>
      <td>True</td>
      <td>{'n_estimators': 1600, 'min_samples_split': 10...</td>
      <td>0.815588</td>
      <td>0.809612</td>
      <td>0.812795</td>
      <td>0.811616</td>
      <td>0.809901</td>
      <td>0.811902</td>
      <td>0.002178</td>
      <td>38</td>
      <td>0.901534</td>
      <td>0.902292</td>
      <td>0.902378</td>
      <td>0.902189</td>
      <td>0.902298</td>
      <td>0.902138</td>
      <td>0.000308</td>
    </tr>
    <tr>
      <th>27</th>
      <td>48.510454</td>
      <td>3.025912</td>
      <td>0.808255</td>
      <td>0.038291</td>
      <td>200</td>
      <td>10</td>
      <td>4</td>
      <td>sqrt</td>
      <td>90</td>
      <td>True</td>
      <td>{'n_estimators': 200, 'min_samples_split': 10,...</td>
      <td>0.814073</td>
      <td>0.808265</td>
      <td>0.813300</td>
      <td>0.812290</td>
      <td>0.809901</td>
      <td>0.811566</td>
      <td>0.002167</td>
      <td>39</td>
      <td>0.901702</td>
      <td>0.901934</td>
      <td>0.901494</td>
      <td>0.901747</td>
      <td>0.901582</td>
      <td>0.901692</td>
      <td>0.000150</td>
    </tr>
    <tr>
      <th>43</th>
      <td>50.583115</td>
      <td>3.968498</td>
      <td>0.882425</td>
      <td>0.078363</td>
      <td>200</td>
      <td>2</td>
      <td>4</td>
      <td>auto</td>
      <td>20</td>
      <td>True</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.813400</td>
      <td>0.807676</td>
      <td>0.810606</td>
      <td>0.808923</td>
      <td>0.809648</td>
      <td>0.810051</td>
      <td>0.001929</td>
      <td>40</td>
      <td>0.892190</td>
      <td>0.893706</td>
      <td>0.891877</td>
      <td>0.893834</td>
      <td>0.894175</td>
      <td>0.893157</td>
      <td>0.000935</td>
    </tr>
    <tr>
      <th>41</th>
      <td>75.386482</td>
      <td>0.686652</td>
      <td>1.048126</td>
      <td>0.041346</td>
      <td>200</td>
      <td>5</td>
      <td>1</td>
      <td>auto</td>
      <td>None</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 5, ...</td>
      <td>0.815672</td>
      <td>0.807844</td>
      <td>0.807744</td>
      <td>0.810606</td>
      <td>0.807964</td>
      <td>0.809966</td>
      <td>0.003047</td>
      <td>41</td>
      <td>0.999684</td>
      <td>0.999726</td>
      <td>0.999790</td>
      <td>0.999811</td>
      <td>0.999642</td>
      <td>0.999731</td>
      <td>0.000063</td>
    </tr>
    <tr>
      <th>13</th>
      <td>230.118033</td>
      <td>9.642856</td>
      <td>4.405551</td>
      <td>0.257905</td>
      <td>1000</td>
      <td>10</td>
      <td>4</td>
      <td>sqrt</td>
      <td>20</td>
      <td>True</td>
      <td>{'n_estimators': 1000, 'min_samples_split': 10...</td>
      <td>0.812137</td>
      <td>0.807592</td>
      <td>0.811279</td>
      <td>0.807828</td>
      <td>0.809816</td>
      <td>0.809731</td>
      <td>0.001811</td>
      <td>42</td>
      <td>0.887982</td>
      <td>0.888887</td>
      <td>0.889057</td>
      <td>0.889141</td>
      <td>0.890303</td>
      <td>0.889074</td>
      <td>0.000741</td>
    </tr>
    <tr>
      <th>14</th>
      <td>149.722669</td>
      <td>1.748836</td>
      <td>2.303158</td>
      <td>0.126286</td>
      <td>400</td>
      <td>2</td>
      <td>1</td>
      <td>sqrt</td>
      <td>90</td>
      <td>False</td>
      <td>{'n_estimators': 400, 'min_samples_split': 2, ...</td>
      <td>0.811380</td>
      <td>0.803552</td>
      <td>0.804545</td>
      <td>0.806650</td>
      <td>0.804176</td>
      <td>0.806061</td>
      <td>0.002856</td>
      <td>43</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>331.014376</td>
      <td>8.354716</td>
      <td>6.730024</td>
      <td>0.607305</td>
      <td>800</td>
      <td>2</td>
      <td>1</td>
      <td>sqrt</td>
      <td>70</td>
      <td>False</td>
      <td>{'n_estimators': 800, 'min_samples_split': 2, ...</td>
      <td>0.812137</td>
      <td>0.802289</td>
      <td>0.804125</td>
      <td>0.806481</td>
      <td>0.802576</td>
      <td>0.805522</td>
      <td>0.003627</td>
      <td>44</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>108.119079</td>
      <td>2.513103</td>
      <td>1.596727</td>
      <td>0.084284</td>
      <td>200</td>
      <td>2</td>
      <td>1</td>
      <td>sqrt</td>
      <td>None</td>
      <td>False</td>
      <td>{'n_estimators': 200, 'min_samples_split': 2, ...</td>
      <td>0.810875</td>
      <td>0.804141</td>
      <td>0.803704</td>
      <td>0.805303</td>
      <td>0.803418</td>
      <td>0.805488</td>
      <td>0.002769</td>
      <td>45</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>180.368051</td>
      <td>0.607344</td>
      <td>2.322978</td>
      <td>0.032859</td>
      <td>800</td>
      <td>10</td>
      <td>1</td>
      <td>auto</td>
      <td>10</td>
      <td>False</td>
      <td>{'n_estimators': 800, 'min_samples_split': 10,...</td>
      <td>0.766434</td>
      <td>0.760290</td>
      <td>0.767593</td>
      <td>0.760690</td>
      <td>0.772942</td>
      <td>0.765589</td>
      <td>0.004709</td>
      <td>46</td>
      <td>0.784697</td>
      <td>0.785707</td>
      <td>0.787016</td>
      <td>0.783880</td>
      <td>0.785762</td>
      <td>0.785412</td>
      <td>0.001062</td>
    </tr>
    <tr>
      <th>15</th>
      <td>460.886029</td>
      <td>4.623063</td>
      <td>6.019484</td>
      <td>0.256234</td>
      <td>2000</td>
      <td>10</td>
      <td>4</td>
      <td>sqrt</td>
      <td>10</td>
      <td>False</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 10...</td>
      <td>0.765171</td>
      <td>0.759700</td>
      <td>0.767424</td>
      <td>0.761616</td>
      <td>0.771931</td>
      <td>0.765168</td>
      <td>0.004321</td>
      <td>47</td>
      <td>0.782550</td>
      <td>0.783729</td>
      <td>0.784870</td>
      <td>0.782681</td>
      <td>0.783805</td>
      <td>0.783527</td>
      <td>0.000847</td>
    </tr>
    <tr>
      <th>34</th>
      <td>159.322675</td>
      <td>1.912035</td>
      <td>2.860052</td>
      <td>0.045670</td>
      <td>1000</td>
      <td>5</td>
      <td>1</td>
      <td>auto</td>
      <td>10</td>
      <td>True</td>
      <td>{'n_estimators': 1000, 'min_samples_split': 5,...</td>
      <td>0.762815</td>
      <td>0.759869</td>
      <td>0.766751</td>
      <td>0.759764</td>
      <td>0.770163</td>
      <td>0.763872</td>
      <td>0.004047</td>
      <td>48</td>
      <td>0.781224</td>
      <td>0.783371</td>
      <td>0.784638</td>
      <td>0.781019</td>
      <td>0.783406</td>
      <td>0.782731</td>
      <td>0.001393</td>
    </tr>
    <tr>
      <th>47</th>
      <td>32.870517</td>
      <td>0.695827</td>
      <td>0.461440</td>
      <td>0.039711</td>
      <td>200</td>
      <td>5</td>
      <td>4</td>
      <td>auto</td>
      <td>10</td>
      <td>True</td>
      <td>{'n_estimators': 200, 'min_samples_split': 5, ...</td>
      <td>0.763235</td>
      <td>0.758859</td>
      <td>0.764394</td>
      <td>0.760606</td>
      <td>0.769406</td>
      <td>0.763300</td>
      <td>0.003618</td>
      <td>49</td>
      <td>0.778425</td>
      <td>0.780130</td>
      <td>0.780703</td>
      <td>0.778914</td>
      <td>0.780754</td>
      <td>0.779785</td>
      <td>0.000949</td>
    </tr>
    <tr>
      <th>0</th>
      <td>427.974828</td>
      <td>2.331909</td>
      <td>8.903878</td>
      <td>1.101831</td>
      <td>2000</td>
      <td>10</td>
      <td>2</td>
      <td>sqrt</td>
      <td>10</td>
      <td>True</td>
      <td>{'n_estimators': 2000, 'min_samples_split': 10...</td>
      <td>0.762730</td>
      <td>0.757596</td>
      <td>0.765909</td>
      <td>0.758754</td>
      <td>0.769995</td>
      <td>0.762997</td>
      <td>0.004576</td>
      <td>50</td>
      <td>0.779036</td>
      <td>0.779835</td>
      <td>0.781271</td>
      <td>0.779566</td>
      <td>0.780312</td>
      <td>0.780004</td>
      <td>0.000756</td>
    </tr>
  </tbody>
</table>
</div>



# Final Model Plus Analysis


```python
best_model = RandomForestClassifier(max_depth=None, n_estimators=320, n_jobs=-1, random_state=42)
score = cross_val_score(best_model, X_test1, y_train_target, scoring='accuracy', cv=5)
```


```python
score.mean()
```




    0.8098149237465961




```python
best_model.fit(X_test1, y_train_target)
```


```python
import matplotlib.pyplot as plt

figsize = (5,20)

name = 'Random Forest Classifier'
importances = pd.Series(best_model.feature_importances_, X_test1.columns)
title = f'{name}, max_depth={best_model.max_depth}'

plt.figure(figsize=figsize)
importances.sort_values().plot.barh(color='grey', title=title)


```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2136a358>




![png](output_25_1.png)



```python
conda install -c conda-forge eli5
```

    Collecting package metadata: done
    Solving environment: \ 
    Warning: 2 possible package resolutions (only showing differing packages):
      - anaconda::ca-certificates-2019.1.23-0
      - defaults::ca-certificates-2019.1.23done
    
    ## Package Plan ##
    
      environment location: /Users/lambda_school_loaner_95/anaconda3
    
      added / updated specs:
        - eli5
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        cairo-1.14.12              |    h9d4d9ac_1005         1.3 MB  conda-forge
        certifi-2019.3.9           |           py37_0         149 KB  conda-forge
        conda-4.6.14               |           py37_0         2.1 MB  conda-forge
        eli5-0.8.1                 |             py_0          65 KB  conda-forge
        fontconfig-2.13.1          |    h1027ab8_1000         269 KB  conda-forge
        fribidi-1.0.5              |    h1de35cc_1000          62 KB  conda-forge
        graphite2-1.3.13           |    h2098e52_1000          84 KB  conda-forge
        graphviz-2.40.1            |       hefbbd9a_2         6.7 MB
        harfbuzz-1.9.0             |    h9889186_1001         769 KB  conda-forge
        openssl-1.1.1b             |       h1de35cc_1         3.5 MB  conda-forge
        pango-1.42.4               |       h060686c_0         523 KB
        pixman-0.34.0              |    h1de35cc_1003         597 KB  conda-forge
        python-graphviz-0.10.1     |             py_0          17 KB  conda-forge
        tabulate-0.8.3             |             py_0          23 KB  conda-forge
        typing-3.6.4               |           py37_0          45 KB
        ------------------------------------------------------------
                                               Total:        16.1 MB
    
    The following NEW packages will be INSTALLED:
    
      cairo              conda-forge/osx-64::cairo-1.14.12-h9d4d9ac_1005
      eli5               conda-forge/noarch::eli5-0.8.1-py_0
      fontconfig         conda-forge/osx-64::fontconfig-2.13.1-h1027ab8_1000
      fribidi            conda-forge/osx-64::fribidi-1.0.5-h1de35cc_1000
      graphite2          conda-forge/osx-64::graphite2-1.3.13-h2098e52_1000
      graphviz           pkgs/main/osx-64::graphviz-2.40.1-hefbbd9a_2
      harfbuzz           conda-forge/osx-64::harfbuzz-1.9.0-h9889186_1001
      pango              pkgs/main/osx-64::pango-1.42.4-h060686c_0
      pixman             conda-forge/osx-64::pixman-0.34.0-h1de35cc_1003
      python-graphviz    conda-forge/noarch::python-graphviz-0.10.1-py_0
      tabulate           conda-forge/noarch::tabulate-0.8.3-py_0
      typing             pkgs/main/osx-64::typing-3.6.4-py37_0
    
    The following packages will be SUPERSEDED by a higher-priority channel:
    
      certifi                                          anaconda --> conda-forge
      conda                                            anaconda --> conda-forge
      openssl                                          anaconda --> conda-forge
    
    
    
    Downloading and Extracting Packages
    openssl-1.1.1b       | 3.5 MB    | ##################################### | 100% 
    fribidi-1.0.5        | 62 KB     | ##################################### | 100% 
    certifi-2019.3.9     | 149 KB    | ##################################### | 100% 
    tabulate-0.8.3       | 23 KB     | ##################################### | 100% 
    conda-4.6.14         | 2.1 MB    | ##################################### | 100% 
    python-graphviz-0.10 | 17 KB     | ##################################### | 100% 
    pixman-0.34.0        | 597 KB    | ##################################### | 100% 
    eli5-0.8.1           | 65 KB     | ##################################### | 100% 
    typing-3.6.4         | 45 KB     | ##################################### | 100% 
    harfbuzz-1.9.0       | 769 KB    | ##################################### | 100% 
    cairo-1.14.12        | 1.3 MB    | ##################################### | 100% 
    graphite2-1.3.13     | 84 KB     | ##################################### | 100% 
    graphviz-2.40.1      | 6.7 MB    | ##################################### | 100% 
    pango-1.42.4         | 523 KB    | ##################################### | 100% 
    fontconfig-2.13.1    | 269 KB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    
    Note: you may need to restart the kernel to use updated packages.



```python
import eli5
from eli5.sklearn import PermutationImportance

permuter = PermutationImportance(best_model, scoring='accuracy', cv='prefit',
                                n_iter=2, random_state=42)

permuter.fit(X_test1.values, y_train_target)
```




    PermutationImportance(cv='prefit',
               estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=320, n_jobs=-1,
                oob_score=False, random_state=42, verbose=0, warm_start=False),
               n_iter=2, random_state=42, refit=True, scoring='accuracy')




```python
feature_names = X_test1.columns.tolist()
eli5.show_weights(permuter, top=None, feature_names=feature_names)
```





    <style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
    }
</style>



    

    

    

    

    

    


    

    

    

    

    

    


    

    

    

    

    
        <table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>
    
        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1608
                
                    &plusmn; 0.0011
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                quantity
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 91.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0508
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                extraction_type
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 91.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0480
                
                    &plusmn; 0.0005
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                waterpoint_type
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 92.99%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0360
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                years_service
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 93.09%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0352
                
                    &plusmn; 0.0009
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                amount_tsh
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.52%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0132
                
                    &plusmn; 0.0007
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                population
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0121
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                id
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.12%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0101
                
                    &plusmn; 0.0008
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                payment_type
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.31%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0091
                
                    &plusmn; 0.0003
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                funder
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.69%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0074
                
                    &plusmn; 0.0003
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                offset_days
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0067
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                gps_height
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0060
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                management
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.06%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0058
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                source
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.39%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0044
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                latitude
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.49%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0040
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                installer
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.50%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0040
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                radial_r
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.54%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0038
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                longitude
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0034
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot60Y
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.66%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0034
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot45Y
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.70%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0032
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot60X
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.76%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0030
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot30X
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0030
                
                    &plusmn; 0.0004
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot30Y
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0027
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                rot45X
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.96%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0023
                
                    &plusmn; 0.0003
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                lga
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0020
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                quality_group
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.11%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0019
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                scheme_management
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.18%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0017
                
                    &plusmn; 0.0002
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                month_recorded
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.61%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0006
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                region
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.81%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0002
                
                    &plusmn; 0.0001
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                district_code
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.85%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0001
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                basin
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.91%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0001
                
                    &plusmn; 0.0000
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                permit
            </td>
        </tr>
    
    
    </tbody>
</table>
    

    


    

    

    

    

    

    







# Realizations
I hope to be doing further exploration on this dataset.  Biggest take aways so far
- Having a clear process is very helpful to make quick incrememntal gains
- Going for an over fitted model early on was helpful mindset to have
- Domain knowledge is best for efficient feature engineering
- Goal should be to plateau on score as fast as possible, then analyze the data to form real conclusions (the extra .0001 percent is not worth the time)
- don't lose track of the problem at hand when trying to make 'the best' model
- Stacking models together can be very valuable
- simple is good

# Submission blocks


```python
# estimator = RandomForestClassifier(max_depth=None, n_estimators=320, n_jobs=-1, random_state=42)
# estimator.fit(X_test1, y_train_target)

# y_pred = estimator.predict(X_teste2)

# sample_submission = pd.read_csv('https://storage.googleapis.com/kaggle-competitions-data/kaggle/14688/453539/sample_submission.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558636843&Signature=F8VvtSyMe9kPksSP5FK2hKnYJw6XMBytCjBIg%2Fkec7Ddcf%2Bue4Ge%2FGxHKWkr%2FZBgR6i%2FZt36WV4cSDnD2XCSJJ4%2F%2Bk23YsN9fi7F29W4E6worpqUlfCTk%2FNNB3y96EIkTZ7YNzh7inKZ93tB%2BxcAkyKnOseWQ3y8iGRDPRU7%2BXPeYretRM%2FBLqSbYU4gRUGxhSCwww3cRbsWi%2FRMcWq5YyypMOCuCBI7hJ7HIUc47u2WAjsleNEKGvJ69I82aral8%2FzercReL2rGCKfTTPsJ0WRH%2FTdJjHQ2O6EJ4l9AfUe7Dqjqjj8XCYFKaFfec5B8htoxTgkBcqG4GKhWEU%2B65w%3D%3D')
# baseline_submission13 = sample_submission.copy()
# baseline_submission13['status_group'] = y_pred
# baseline_submission13.to_csv('baseline_submission13.csv', index=False)
```


```python
#best_estimator = searchRF.best_estimator_
# model.fit(X_test1, y_train_target)

# y_pred = model.predict(X_teste2)


```


```python
# sample_submission = pd.read_csv('https://storage.googleapis.com/kaggle-competitions-data/kaggle/14688/453539/sample_submission.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558905720&Signature=INefpYzy5J4FBdIujvu9QVAhAMmJ0ahuW3bootCsHUaUuuKoJzgyA5rAsknVCOFY8hSWDQrZ%2FgBkCgnLniVB6rHKeMNEl4HAYOe0RN54uY0bvqmYKWr3aARaQk7YCKka28S5WtjI7HkjM%2BbXZkP4dYlt8NRjAoVSdiaWsysUwxXeRRGIBuhrt%2FuBSAPk3CegyHXikAsEHx0VKhdRNIv%2BrHyXOu2Y1ldloSvxcrQdmI%2BirQrxXJLT7XD6tI5leaYbXit6tQ2X5syfBek31o31B7AkMqz%2B9zIlZ5SBDMhOXbdaxEZMGaE7NRFEioLq%2Fp3GtolWe97qrOnvoHybd2PBsg%3D%3D')
# baseline_submission16 = sample_submission.copy()
# baseline_submission16['status_group'] = y_pred
# baseline_submission16.to_csv('baseline_submission16.csv', index=False)
```


```python
#Filenames of your submissions you want to ensemble
# sample_submission = pd.read_csv('https://storage.googleapis.com/kaggle-competitions-data/kaggle/14688/453539/sample_submission.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558636843&Signature=F8VvtSyMe9kPksSP5FK2hKnYJw6XMBytCjBIg%2Fkec7Ddcf%2Bue4Ge%2FGxHKWkr%2FZBgR6i%2FZt36WV4cSDnD2XCSJJ4%2F%2Bk23YsN9fi7F29W4E6worpqUlfCTk%2FNNB3y96EIkTZ7YNzh7inKZ93tB%2BxcAkyKnOseWQ3y8iGRDPRU7%2BXPeYretRM%2FBLqSbYU4gRUGxhSCwww3cRbsWi%2FRMcWq5YyypMOCuCBI7hJ7HIUc47u2WAjsleNEKGvJ69I82aral8%2FzercReL2rGCKfTTPsJ0WRH%2FTdJjHQ2O6EJ4l9AfUe7Dqjqjj8XCYFKaFfec5B8htoxTgkBcqG4GKhWEU%2B65w%3D%3D')
# files = ['baseline_submission15.csv', 'baseline_submission14.csv', 'baseline_submission13.csv',
#         'baseline_submission5.csv', 'baseline_submission4.csv', 'baseline_submission16.csv']

# submissions = (pd.read_csv(file)[['status_group']] for file in files)
# ensemble = pd.concat(submissions, axis='columns')
# majority_vote = ensemble.mode(axis='columns')[0]


# sample_submission = pd.read_csv('https://storage.googleapis.com/kaggle-competitions-data/kaggle/14688/453539/sample_submission.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1558905720&Signature=INefpYzy5J4FBdIujvu9QVAhAMmJ0ahuW3bootCsHUaUuuKoJzgyA5rAsknVCOFY8hSWDQrZ%2FgBkCgnLniVB6rHKeMNEl4HAYOe0RN54uY0bvqmYKWr3aARaQk7YCKka28S5WtjI7HkjM%2BbXZkP4dYlt8NRjAoVSdiaWsysUwxXeRRGIBuhrt%2FuBSAPk3CegyHXikAsEHx0VKhdRNIv%2BrHyXOu2Y1ldloSvxcrQdmI%2BirQrxXJLT7XD6tI5leaYbXit6tQ2X5syfBek31o31B7AkMqz%2B9zIlZ5SBDMhOXbdaxEZMGaE7NRFEioLq%2Fp3GtolWe97qrOnvoHybd2PBsg%3D%3D')
# submission = sample_submission.copy()
# submission['status_group'] = majority_vote
# submission.to_csv('ensemble-submission2.csv', index=False)
```
