---
layout: post
title: Should we fix the wells?
subtitle: Predictive Model Exploration with Tanzania Water Wells
image: /img/WaterHole008A.jpg
published: True
---

I participated in a competition using a dataset describing water wells in Tanzania.  The goal was to build the best predictive model for classifying the water wells in Tanzania.  There were three classes of wells, functioning, non-functioning, and functioning but needs repairs.  Using some machine learning techniques, and data analysis, this post will go through my process of fine tuning a model and thought process to help improve a real world problem.


**Quick Baseline**

The first thing I look for when I train a model is a quick dumb baseline. In other words, what accuracy should I be able to obtain if I predicted that every well would fit into the most likely case.  Essentially guess c on every answer on a test.  




    functional                 0.543081
    non functional             0.384242
    functional needs repair    0.072677
    Name: status_group, dtype: float64


It is simple enough to find this by counting the amount of each well from past data.  In this case the most likely scenario is that a well would be functioning. One can see from the chart above that the baseline is 54%.  If my model can be over 54% accurate, then I know I am heading in the right direction.

**Note**: I try my best to be cognizant of the actual problem at hand.  You can get 54% accuracy by guessing all wells will be functional, theoretically.  What is most useful though is if you can determine which wells are functional and just need to be repaired.  So guessing all wells will be functional will get you just the bare baseline, but it would not be useful at all.

# Beat the baseline

To find the baseline that I needed to imporve upon, I quickly counted the values for each category of wells.

The results:

    functional                 0.543081
    non functional             0.384242
    functional needs repair    0.072677

I discovered by normalizing the counts that 54% accuracy was the baseline I had to beat.  Using the sklearn python library functionality, I created a quick Logistic Regression model to see if I could do better.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

basic_model = LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42)
cross_val_score(basic_model, X_train_numbers, y_train_target, scoring='accuracy', cv=5).mean()
```

      accuracy: 0.551801290559174


The logistic regression did not do much better.  But it was a step in the right direction.  I only used numerical features that were already pretty clean. After cleaning and engineering new features, coupled with more advanced models, the results proved to be much more promising.

# Clean data, feature engineer, test different models

After spending more time looking and wrangling with the data, I created a function that would clean, impute and scale the data so that it would play nice with more advanced models.

Data Cleaning/ Feature Engineering Code:

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

After wrangling the data, I created a pipeline to test out different models and compare their performance.
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

    LogisticRegression Cross-Validated Accuracy: 0.6063973696515584 
     
    
    DecisionTreeClassifier Cross-Validated Accuracy: 0.6934514989614184 
     
    
    DecisionTreeClassifier Cross-Validated Accuracy: 0.7466330313376597 
     
    RandomForestClassifier1 Cross-Validated Accuracy: 0.6887042027413878 
    
    RandomForestClassifier2 Cross-Validated Accuracy: 0.8069023510130691 
      
    XGBClassifier Cross-Validated Accuracy: 0.7344784863650176 
     
    
The second random forest model I tested ended up spitting out the highest accuracy.  Although hyper tuning the XG Boost model might have also given better results, I decided to stick with the Random Forest model and further tune it.

# Final Model Plus Analysis


After hyper tuning my random forest model, I ended up getting about an 80% accuracy.  I knew to further increase the accuracy I could further hyper tune, add more features, and bag my best models together.  After doing all of those steps, my highest accuracy ended up being 82.5%.  While I was proud of the results, I knew the more important part of my work was in the analysis of the information.  

It was useful to take the model that worked best and determine the most important features that helped determine the score. With random forest it can be tricky to explain how the algorithm is making decisions.  With the eli5 library though, I could determine the permutation importance for all the features.

Permutation Importance:

(import picture of permutation importance here)

Further, the most important wells to categorize were the functional wells that needed repair.  Using a confusion matrix, I was able to see how many of those wells I was able to correctly classify.

(insert confusion matrix)

Even better, from the correctly classified wells, I was able to determine the ones that my model determined had the highest probablity of being correctly classified.  With that information, aid can priortize fixing wells that have a chance of being restored, as opposed to wasting time checking all the wells, without knowing if they are functional or not to begin with.  

# Realizations
I learned a lot with this project.  Understanding how to manipulate data to aid a model in categorizing information is important.  It is equally important to understand the model and determine how to best use the information to solve the problem at hand.  



```
