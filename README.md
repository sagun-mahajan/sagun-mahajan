CREDIT CARD FRAUD DETECTION
In this kernel, we are going to predict whether a credit card transaction is fraud or not using Machine Learning.

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

Outline:
Set Up
Explore and Prepare Data
Splitting Data
outliers
Undersampling
Train Models
ROC Curves
Fine Tuning XGBoost
Evaluate on test data
Summary
OUR GOALS:
Understand the imbalanced dataset and will perform various approaches like undersampling/ oversampling, Choosing right metrics of ROC- AUC to deal with imbalanced dataset.
After trying different approaches and training different models, we will compare their results and decides the one which fits best for our application.
Set Up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import warnings      # ignore warnings
warnings.filterwarnings('ignore')

Explore and Prepare data
Train_test_split: we'll split our dataset into train, test set and Validation set.
Scaling: Perform Feature Scaling on Amount and Time Columns as the rest columns has already been scaled, so we do need to scale them again
Class imbalance: Next, we'll check for class imbalance and perform undersampling/oversampling to balance our class (50:50 ratio).

Outliers
We cannot remove every outliers, otherwise we'll loose some important data. so, we'll try to remove few of them from those features which are highly correlated with our labels.

Random undersampling
We neither Removed outliers nor performed random undersamping on Validation set as well as test set because we want to check our model performance on natural dataset. So, class imbalanced and outliers still exist in Validation set as well as test set.
Also there is huge difference in minority and majority Class, we might face the problem of loosing our information by performing Undersampling on our data.

 Ensemble algorithms do not have 'Decision Function method', so we'll be using positive Class probabilities as y_scores to plot ROC curves. Also we are cross validating on validation set so that we can have an exact estimate of our model perfomance on test set.

 Summary:
 Never perform Cross validation on Oversampling data otherwise your model would tend to overfit. Moreover, if you undersample a large majority Class into fewer samples, you might loose information of data.
There is always going to be a trade off between TPR, FPR as well as Recall/Precision, Choose wisely based on your need.
Also i believe that, if you fine tune the Rest Classifiers, and used them under Voting Classifier to predict, you might get 100 % roc_auc_score.
