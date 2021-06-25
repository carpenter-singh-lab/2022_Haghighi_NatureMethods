# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# score(X, y, sample_weight=None)[source]
# Return the coefficient of determination R^2 of the prediction.

# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold,LeaveOneGroupOut
from sklearn import metrics
import numpy as np


def lasso_cv(X,y,k,group_labels):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, 1 (feature value)]
    from sklearn import linear_model
    n_j=3
    # build sklearn model
    clf = linear_model.Lasso(alpha=0.1,max_iter=10000)

#     k=np.unique(group_labels).shape[0]
    split_obj=GroupKFold(n_splits=k)
#     split_obj = LeaveOneGroupOut()    
    # Perform k-fold cross validation
    scores = cross_val_score(clf, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
    
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(clf, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return scores, scores_rand


def MLP_cv(X,y,k,group_labels):
    from sklearn.neural_network import MLPRegressor

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
    regr = MLPRegressor(random_state=1,hidden_layer_sizes = (100), max_iter=10000,activation='tanh')

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
    scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return scores, scores_rand
