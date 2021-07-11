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


def lasso_cv_plus_model_selection(X0,y0,k,group_labels):
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

#     alphas = np.linspace(0, 0.02, 11)
    alphas1 = np.linspace(0, 0.2, 20)
    alphas2 = np.linspace(0.2, 0.5, 10)[1:]
    alphas=np.concatenate((alphas1,alphas2))
#     alphas = np.logspace(-4, -0.5, 30)
    lasso_cv = linear_model.LassoCV(alphas=alphas, random_state=0, max_iter=10000)
    
    X,y=X0.values,y0.values
    
#     scores=np.zeros(k,)
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        lasso_cv.fit(X_train, y_train)  
        scores.append(lasso_cv.score(X_test, y_test))   
#         print(lasso_cv.alpha_)
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(clf, X0, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return np.array(scores), scores_rand


# def MLP_cv(X,y,k,group_labels):
#     from sklearn.neural_network import MLPRegressor

#     n_j=-1
# #     hidden_layer_sizes=100,
# #     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(random_state=1,hidden_layer_sizes = (100), max_iter=10000,activation='tanh',early_stopping=True)

#     split_obj=GroupKFold(n_splits=k)    
#     # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
#     # Perform k-fold cross validation on the shuffled vector of lm GE across samples
#     # y.sample(frac = 1) this just shuffles the vector
#     scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
#     return scores, scores_rand
 # X is train samples and y is the corresponding labels

def MLP_cv(X,y,k,group_labels):
    from sklearn.neural_network import MLPRegressor

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
    regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
                        alpha=0.01,early_stopping=True)

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
    scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return scores, scores_rand


def MLP_cv_plus_model_selection(X0,y0,k,group_labels):
    from sklearn.neural_network import MLPRegressor

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
#                         alpha=0.01,early_stopping=True)

    mlp_gs = MLPRegressor()

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(50,),(200,),(500,),(10,30,10),(50,10),(50,10,10)],
        'activation': ['tanh', 'relu','logistic'],
        'alpha': [0.0001, 0.05,0.01,0.1,0.2],
        'early_stopping':[True,False]
    }
    
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)

    X,y=X0.values,y0.values
    
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
#         lasso_cv.fit(X_train, y_train)  
        clf.fit(X, y)
        scores.append(clf.score(X_test, y_test))   
        print(clf.best_params_)

    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
#     scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    scores_rand=0
    return scores, scores_rand








