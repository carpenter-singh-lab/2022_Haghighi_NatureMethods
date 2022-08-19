from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold,LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
from warnings import simplefilter
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# simplefilter("ignore", category=ConvergenceWarning)
# from sklearn.exceptions import ConvergenceWarning
# ConvergenceWarning('ignore')


########################## Lasso models
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


def lasso_cv_plus_model_selection(X0,y0,k,group_labels,rand_added_flag):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, 1 (feature value)]
    from sklearn import linear_model
    n_j=3
    # build sklearn model
    clf = linear_model.Lasso(alpha=0.1,max_iter=1000)

#     k=np.unique(group_labels).shape[0]
    split_obj=GroupKFold(n_splits=k)
#     split_obj = LeaveOneGroupOut()    
    # Perform k-fold cross validation

#     alphas = np.linspace(0, 0.02, 11)
    alphas1 = np.linspace(0, 0.2, 20)
    alphas2 = np.linspace(0.2, 0.5, 10)[1:]
    alphas=np.concatenate((alphas1,alphas2))
#     alphas = np.logspace(-4, -0.5, 30)
    lasso_cv = linear_model.LassoCV(alphas=alphas, random_state=0, max_iter=1000,selection='random',n_jobs=k)
#     lasso_cv = linear_model.LassoLarsCV(cv=5)
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
    if rand_added_flag:
        scores_rand = cross_val_score(clf, X0, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0
    return np.array(scores), scores_rand


def ridge_cv_plus_model_selection(X0,y0,k,group_labels,rand_added_flag):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, 1 (feature value)]
    from sklearn import linear_model
    n_j=3
    # build sklearn model
    clf = linear_model.Ridge(alpha=0.1,max_iter=10000)

#     k=np.unique(group_labels).shape[0]
    split_obj=GroupKFold(n_splits=k)
#     split_obj = LeaveOneGroupOut()    
    # Perform k-fold cross validation

#     alphas = np.linspace(0, 0.02, 11)
    alphas1 = np.linspace(0.1, 0.2, 10)
    alphas2 = np.linspace(0.2, 0.5, 10)[1:]
    alphas=np.concatenate((alphas1,alphas2))
#     alphas = np.logspace(-4, -0.5, 30)
    lasso_cv = linear_model.RidgeCV(alphas)
    
#     X,y=X0,y0
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
    if rand_added_flag:
        scores_rand = cross_val_score(clf, X0, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0
    return np.array(scores), scores_rand

########################## MLP
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

def MLP_cv(X,y,k,group_labels,rand_added_flag):
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
    
    if rand_added_flag:
        scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0    
    
    return scores, scores_rand



def MLP_cv_plus_model_selection(X0,y0,k,group_labels,rand_added_flag):
    

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
#                         alpha=0.01,early_stopping=True)

    mlp_gs = MLPRegressor(random_state=0,early_stopping=True,n_iter_no_change=4,learning_rate='adaptive')

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)
#     parameter_space = {
#         'hidden_layer_sizes': [(50,),(200,),(500,),(10,30,10),(50,10),(50,10,10)],
#         'activation': ['tanh', 'relu','logistic'],
#         'alpha': [0.0001, 0.05,0.01,0.1,0.2],
#         'early_stopping':[True,False]
#     }

    parameter_space = {
        'max_iter':[10,100,300,500],
        'hidden_layer_sizes': [(32,64),(64,32),(50,10),(50,10,10),(20,10),()], #(50,5),(50,),(10,)
        'activation': ['logistic','tanh'],
        'alpha': [0.0005,0.01,0.3,1,2,3,4,5,6,7],
#         'learning_rate': ['constant','adaptive']
#         'early_stopping':[True,False]
    }
    
    
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=k, cv=4)

    X,y=X0,y0.values
    
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)  
#         clf.fit(X, y)
        scores.append(clf.best_estimator_.score(X_test, y_test))   
        print(clf.best_params_)

    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
#     scores_rand=0

    if rand_added_flag:
        scores_rand = cross_val_score(mlp_gs, X, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0    
    return scores, scores_rand


def MLP_cv_plus_model_selection_keras(X0,y0,k,group_labels,rand_added_flag):
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, Flatten
    from sklearn.metrics import mean_squared_error,r2_score
    from keras.callbacks import EarlyStopping
    from keras import backend as K
    
    X = X0.reshape(X0.shape[0], X0.shape[1], 1)
    y=y0.values
#     model.summary()
    model = Sequential()
    model.add(Dense(16,activation="relu", input_shape=(X0.shape[1],1)))
#     model.add(Conv1D(32, 2, activation="relu", input_shape=(X0.shape[1],1)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")#,metrics=[coeff_determination])        
#     model.compile(loss=coeff_determination, optimizer="adam")#,metrics=[coeff_determination])   

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    
    Wsave = model.get_weights()

    split_obj=GroupKFold(n_splits=k)    

    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        XTraining, XValidation, YTraining, YValidation = train_test_split(X_train,y_train,test_size=0.1)

        model.set_weights(Wsave)
        model.fit(XTraining,YTraining, batch_size=XTraining.shape[0],epochs=1000,
                  validation_data=(XValidation,YValidation),callbacks=[es],verbose=0)            
        ypred = model.predict(X_test)
        
        scores.append(r2_score(y_test, ypred))

    return scores, 0




def SVR_cv_plus_model_selection(X0,y0,k,group_labels,rand_added_flag):
    

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
#                         alpha=0.01,early_stopping=True)

    svr_gs = SVR(epsilon=0.2)

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)
#     parameter_space = {
#         'hidden_layer_sizes': [(50,),(200,),(500,),(10,30,10),(50,10),(50,10,10)],
#         'activation': ['tanh', 'relu','logistic'],
#         'alpha': [0.0001, 0.05,0.01,0.1,0.2],
#         'early_stopping':[True,False]
#     }

    parameter_space = {
        'kernel':('poly', 'rbf', 'sigmoid'),
        'C': [1,2,3,5,20,100,500,1000], #(50,5),(50,),(10,)
        'degree': [1,2,3,4],
        'coef0': [0.01,0.5,1,10],
        'gamma': ('auto','scale'),
#         'epsilon':[0.1,0.2,0.5,0.3]
#         'early_stopping':[True,False]
    }
    
    
    clf = GridSearchCV(svr_gs, parameter_space, n_jobs=k, cv=4)

    X,y=X0,y0.values
    
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)  
#         clf.fit(X, y)
        scores.append(clf.best_estimator_.score(X_test, y_test))   
        print(clf.best_params_)

    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
#     scores_rand=0

    if rand_added_flag:
        scores_rand = cross_val_score(svr_gs, X, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0    
    return scores, scores_rand

def MLP_cv_plus_model_selection_rand_test(X0,y0,k,group_labels,rand_added_flag):
    

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
#                         alpha=0.01,early_stopping=True)

    mlp_gs = MLPRegressor(random_state=0,early_stopping=True,n_iter_no_change=20)

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)
#     parameter_space = {
#         'hidden_layer_sizes': [(50,),(200,),(500,),(10,30,10),(50,10),(50,10,10)],
#         'activation': ['tanh', 'relu','logistic'],
#         'alpha': [0.0001, 0.05,0.01,0.1,0.2],
#         'early_stopping':[True,False]
#     }

    parameter_space = {
        'max_iter':[10,100,300,500],
        'hidden_layer_sizes': [(32,64),(64,32),(50,10),(50,10,10),(20,10)], #(50,5),(50,),(10,)
        'activation': ['logistic','tanh'],
        'alpha': [0.0005,0.01,0.3,1,2],
#         'learning_rate': ['constant','adaptive']
#         'early_stopping':[True,False]
    }
    
    
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=k, cv=4)

    X,y=X0,y0.values
    
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)  
#         clf.fit(X, y)
        scores.append(clf.best_estimator_.score(X_test, y_test))   
        print(clf.best_params_)

    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
#     scores_rand=0

    if rand_added_flag:
        scores_rand = cross_val_score(mlp_gs, X, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0    
    return scores, scores_rand



def MLP_cv_plus_model_selection_taorf(X0,y0,k,group_labels,rand_added_flag):
    from sklearn.neural_network import MLPRegressor

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
#     regr = MLPRegressor(hidden_layer_sizes = (50,10),activation='logistic',\
#                         alpha=0.01,early_stopping=True)

    mlp_gs = MLPRegressor(random_state=0,max_iter=1000)

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)
#     parameter_space = {
#         'hidden_layer_sizes': [(50,),(200,),(500,),(10,30,10),(50,10),(50,10,10)],
#         'activation': ['tanh', 'relu','logistic'],
#         'alpha': [0.0001, 0.05,0.01,0.1,0.2],
#         'early_stopping':[True,False]
#     }

    parameter_space = {
        'hidden_layer_sizes': [(50,),(10,30,10),(50,10),(50,10,10)],
        'activation': ['tanh', 'relu','logistic'],
        'alpha': [0.0001, 0.05,0.01,0.2,0.5,0.7],
        'learning_rate':['constant', 'adaptive']
#         'early_stopping':[True,False]
    }
    
#         parameter_space = {
#         'hidden_layer_sizes': [(50,),(10,),(50,10),(50,10,10)],
#         'activation': ['tanh', 'relu','logistic'],
#         'alpha': [0.05,0.01,0.2,0.5],
# #         'early_stopping':[True,False]
#     }
    
    
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=6, cv=2)

    X,y=X0.values,y0.values
    
    scores=[]
    for train_index, test_index in split_obj.split(X, y, group_labels):
        
        
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)  
#         clf.fit(X, y)
        scores.append(clf.score(X_test, y_test))   
        print(clf.best_params_)

    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    
#     scores_rand=0

    if rand_added_flag:
        scores_rand = cross_val_score(mlp_gs, X, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    else:
        scores_rand =0    
    return scores, scores_rand


# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)


########################## Random Forest
def RFR_cv_plus_model_selection(X0,y0,k,group_labels,rand_added_flag):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    n_j=-1

#     parameter_space ={'bootstrap': [True, False],\
#      'max_depth': [10, 20, 40, 50, 100, None],\
#      'max_features': ['auto', 'sqrt'],\
#      'min_samples_leaf': [1, 2, 4],\
#      'min_samples_split': [2, 5, 10],\
#      'n_estimators': [200, 400, 600, 800, 1000]}

    parameter_space ={
     'max_depth': [10, 20, None],\
     'min_samples_leaf': [1,4],\
     'min_samples_split': [2, 5, 10]}
    

    rfr_gs = RandomForestRegressor(bootstrap=True,max_features='auto')

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
#     scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)

#     mlp_gs = MLPClassifier(max_iter=100)

    clf = GridSearchCV(rfr_gs, parameter_space, n_jobs=-1, cv=2)

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
    scores_rand = cross_val_score(rfr_gs, X0, y0.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
#     scores_rand=0
    return scores, scores_rand



############################## Feature Ranking #########################
def linear_model_feature_ranking(X0,y0,k,group_labels,l1k_features_gn):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, 1 (feature value)]
    from sklearn import linear_model
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_regression    
    
    n_j=3
    # build sklearn model
#     clf = linear_model.Lasso(alpha=0.1,max_iter=10000)
    clf = linear_model.LinearRegression()

#     k=np.unique(group_labels).shape[0]

    split_obj=GroupKFold(n_splits=k)
#     split_obj = LeaveOneGroupOut()    
    # Perform k-fold cross validation

#     alphas = np.linspace(0, 0.02, 11)
    alphas1 = np.linspace(0, 0.2, 20)
    alphas2 = np.linspace(0.2, 0.5, 10)[1:]
    alphas=np.concatenate((alphas1,alphas2))
#     alphas = np.logspace(-4, -0.5, 30)
#     lasso_cv = linear_model.LassoCV(alphas=alphas, random_state=0, max_iter=1000,selection='random')
    
    X,y=X0.values,y0.values
    
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    fs.fit(X, y) 
    
    
    clf.fit(X, y)  
    return clf.coef_,fs.scores_
#     return ranking(np.abs(lasso_cv.coef_), l1k_features_gn)


ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = preprocessing.MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

