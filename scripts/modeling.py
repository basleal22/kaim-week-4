from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
def models(data):
    rf=RandomForestRegressor(Random_state=42)
    param_grid={'n_estimators':[10,50,100],
                'max_depth':[None,'10','20']
                              
    }
    #grid search with cross-validation
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='accuracy')
    grid_search.fit()
    