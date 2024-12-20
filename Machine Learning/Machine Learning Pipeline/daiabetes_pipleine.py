import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier  # If you want to use CatBoost



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    '''
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    '''
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe,col_name,q1 = 0.25 , q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 +1.5 * interquantile_range
    low_limit = quartile1 -1.5 *interquantile_range
    return low_limit , up_limit
    
def check_outlier(dataframe, col_name,q1 = 0.25 , q3 = 0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1,q3)  # Değişken adı düzeltildi
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):  # Parantezler ve operatör düzeltildi
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit




def one_hot_encoder(dataframe, categorical_cols , drop_first=True):
    dataframe = pd.get_dummies(dataframe , columns=categorical_cols , drop_first=drop_first)
    return dataframe




def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])
    
    # Age kategorize etme
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI sınıflandırması
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], 
                                       bins=[-1, 18.5, 24.9, 29.9, 100], 
                                       labels=["underweight", "healty", "overweight", "obese"])
    
    # Blood Pressure sınıflandırması
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], 
                                            bins=[-1, 79, 89, 123], 
                                            labels=["normal", "hs1", "hs2"])

    dataframe.columns = [col.upper() for col in dataframe.columns]
    
    # Grab column names
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    # Summary for categorical columns
    '''
    for col in cat_cols: 
        cat_summary(dataframe, col)

    # Target summary with categorical variables
    for col in cat_cols: 
        target_summary_with_cat(dataframe, "OUTCOME", col)
'''
    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
    
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    #check_df(dataframe)

    replace_with_thresholds(dataframe, "INSULIN")

    for col in num_cols:
        print(col, check_outlier(dataframe, col, 0.05, 0.95))

    # Scaling numerical columns
    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)

    # X ve y dönüşleri
    X = dataframe.drop("OUTCOME", axis=1)
    y = dataframe["OUTCOME"]
    return X, y


def base_models(X, y, scoring="roc_auc"):
  """
  Farklı sınıflandırma algoritmalarının performansını karşılaştırır.

  Args:
    X: Bağımsız değişkenler
    y: Hedef değişken
    scoring: Skorlama metrik (varsayılan: roc_auc)
  """

  print("Base Models....")

  classifiers = [
    ('LR', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('SVC', SVC()),
    ('CART', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('GBM', GradientBoostingClassifier()),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('LightGBM', LGBMClassifier()),
    # ('CatBoost', CatBoostClassifier(verbose=False))
  ]

  for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
    print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")





knn_params = {'n_neighbors': range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {'max_depth': [8, 15, None],
            "max_features": [5, 7, "auto"],
            "min_samples_split": [15, 20],
            "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                 "max_depth": [5, 8],
                 "n_estimators": [100, 200],
                 "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                  "n_estimators": [300, 500],
                  "colsample_bytree": [0.7, 1]}






classifiers = [
    ('KNN', KNeighborsClassifier(), knn_params),
    ('CART', DecisionTreeClassifier(), cart_params),
    ('RF', RandomForestClassifier(), rf_params),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(), lightgbm_params)
]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
  """
  Farklı sınıflandırma algoritmaları için hiperparametre optimizasyonu yapar.

  Args:
    X: Bağımsız değişkenler
    y: Hedef değişken
    cv: Çapraz doğrulama kat sayısı
    scoring: Skorlama metrik (varsayılan: roc_auc)
  """

  print("Hyperparameter Optimization...")

  best_models = {}

  for name, classifier, params in classifiers:
    print(f"########## {name} ##########")

    cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

    gs = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
    print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
    print(f"{name} best params: {gs.best_params_}", end="\n\n")

    best_models[name] = final_model

  return best_models




def voting_classifier(best_models, X, y):
  """
  Birden fazla sınıflandırma modelini birleştiren bir oylama sınıflandırıcısı oluşturur.

  Args:
    best_models: Daha önce eğitilmiş en iyi modellerin bulunduğu sözlük
    X: Bağımsız değişkenler
    y: Hedef değişken
  """

  print("Voting Classifier...")

  voting_clf = VotingClassifier(estimators=[('KNN', best_models['KNN']), ('RF', best_models['RF']),
                                             ('LightGBM', best_models['LightGBM'])],
                                voting='soft').fit(X, y)

  cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])

  print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
  print(f"F1Score: {cv_results['test_f1'].mean()}")
  print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")

  return voting_clf













def main():
    import warnings
    warnings.filterwarnings('ignore')

    df = pd.read_csv("../datasets/diabetes.csv")
    X,y = diabetes_data_prep(df)
    base_models(X,y)
    best_models = hyperparameter_optimization(X,y)
    voitng_clf = voting_classifier(best_models,X,y)
    joblib.dump(voitng_clf,"voting_clf.pkl")
    return voitng_clf

if __name__ == "__main__":
   main()