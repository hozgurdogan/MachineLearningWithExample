import joblib
import pandas as pd

from daiabetes_pipleine import diabetes_data_prep





df = pd.read_csv("../datasets/diabetes.csv")



X,y = diabetes_data_prep(df)


random_user = X.sample(1,random_state=50)

new_model = joblib.load("voting_clf.pkl")

print(new_model.predict(random_user))