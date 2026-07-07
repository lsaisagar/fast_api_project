import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


## Load dataset
df= pd.read_csv("model.csv")

##Features

X = df[["Age","Salary"]]

#Target

y = df["Approved"]
 
 
 #train model
 
model = RandomForestClassifier(random_state=42)

model.fit(X,y)

#Save model

joblib.dump(model,"loan_model.joblib")

print("model saved sucessfully")