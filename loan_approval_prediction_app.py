import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

#datapreprocessing stage
a = pd.read_csv(r"C:\Users\sumit\Downloads\train.csv")
b = pd.DataFrame(a)
h = b.copy()

c = LabelEncoder()
h["Gender"]=c.fit_transform(h["Gender"])
h["Married"]=c.fit_transform(h["Married"])
h["Education"]=c.fit_transform(h["Education"])
h["Employment_Status"]=c.fit_transform(h["Employment_Status"])
h["Loan_Status"]=c.fit_transform(h["Loan_Status"])

h = pd.get_dummies(h,columns=["Property_Area"],drop_first=False,dtype=int)

#model training stage start..
inputs = h[["Gender","Married","Dependents","Education","Age",
            "Property_Area_Rural","Property_Area_Semiurban","Property_Area_Urban",
            "Loan_Term","Credit_History","Employment_Status",
            "Applicant_Income","Coapplicant_Income","Loan_Amount"]]

output = h["Loan_Status"]

# ❗ FIXED train-test split
x_train , x_test , y_train , y_test = train_test_split(inputs,output, test_size=0.2 , train_size=0.8 , random_state=42)
get = RandomForestClassifier()
get.fit(x_train,y_train)

sames = get.predict(x_test)
print("accuracy score:" , accuracy_score(y_test,sames))
print(" precison score:" , precision_score(y_test,sames,average='macro'))
print("F1 score:" , f1_score(y_test,sames, average='macro'))
print("reacll score:" , recall_score(y_test,sames, average='macro'))




# --- prediction from user inputs ---
chop = int(input("enter your Gender :"))
pop = int(input("enter your married :"))
gop = int(input("enter your dependent :"))
nope = int(input("enter your education :"))
yop = int(input("enter your age :"))
mop = int(input("enter your  Property_Area_Rural:"))
dop = int(input("enter your  Property_Area_semiurban:"))
sop = int(input("enter your  Property_Area_urban:"))
fop = int(input("enter your  loan_term:"))
op = float(input("enter your credit_history:"))
zop = int(input("enter your  employment_status:"))
kip = float(input("enter you Applicant_Income"))
jop = int(input("enter your coapplicant_income::"))
xop = float(input("enter your Loan_Amount:"))

naw = get.predict([[chop,pop,gop,nope,yop,mop,dop,sop,fop,op,zop,kip,jop,xop]])

if naw == 1:
    print("loan not approve")
else:
    print("loan approved")
