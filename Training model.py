# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:43:33 2025

@author: ASUS
"""
#مشروع ال linear regression 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
#Read the data  
data =pd.read_csv(r"C:\Users\ASUS\Downloads\train data linear\Student_Performance.csv")
data=pd.DataFrame(data)
#EDA to data 
print(data.values)
print(data.info())
print(data.describe())
x =data.describe()
print(x)
print(data.describe(include="object"))
print(data.describe(exclude="int64"))
print(data.info())

#samples of data 
subdataset=data[["Performance Index","Hours Studied","Sleep Hours"]]
print(subdataset.sample(10))



cato_column= data[["Extracurricular Activities"]]
print(data.info())

reltation_bet_sleep_perf = data[["Sleep Hours","Performance Index"]]
sub_data_about_rel1 = data.loc[1:100,["Sleep Hours","Performance Index"]]
print(sub_data_about_rel1)

#Use histograme for two numerical columns 
viz = subdataset[["Sleep Hours","Performance Index"]]
viz.hist()
plt.show()
selection_of_sum_data_rel1 =data[(data["Sleep Hours"] ==9) &(data["Performance Index"] >85)]
print(len(selection_of_sum_data_rel1))
#106
#show linearity 
plt.scatter(subdataset["Sleep Hours"],subdataset["Performance Index"],color ="blue")
plt.xlabel("Hours")
plt.ylabel("performence")
plt.show()
#from scatter plot we show that if we use Sleep Hours as independent var it will not work 

reltation_bet_houStud_perf = data[["Hours Studied","Performance Index"]]
sub_data_about_rel2=data.loc[1:100,["Hours Studied","Performance Index"]]
print(sub_data_about_rel2)

selection_of_sum_data_rel2 =data[(data["Hours Studied"] == 9) & (data["Performance Index"] >85)]
print(len(selection_of_sum_data_rel2))
#202
plt.scatter(subdataset["Hours Studied"],subdataset["Performance Index"],color ="blue")
plt.xlabel("Hours studied")
plt.ylabel("performence")
plt.show()
#from scatter plot we show that if we use Hours Studied as independent var it will work perfictly 

subdataset=data[["Performance Index","Hours Studied","Sleep Hours"]]
print(subdataset.sample(10))



#prepare data 

print(data.isna().sum())
#the data was filled 

print(data.info())
#Here we have Extracurricular Activities as categorical columns 
#to prepare it we use 
print(data["Extracurricular Activities"].value_counts())
#use binary encoding (one hot )

#we have two ways 
#1 is get_dummies 
data = pd.get_dummies(data,columns =["Extracurricular Activities"],drop_first =True)
print(data['Extracurricular Activities_Yes'].dtype)
#Done 
print(data.columns)
#Build the model 
X=data[["Hours Studied","Previous Scores","Sleep Hours","Sample Question Papers Practiced","Extracurricular Activities_Yes"]]
Y=data[["Performance Index"]]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))

print(type(X_train),np.shape(X_train))
print(type(X_train), np.shape(X_train), np.shape(X_train))

#Scalling to the data 
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn import linear_model 
regressor =linear_model.LinearRegression()
regressor.fit(X_train_scaled,Y_train)
print(type(X_train_scaled),np.shape(X_train_scaled))
print("Cofeciont:",regressor.coef_[0])
print("Intercept:",regressor.intercept_)

#predict Results 
Y_test_predict = regressor.predict(X_test_scaled)

print("Predicted Performance Index (Y_test_pred):")
print(Y_test_predict[:10])
#طيب هسا بدي اعمل predict


#Metrics 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
r2 = r2_score(Y_test, Y_test_predict)
mae = mean_absolute_error(Y_test, Y_test_predict)
mse = mean_squared_error(Y_test, Y_test_predict)
print("\n Final Evaluation Metrics ")
print(f"1. R-squared (R2) Score: {r2:.4f}")
print(f"2. Mean Absolute Error (MAE): {mae:.2f} points")
print(f"3. Mean Squared Error (MSE): {mse:.2f}")


    
def predict_user_performance(hours, previous_scores, sleep_hours, sample_papers, activities):
    activities_encoded = 1 if activities.lower() == 'yes' else 0
    new_input = np.array([[hours, previous_scores, sleep_hours, sample_papers, activities_encoded]])
    
    new_input_scaled = scaler.transform(new_input) 
    
    predicted_performance = regressor.predict(new_input_scaled)
    
    return predicted_performance[0][0]


print("\n--- Enter Student Data for Performance Prediction ---")
h = float(input("1. Hours Studied: "))
p = float(input("2. Previous Scores: "))
s = float(input("3. Sleep Hours: "))
q = float(input("4. Sample Papers Practiced: "))
a = input("5. Extracurricular Activities (Yes/No): ")

result = predict_user_performance(h, p, s, q, a)
print("\n--- Prediction Result ---")
print(f"Predicted Performance Index is: {result:.2f} out of 100")

#model save 
import pickle 
import os 
model_path = "model_files/"
os.makedirs(model_path , exist_ok=True)

with open(model_path +"performence_model.pkl","wb") as file:
    pickle.dump(regressor, file)
with open(model_path + 'scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\n  Model and Scaler successfully saved to the 'model_files' folder.")

#Deploy to model 

# app2.py
from flask import Flask, request, render_template
import pickle
import numpy as np
import os

BASE_DIR = r'C:\Users\ASUS\model_files' 
MODEL_PATH = os.path.join(BASE_DIR, 'performence_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

regressor = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prev_scores = float(request.form['previous_scores'])
    sleep = float(request.form['sleep_hours'])
    papers = float(request.form['sample_papers'])
    activities = int(request.form['activities']) # بتيجيا 1 او 0 جاهزة من الـ Select

    features = np.array([[hours, prev_scores, sleep, papers, activities]])

    scaled_features = scaler.transform(features)
    prediction = regressor.predict(scaled_features)
    
    output = round(prediction[0][0], 2)

    return render_template('index.html', prediction_text=f'مؤشر الأداء المتوقع: {output} / 100')

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)




