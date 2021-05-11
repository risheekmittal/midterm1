import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("Risheek.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Risheek Mittal - Classification Dataset1.csv')
X = dataset.iloc[:,1:10].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,2:3]) 
#Replacing missing data with the calculated mean value  
X[:,2:3]= imputer.transform(X[:,2:3])  

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 3:9]) 
#Replacing missing data with the constant value  
X[:, 3:9]= imputer.transform(X[:,3:9])  

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:,2])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
  print("Leaving Chances",output)
  if output==[1]:
    prediction="Will Exit"
  else:
    prediction="Won't Exit"
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Deep Learning  Lab Experiment Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Leaving Chance Prediction")
    CreditScore = st.number_input('Insert a Credit Score',0,1000)
    Geography = st.number_input('Insert a Geography',0,1)
    Gender = st.number_input('Insert a Gender',0,1)
    Age = st.number_input('Insert a age',18,60)
    Tenure = st.number_input('Insert a tenure',0,9)
    Balance = st.number_input('Insert balance',0,200000)
    HasCrCard= st.number_input('Insert 1 for crcard or 0',0,1)
    IsActiveMember= st.number_input('1 for active otherwise 0',0,1)
    EstimatedSalary = st.number_input('Insert a estimated salary',0,200000)
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Risheek Mittal")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
