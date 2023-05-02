#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:29:48 2023

@author: student
"""
import numpy as np
import pickle
import streamlit as st
#Loading the Saved Model
loaded_model=pickle.load(open('/home/student/Desktop/diabeties prediction/trained_model.sav','rb'))
scaler = pickle.load(open('/home/student/Desktop/diabeties prediction/scaler.sav', 'rb'))
# Creating a function for Prediction
def Diabetes_prediction(input_data):


	# changing the input_data to numpy array
	input_data_as_numpy_array = np.asarray(input_data)

	# reshape the array as we are predicting for one instance
	input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

	std_data = scaler.transform(input_data_reshaped)

	prediction = loaded_model.predict(std_data)
	print(prediction)

	if (prediction[0] == 0):
		return 'The person is not diabetic'
	else:
		return 'The person is diabetic'

def main():
    #Giving a title 
    st.title('DIABETES PREDICTION WEB APP')
    
    Pregnancies= st.text_input('No of PREGNANCIES')
    
    Glucose= st.text_input('GLUCOSE')
    
    BloodPressure= st.text_input('BLOOD PRESSURE')
    
    SkinThickness= st.text_input('SKIN-THICKNESS')
    
    Insulin= st.text_input('INSULIN')
    
    BMI= st.text_input('BMI VALUE')
    
    DiabetespedigreeFunction= st.text_input('DIABETES PEDGREE FUNCTION VALUE')
    
    Age= st.text_input('AGE OF THE PERSON')

    #Code for prediction
    diagnosis=''
    
   #Creating a button for Prediction
    if st.button('DIABETES  TEST RESULTS'):
        diagnosis = Diabetes_prediction([ Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetespedigreeFunction,Age])
        print([ Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetespedigreeFunction,Age])
        
    st.success(diagnosis)
    


if __name__ == '__main__': 
    main()      
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
    
    
