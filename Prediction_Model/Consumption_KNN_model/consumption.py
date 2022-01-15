import pandas as pd 
import numpy as np 
import joblib
import pickle


def predict_consumption_power( user_inputs = {} ):
    '''
        for testing comment this before production.
    '''
    # user_inputs = {'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
    #             'irradiance': [400, 520 , 715], 
    #             'temperature': [20, 24, 25]}

    print('Predicting User Inputs')
    df_user = pd.DataFrame(data=user_inputs , index = user_inputs['measurementtimestamp'] , columns=['irradiance','temperature'] )
    df_user.index.name = 'measurementtimestamp'
    df_user.astype(float)
    '''
    Here we will  if else clauses  
    '''
   
    #model = joblib.load('/Users/arvindchandrasekarreddy/Desktop/PythonProjects/Digital_Twin_self/Prediction_Model/Consumption_KNN_model/KNN.sav')
    model = joblib.load('KNN.sav')
    
   
    '''
        for testing comment this before production.
    '''
    
    print('Predicting User Inputs')
    regressor = model
    print(df_user[['irradiance','temperature']].to_numpy())
    y_user = regressor.predict(df_user[['irradiance','temperature']].to_numpy())


    predicted_activePower = pd.DataFrame(data=y_user , columns=["ActivePowerPrediction"] , index = user_inputs['measurementtimestamp'] )
    predicted_activePower.index.name = 'measurementtimestamp'


    return predicted_activePower.reset_index().to_dict( orient = 'records' )


#Example_of_how to call the function
# predict_consumption_power({'measurementtimestamp': ['01', '02', '03'],
#                 'irradiance': [400, 520 , 715], 
#                 'temperature': [20, 24, 25]} )
