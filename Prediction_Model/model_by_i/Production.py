import pandas as pd 
import numpy as np 
import joblib
import pickle


def predict_act_power( user_inputs = {} ):
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
   
    #model = joblib.load('/Users/arvindchandrasekarreddy/Desktop/PythonProjects/Digital_Twin_self/Prediction_Model/model_by_i/model_two_ip')
    with open('model_two_ip','rb') as file:
        model = pickle.load(file)
    #model = pickle.load(open('/Users/arvindchandrasekarreddy/Desktop/PythonProjects/Digital_Twin_self/Prediction_Model/model_by_i/model_two_ip.sav', 'rb'))
    '''
        for testing comment this before production.
    '''
    print('Predicting User Inputs')
    #regressor = model
    print(df_user[['irradiance','temperature']].to_numpy())
    
    y_user = model.predict(df_user[['irradiance','temperature']].to_numpy())

    predicted_activePower = pd.DataFrame(data=y_user , columns=["ActivePowerPrediction"] , index = user_inputs['measurementtimestamp'] )
    predicted_activePower.index.name = 'measurementtimestamp'


    return predicted_activePower.reset_index().to_dict( orient = 'records' )



#Example  of how to call the function 
predict_act_power({'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
                'irradiance': [400, 520 , 715], 
                'temperature': [20, 24, 25]} )