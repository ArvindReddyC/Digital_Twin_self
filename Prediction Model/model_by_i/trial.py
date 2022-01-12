import pandas as pd 
import numpy as np 
import joblib


def predict_act_power(   resource_flag , user_inputs = {}):
    '''
        for testing comment this before production.
    '''
    user_inputs = {'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
                   'irradiance': [400, 520 , 715] }

    print('Predicting User Inputs')
    df_user = pd.DataFrame(data=user_inputs)
    df_user.irradiance.astype(float)
    '''
    Here we will  if else clauses  
    '''
    if(resource_flag == 'model_bk1'):
        model = joblib.load('model_bk1.pkl')
    elif(resource_flag == 'model_bk2' ):
        model = joblib.load('model_bk2.pkl')
    elif(resource_flag == 'model_bk3' ):
        model = joblib.load('model_bk1.pkl')
    elif(resource_flag == 'model_inv1' ):
        model = joblib.load('model_inv1.pkl')
    elif(resource_flag == 'model_inv2' ):
        model = joblib.load('model_inv2.pkl')
    elif(resource_flag == 'model_inv3' ):
        model = joblib.load('model_inv3.pkl')
    elif(resource_flag == 'model_inv4' ):
         model = joblib.load('model_inv4.pkl')
        
    
           
    regressor = model
    y_user = regressor.predict(df_user['irradiance'])


    predicted_activePower = pd.DataFrame(data=y_user , columns=["ActivePowerPrediction"] , index = user_inputs['measurementtimestamp'] )
    predicted_activePower.index.name = 'measurementtimestamp'


    return predicted_activePower.reset_index().to_dict( orient = 'records' )



#Example  of how to call the function 
predict_act_power (resource_flag ,user_inputs )
#resouce flag indicates from where the data is coming from.? 