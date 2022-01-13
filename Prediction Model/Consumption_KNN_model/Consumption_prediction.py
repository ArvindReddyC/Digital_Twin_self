import pandas as pd 
import requests
from json import JSONEncoder
import numpy as np

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn import utils

#Import svm model
from sklearn.svm import SVR
import datetime
import joblib
import pickle
import math

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()



def predict_consumption(model ,user_inputs = {}):
    '''
        for testing comment this before production.
    '''
    # user_inputs = {'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
    #             'irradiance': [400, 520 , 715], 
    #             'temperature': [20, 24, 25]}

    print('Predicting User Inputs')
    df_user = pd.DataFrame(data=user_inputs)

    
    regressor = model
    y_user = regressor.predict(X_user)


    userinput_activepower = pd.DataFrame(data= y_user , columns=["Consumption_Prediction"] , index = user_inputs['measurementtimestamp'] )
    userinput_activepower.index.name = 'measurementtimestamp'
    
    return userinput_activepower.reset_index().to_dict( orient = 'records' )



#Example  of how to call the function 
# predict_act_power( '2021-03-01 00:00:00' , '2021-03-02 00:00:00' ,model , pickle_obj ,user_inputs )


