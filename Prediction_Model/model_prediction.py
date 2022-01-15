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
from datetime import datetime
from datetime import timedelta
import joblib
import pickle
import math

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()

def get_data(id,attri,start_date , end_date):
    ''' 
    Edit this if your running a cron Job 
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    '''
    
    #Change the start data and End Data for longer duration 
    query = {'id':id,'attributes':attri,'startDate':start_date,'endDate':end_date}
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    response = requests.get('http://54.206.42.58:8006/api/v2/historicalData/getObjectAttributeHistoricalData', params=query)

    data = response.json()
    exportedData = data["data"]["ObjectData"]
    df = pd.DataFrame(exportedData)


    pivoted = df.pivot( index= 'measurementtimestamp' , columns='attributeserviceid' , values= 'value' )
    pivoted.replace(np.nan, 0 , inplace=True)
    pivoted.rename( columns={ 'Irradiance Global (W/m^2)': 'irradiance' , 'Back-of-Module Temperature 2 (deg C)' : 'backtmp2' , 'Active Power' : 'activepower'   } , inplace=True )
    return pivoted


def use_input_normalize( irr_mean,backtmp2_mean,df_user ):
    df_user['agg_norm'] = 0.9*df_user['irradiance']/irr_mean + 0.1*df_user['temperature']/backtmp2_mean
    X = np.array(df_user['agg_norm'])
    X_user = X.reshape(-1,1)
    return X_user

def predict_act_power( start_date , end_date , model , pickle_obj , user_inputs = {}):
    '''
        for testing comment this before production.
    '''
    user_inputs = {'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
                   'irradiance': [400, 520 , 715], 
                   'temperature': [20, 24, 25]}

    print('Predicting User Inputs')
    df_user = pd.DataFrame(data=user_inputs)

    
    irr_mean ,  backtmp2_mean , RMSD_perKw ,activepower_mean = pickle.load(pickle_obj)


    X_user = use_input_normalize(irr_mean,backtmp2_mean,df_user)
    regressor = model
    y_user = regressor.predict(X_user)


    userinput_activepower = pd.DataFrame(data=y_user * activepower_mean, columns=["ActivePowerPrediction"] , index = user_inputs['measurementtimestamp'] )
    userinput_activepower.index.name = 'measurementtimestamp'
    #userinput_activepower.insert(loc=0, column='TimestampUTC', value = user_time_f)
    #userinput_activepower['activepower_predicted'] = userinput_activepower['ActivePowerPrediction']*activepower_mean

    userinput_activepower['lower'] = userinput_activepower["ActivePowerPrediction"]*(1-RMSD_perKw)
    userinput_activepower['upper'] = userinput_activepower["ActivePowerPrediction"]*(1+RMSD_perKw)
    train_data = get_data('WP_SF_MVPS4.PM1' , 'Active Power',start_date , end_date).reset_index().to_dict( orient = 'records' )
    return userinput_activepower.reset_index().to_dict( orient = 'records' ) , train_data



#Example  of how to call the function 
predict_act_power( '2021-03-01 00:00:00' , '2021-03-02 00:00:00' ,model , pickle_obj ,user_inputs )



