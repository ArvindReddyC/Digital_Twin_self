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




def get_data(id,attri):
    ''' 
    Edit this if your running a cron Job 
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    '''
    
    #Change the start data and End Data for longer duration 
    query = {'id':id,'attributes':attri,'startDate':'2021-03-01 00:00:00','endDate':'2021-03-02 00:00:00'}
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


def normalize_data(df_merged):
  irr_mean = df_merged['irradiance'].mean()
  backtmp2_mean = df_merged['backtmp2'].mean()
  activepower_mean = df_merged['activepower'].mean()

  df_merged['ap_norm'] = df_merged['activepower']/activepower_mean

  df_merged['ag_val'] = 0.9*(df_merged['irradiance']/irr_mean) + 0.1*(df_merged['backtmp2']/backtmp2_mean)

  return df_merged, activepower_mean,irr_mean,backtmp2_mean


def pre_process(df_merged):
  X = np.array(df_merged['ag_val'])
  y = np.array(df_merged['ap_norm'])

  split = int(len(df_merged)*0.9)

  X_train = X[:split]
  X_test = X[split:]
  y_train = y[:split]
  y_test = y[split:]
  X_train = X_train.reshape(-1,1)
  X_test = X_test.reshape(-1,1)

  return X_train, X_test, y_train, y_test, split

def get_error_rate(df_merged,split,predicted_activepower):
    

  y = df_merged['activepower'][split:]
  y_bar = predicted_activepower['ActivePowerPrediction']

  summation = 0 
  n = df_merged['activepower'][split:].count()


  for i in range (0,n-1):
    difference = y.iloc[i-1] - y_bar.iloc[i] 
    squared_difference = difference**2
    summation = summation + squared_difference

  RMSD_perKw = math.sqrt(summation)/df_merged['activepower'][split:].sum()
  return RMSD_perKw


def use_input_normalize( irr_mean,backtmp2_mean,df_user ):
    df_user['agg_norm'] = 0.9*df_user['irradiance']/irr_mean + 0.1*df_user['temperature']/backtmp2_mean
    X = np.array(df_user['agg_norm'])
    X_user = X.reshape(-1,1)
    return X_user




def bootstrap(request_type ='train' , user_inputs = {}):
    if( request_type == 'train' ):
        print('Building the mode')
        pivoted  = get_data('WP_SF_MVPS4.PM1,WP_SF_MVPS4.WS1','Irradiance Global (W/m^2),Back-of-Module Temperature 2 (deg C),Active Power')
        pivoted = pivoted.astype( 'float' )
        df_merged, activepower_mean,irr_mean,backtmp2_mean = normalize_data(pivoted)
        X_train, X_test, y_train, y_test, split  = pre_process(df_merged)
        clf = SVR(kernel='rbf')
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model.pkl')

        y_pred = clf.predict(X_test)
        predicted_activepower = pd.DataFrame(data=y_pred, columns=["ActivePowerPrediction"])
        predicted_activepower['activepower_predicted'] = predicted_activepower['ActivePowerPrediction']*activepower_mean
        RMSD_perKw = get_error_rate(df_merged,split,predicted_activepower)

        #save the 3 parameters irr_mean,backtmp2_mean and RMSD_perKw to be used for predicting user inputs.

        obj = [ irr_mean ,  backtmp2_mean , RMSD_perKw ,activepower_mean ]

        f = open('store.pckl', 'wb')
        pickle.dump(obj, f)
        f.close()
        print('Model Created')
    else:
        '''
        for testing comment this before production.
        '''
        user_inputs = {'measurementtimestamp': ['some_date', 'some_date', 'some_date'],
                    'irradiance': [400, 520 , 715], 
                    'temperature': [20, 24, 25]}
    
        print('Predicting User Inputs')
        df_user = pd.DataFrame(data=user_inputs)

        f = open('store.pckl', 'rb')
        irr_mean ,  backtmp2_mean , RMSD_perKw ,activepower_mean = pickle.load(f)
        f.close()

        X_user = use_input_normalize(irr_mean,backtmp2_mean,df_user)
        clf = joblib.load('model.pkl')
        y_user = clf.predict(X_user)


        userinput_activepower = pd.DataFrame(data=y_user * activepower_mean, columns=["ActivePowerPrediction"] , index = user_inputs['measurementtimestamp'] )
        userinput_activepower.index.name = 'measurementtimestamp'
        #userinput_activepower.insert(loc=0, column='TimestampUTC', value = user_time_f)
        #userinput_activepower['activepower_predicted'] = userinput_activepower['ActivePowerPrediction']*activepower_mean

        userinput_activepower['lower'] = userinput_activepower["ActivePowerPrediction"]*(1-RMSD_perKw)
        userinput_activepower['upper'] = userinput_activepower["ActivePowerPrediction"]*(1+RMSD_perKw)
        train_data = get_data('WP_SF_MVPS4.PM1' , 'Active Power').reset_index().to_dict( orient = 'records' )
        return userinput_activepower.reset_index().to_dict( orient = 'records' ) , train_data 
    

#testing the function
bootstrap()
print(bootstrap( request_type ='user'  ))