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

def bootstrap( start_date , end_date):
    pivoted  = get_data('WP_SF_MVPS4.PM1,WP_SF_MVPS4.WS1','Irradiance Global (W/m^2),Back-of-Module Temperature 2 (deg C),Active Power',start_date , end_date )
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
    #calling the load model method in the app.py to reload the model and pickle file to RAM
    from app import load_model
    load_model()


#Example  of how to call the function 
bootstrap( '2021-03-01 00:00:00' , '2021-03-02 00:00:00')


'''
The below code should be in the App.py file 
'''
pickle_obj  = '' # Initialising Global Variables 
model =  ''  

def load_model():
    global pickle_obj,model
    pickle_obj = open('store.pckl', 'rb')
    model = joblib.load('model.pkl')

