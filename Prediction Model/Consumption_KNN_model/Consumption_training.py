import pandas as  pd
import numpy as np
from json import JSONEncoder
import requests
import matplotlib.pyplot as plt 
import time

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
import sklearn
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
    query = {'id':id,'attributes':attri,'startDate':start_date,'endDate':end_date}
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    response = requests.get('http://54.206.42.58:8006/api/v2/historicalData/getObjectAttributeHistoricalData', params=query)

    data = response.json()
    
    exportedData = data["data"]["ObjectData"]
    df = pd.DataFrame(exportedData)

    #print(df.head(5))
    pivoted = df.pivot( index= 'measurementtimestamp' , columns='attributeserviceid' , values= 'value' )
    pivoted.replace(np.nan, 0 , inplace=True)
    #pivoted.rename( columns={ 'Irradiance Global (W/m^2)': 'irradiance' , 'Back-of-Module Temperature 2 (deg C)' : 'backtmp2' , 'Active Power' : 'activepower'   } , inplace=True )
    
    return pivoted


def KNN_training(  ):
    df = get_data('WP_HVW_SWB.1J07_PM_WestFdr,WP_SF_MVPS4.WS1','Active Power,Irradiance Global (W/m^2),Back-of-Module Temperature 2 (deg C)')
    subset_df = df['Active Power,Irradiance Global (W/m^2),Back-of-Module Temperature 2 (deg C)'].astype(float)
    print(subset_df.dtypes)
    X_train, X_test, y_train, y_test = train_test_split(subset_df , dependent ,
    test_size=0.25) 
    neigh = KNeighborsRegressor(n_neighbors=100,algorithm='brute')
    neigh.fit(X_train,y_train)
    # #predictions = neigh.predict(X_test)
    # mse = sklearn.metrics.mean_squared_error(y_test, predictions)
    # rmse = math.sqrt(mse)
    joblib.dump(neigh, 'model_KNN.pkl')
    return joblib.load('model_KNN.pkl')


