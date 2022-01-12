import requests
import requests
import pandas as pd
import datetime
from json import JSONEncoder
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()

def bring_weather_data():
    weather_data = []
    for each in ['Ambient Temperature Celsius' ,'AVG in-plane irradiance' ,'Back-of-Module Temperature (deg C)','Irradiance Plane-of-Array (W/m^2)']:
        weather_data.append(get_data('WP_SF_MVPS4.WS1' ,each ))
    pd.concat(weather_data,axis =1 )
    return pd.concat(weather_data,axis =1 )

def resample_to_hour( df ):
  temp = df.reset_index()
  temp['measurementtimestamp']  = pd.to_datetime(temp['measurementtimestamp'],utc=True)
  temp = temp.resample('60T', on='measurementtimestamp').mean().dropna(how='any')
  return temp 


def OneSVM(df):
    temp = df
    print(len(temp))
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df)
    data = pd.DataFrame(np_scaled)
    # train isolation forest
    outliers_threshold = 0.05
    model = OneClassSVM(nu=outliers_threshold, kernel="rbf", gamma=0.01)
    # outliers_threshold = 0.05
    # model =  IsolationForest(contamination=outliers_threshold )
    model.fit(data)
    temp['Label'] = list(pd.Series(model.predict(data)))
    temp['Label'] = temp['Label'].map( { -1 : 'A' , 1 : 'N' }  )
    return temp

def get_data(id,attri):
    ''' 
    Edit this if your running a cron Job 
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    '''
    #query = {'id':id,'attributes':attri,'startDate':'2021-03-01 00:00:00','endDate':'2021-03-05 23:59:59'}
    #query = {'id':id,'attributes':attri,'startDate':startDate,'endDate':endDate}
    #query = {'id':id,'attributes':attri,'startDate':'2021-09-15 00:00:00','endDate':'2021-09-19 23:59:59'}
    query = {'id':id,'attributes':attri,'startDate':'2021-09-14 00:00:00','endDate':'2021-09-19 23:59:59'}
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    response = requests.get('http://54.206.42.58:8006/api/historicalData/getObjectAttributeHistoricalData', params=query)

    data = response.json()
    exportedData = data["data"]["ObjectData"][id][attri]["timeSeriesData"]

    required_data = pd.DataFrame(exportedData, columns=["measurementtimestamp", "value"])
    required_data.set_index('measurementtimestamp', inplace = True)
    required_data["value"] = required_data["value"].astype('float')
    #required_data.columns = [id + '_' + 'AP']
    return required_data

def osvm_prediction( objectId,attributeId ):
    variable_of_intererst = get_data(objectId,attributeId)
    main_df = bring_weather_data()
    main_df[attributeId] =variable_of_intererst
    #print(main_df)
    main_df.columns = ['Ambient Temperature Celsius' ,'AVG in-plane irradiance' ,'Back-of-Module Temperature (deg C)','Irradiance Plane-of-Array (W/m^2)' , attributeId]
    print(main_df.isna().sum())
    temp = main_df.copy( deep=True )
    hourly_df = resample_to_hour(main_df)
    Anamoly_df =  OneSVM(hourly_df)
    results =  Anamoly_df.reset_index().to_dict(orient = 'records')
    finalData = []
    counter = 0
    for data in results:
        tempData = {
        "objectid": objectId,
        "attributeserviceid":"OSVM_"+attributeId,
        "measurementtimestamp": data['measurementtimestamp'],
        "value" : data[attributeId],
        "customData": {
        "Label": data['Label'],
        "Ambient Temperature Celsius": data['Ambient Temperature Celsius'],
        "AVG in-plane irradiance": data['AVG in-plane irradiance'],
        "Back-of-Module Temperature (deg C)": data['Back-of-Module Temperature (deg C)'],
        "Irradiance Plane-of-Array (W/m^2)": data['Irradiance Plane-of-Array (W/m^2)'],
        }
        }
        finalData.append(tempData)
    return finalData 

def runTestFunction():
    getLPFAObjects = {}
    osvmResponse = requests.get('http://54.206.42.58:8000/api/mlservice', params=getLPFAObjects)
        
    osvmResponse = osvmResponse.json()
    osvmResponse = osvmResponse['data']['data']
    print(osvmResponse)
    finalData = []
    for data in osvmResponse:
        predictionData = osvm_prediction(data['objectid'],data['attributeserviceid'])
        tempData = {
            "objectid": data['objectid'],
            "attributeserviceid": "OSVM_"+data['attributeserviceid'],
            "osvmData": predictionData
        }
        finalData.append(tempData)

    return finalData


start = timer()
print(osvm_prediction('WP_SF_MVPS1.INV1','Active Power'))
end = timer()
time_taken = end - start
print(f'Time taken = {time_taken}')