import requests
import requests
import pandas as pd
import datetime
from json import JSONEncoder
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import  StandardScaler
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


def Isolation_forest(df):
    temp = df
    print(len(temp))
    
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df)
    data = pd.DataFrame(np_scaled)
    # train isolation forest
    print(np.where(np.isnan(data)))
    outliers_threshold = 0.05
    model =  IsolationForest(contamination=outliers_threshold )
    model.fit(data)
    temp['Label'] = list(pd.Series(model.predict(data)))
    temp['Label'] = temp['Label'].map( { -1 : 'A' , 1 : 'N' }  )
    return temp

def get_data(id,attri):
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    #query = {'id':id,'attributes':attri,'startDate':startDate,'endDate':endDate}
    query = {'id':id,'attributes':attri,'startDate':'2021-09-14 00:00:00','endDate':'2021-09-19 23:59:59'}
    #query = {'id':id,'attributes':attri,'startDate':'2021-11-23T04:46:31.448310','endDate':'2021-11-23T06:00:00'}
    
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    response = requests.get('http://54.206.42.58:8006/api/historicalData/getObjectAttributeHistoricalData', params=query)

    data = response.json()
    exportedData = data["data"]["ObjectData"]
    if id in exportedData:
        exportedData = data["data"]["ObjectData"][id][attri]["timeSeriesData"]
        required_data = pd.DataFrame(exportedData, columns=["measurementtimestamp", "value"])
        required_data.set_index('measurementtimestamp', inplace = True)
        required_data["value"] = required_data["value"].astype('float')
    else:
        required_data = []
    
    #required_data.columns = [id + '_' + 'AP']
    return required_data

def Isolation_forest_prediction( objectId,attributeId ):
    variable_of_intererst = get_data(objectId,attributeId)
    main_df = bring_weather_data()
    main_df[attributeId] =variable_of_intererst
    #print(main_df)
    main_df.columns = ['Ambient Temperature Celsius' ,'AVG in-plane irradiance' ,'Back-of-Module Temperature (deg C)','Irradiance Plane-of-Array (W/m^2)' , attributeId]
    print(main_df.shape)
    print(main_df)
    temp = main_df.copy( deep=True )
    hourly_df = resample_to_hour(main_df)
    print(hourly_df.shape)
    print(hourly_df)
    Anamoly_df =  Isolation_forest(hourly_df)
    Anamoly_df = Anamoly_df.dropna(axis=0,how="any")
    results =  Anamoly_df.reset_index().to_dict(orient = 'records')
    finalData = []
    for data in results:
        tempData = {
            "objectid": objectId,
            "attributeserviceid":"IFA_"+attributeId,
            "measurementtimestamp": data['measurementtimestamp'],
            "value" : data[attributeId],
            "customData": {
                "value": data['Label'],
                "Ambient Temperature Celsius": data['Ambient Temperature Celsius'],
                "AVG in-plane irradiance": data['AVG in-plane irradiance'],
                "Back-of-Module Temperature (deg C)": data['Back-of-Module Temperature (deg C)'],
                "Irradiance Plane-of-Array (W/m^2)": data['Irradiance Plane-of-Array (W/m^2)'],
            }
        }
        finalData.append(tempData)
    return finalData 

def ipfaPrediction():
    getLPFAObjects = {'serviceId':"IFA"}
    ifaResponse = requests.get('http://54.206.42.58:8000/api/mlservice', params=getLPFAObjects)
        
    ifaResponse = ifaResponse.json()
    ifaResponse = ifaResponse['data']['data']
    print(ifaResponse)
    finalData = []
    for data in ifaResponse:
        predictionData = Isolation_forest_prediction(data['objectid'],data['attributeserviceid'])
        tempData = {
            "objectid": data['objectid'],
            "attributeserviceid": "IFA_"+data['attributeserviceid'],
            "lpfaData": predictionData
        }
        finalData.append(tempData)

    return finalData

start = timer()
print(Isolation_forest_prediction('WP_SF_MVPS1.INV1','Active Power'))
end = timer()
time_taken = end - start
print(f'Time taken = {time_taken}')
