import requests
import pandas as pd
import datetime
from json import JSONEncoder
import numpy as np
from timeit import default_timer as timer

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()


#median absolute deviation
mad = lambda x: np.median(np.fabs(x - np.median(x)))

'''
Using Median to find anmolies 
'''
def find_median_mad_Anomaly(df , w):
  '''
  This function finds Anmolies in data, based on the median 
  param df: can be a dataframe or  series with data_time as index 
  param w: is the rolling window size
  for live purpose always use the number 3 in TH and TL as it gives higher accuracy
  '''
  TH = df.rolling(window = w  ).median() +  ( 3 * df.rolling(window = w ).apply(mad,raw=True)) 
  TL = df.rolling(window = w  ).median() -  ( 3 * df.rolling(window = w ).apply(mad,raw=True)) 
  

  concated  =  pd.concat( [df,TH,TL] , axis = 1)
  concated.columns = ['Power','TH','TL']

  concated.TH = concated.TH.shift( +1)
  concated.TL = concated.TL.shift( +1 )

  Anomalies = concated[(concated.Power > concated.TH) | (concated.Power < concated.TL )]
  concated['Label'] = 'N'

  concated.loc[Anomalies.index,'Label'] = 'A'
  return concated


def lpfaPrediction(objectId,attributeId):
   queryTimeDiff = datetime.timedelta(minutes=90)
   endDate = datetime.datetime.now()
   startDate = datetime.datetime.now() - queryTimeDiff
   #query = {'id':objectId,'attributes':attributeId,'startDate':startDate,'endDate':endDate}
   query = {'id':objectId,'attributes':attributeId,'startDate':'2021-09-14 00:00:00','endDate':'2021-09-19 23:59:59'}
   query = DateTimeEncoder().encode(query)
   query = eval(query)
   print(query)
   response = requests.get('http://54.206.42.58:8006/api/historicalData/getObjectAttributeHistoricalData', params=query)
      
   data = response.json()
   exportedData = data["data"]["ObjectData"]
   if objectId in exportedData:
      newExportedData = exportedData[objectId][attributeId]["timeSeriesData"]
      required_data = pd.DataFrame(newExportedData, columns=["measurementtimestamp", "value"])
      required_data.set_index('measurementtimestamp', inplace = True)
      df = required_data["value"].astype('int')
   else:
      required_data = []
      #reading the variable of interest in a series 
      df = pd.Series(required_data)
   
      

   concated_df = find_median_mad_Anomaly(df , 12)
   concated_df.reset_index(inplace=True)
   concated_df = concated_df.dropna()
   results = concated_df.to_dict(orient='records')
   
   finalData = []
   for data in results:
      tempData = {
         "objectid": objectId,
         "attributeserviceid":"LPFA_"+attributeId,
         "measurementtimestamp": data['measurementtimestamp'],
         "value" : data['Power'],
         "customData": {
            "value": data['Label'],
            "upper": data['TH'],
            "lower": data['TL']
         }
      }
      finalData.append(tempData)

   return finalData     
   
def runTestFunction():
   getLPFAObjects = {'serviceId':"LPFA"}
   lpfaResponse = requests.get('http://54.206.42.58:8000/api/mlservice', params=getLPFAObjects)
      
   lpfaDataResponse = lpfaResponse.json()
   lpfaDataResponse = lpfaDataResponse['data']['data']
   print(lpfaDataResponse)
   finalData = []
   for data in lpfaDataResponse:
      predictionData = lpfaPrediction(data['objectid'],data['attributeserviceid'])
      tempData = {
         "objectid": data['objectid'],
         "attributeserviceid": "LPFA_"+data['attributeserviceid'],
         "lpfaData": predictionData
      }
      finalData.append(tempData)

   return finalData
start = timer()
lpfaPrediction(objectId = 'WP_SF_MVPS1.INV1' ,attributeId = 'Active Power' )
end = timer()
time_taken = end - start
print(f'Time taken = {time_taken}')