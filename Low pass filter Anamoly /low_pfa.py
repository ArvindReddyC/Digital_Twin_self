import requests
import pandas as pd
import datetime
from json import JSONEncoder

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()



'''
Using Median to find anmolies 
'''
def find_median_mad_Anomaly(df , w):
  '''
  This function finds Anmolies in data, based on the median 
  param df: can be a dataframe or  series with data_time as index 
  param w: is the rolling window size
  '''
  TH = df.rolling(window = w  ).median() +  ( 3 * df.rolling(window = w ).apply(mad,raw=True)) 
  TL = df.rolling(window = w  ).median() -  ( 3 * df.rolling(window = w ).apply(mad,raw=True)) 
  

  concated  =  pd.concat( [df,TH,TL] , axis = 1)
  print(concated)
  concated.columns = ['Power','TH','TL']

  concated.TH = concated.TH.shift( +1)
  concated.TL = concated.TL.shift( +1 )

  Anomalies = concated[(concated.Power > concated.TH) | (concated.Power < concated.TL )]
  concated['Label'] = 'N'

  concated.loc[Anomalies.index,'Label'] = 'A'
  return concated

   
   
def runTestFunction():
   queryTimeDiff = datetime.timedelta(minutes=10)
   endDate = datetime.datetime.now()
   startDate = datetime.datetime.now() - queryTimeDiff
   query = {'id':'WP_SF_MVPS1.INV1','attributes':'Active Power','startDate':startDate,'endDate':endDate}
   query = DateTimeEncoder().encode(query)
   query = eval(query)
   print(query)
   response = requests.get('http://54.206.42.58:8006/api/historicalData/getObjectAttributeHistoricalData', params=query)
      
   data = response.json()
   exportedData = data["data"]["ObjectData"]
   if "WP_SF_MVPS1.INV1" in exportedData:
      newExportedData = exportedData["WP_SF_MVPS1.INV1"]["Active Power"]["timeSeriesData"]
      required_data = pd.DataFrame(newExportedData, columns=["measurementtimestamp", "value"])
      required_data.set_index('measurementtimestamp', inplace = True)
      df = required_data["value"].astype('int')
   else:
      required_data = []
      #reading the variable of interest in a series 
      df = pd.Series(required_data)
   
      

   concated_df = find_median_mad_Anomaly(df , 12)
   concated_df.reset_index(inplace=True)
   results = concated_df.to_dict(orient='records')

   
   finalData = []
   counter = 0
   for data in result:
      #time_change = datetime.timedelta(seconds=(counter*1))
      counter = counter + 1
      tempData = {
         "objectid": "WP_SF_MVPS1.INV1",
         "attributeserviceid":"LPFA",
         "measurementtimestamp": data['measurementtimestamp'],
         "value" : data['Power'],
         "customData": {
            "value": data['Label'],
            "upper": data['TH'],
            "lower": data['TL']
         }
      }
      finalData.append(tempData)

   finalData = DateTimeEncoder().encode(finalData)

   return finalData