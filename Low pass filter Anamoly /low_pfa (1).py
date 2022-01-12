import requests
import pandas as pd
import datetime
from json import JSONEncoder

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()



def find_outliers(df , w):

   '''
   This function will compute anamolies in pandas seriers 

   param df: Pandas series to find anamolies in 
   param w: window size for rolling function 

   The Global variable index_of_outliers will have list of all the outliers 
   '''
   global conc
   index_of_outliers  = []

   TH = df.rolling(window = w  ).mean() +  ( 3 * df.rolling(window = w ).std() ) 
   TL = df.rolling(window = w  ).mean() -  ( 3 * df.rolling(window = w ).std() ) 
   concated  =  pd.concat( [df,TH,TL] , axis = 1)

   concated.columns = ['Power','TH','TL']
   
   concated.TH = concated.TH.shift( +1 )
   concated.TL = concated.TL.shift( +1 )
   if len(concated[(concated.Power > concated.TH) | (concated.Power < concated.TL ) ]):
      check = list(concated[(concated.Power > concated.TH) | (concated.Power < concated.TL )].index)[0]
      index_of_outliers.append( check  )
      df.drop( check  , axis = 'index'  , inplace = True  )
      find_outliers( df , w)

   else:

      conc = concated.copy(deep = True)
         
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
      #reading the variable of interest in a series 
      df = pd.Series(required_data["value"].astype(int))
   else:
      required_data = []
      #reading the variable of interest in a series 
      df = pd.Series(required_data)
   
      
   # required_data = pd.DataFrame(exportedData, columns=["measurementtimestamp", "value"])
   #    #reading the variable of interest in a series 
   # df = pd.Series(required_data["value"].astype(int))

      #intialising a list for to store the outliers
   index_of_outliers = []

      #copying the series containing the variable of interest to another variable
   copyData = df.copy( deep=True  )

      #parsing the series of interest into the anormaly function
   print(df)
   find_outliers(df , 12)

      #calculating the higher and lower bounds
   TH = df.rolling(window = 12  ).mean() +  ( 3 * df.rolling(window = 12 ).std() ).shift(+1) 
   TL = df.rolling(window = 12  ).mean() -  ( 3 * df.rolling(window = 12 ).std() ).shift(+1)

      #transforming the series containing the variable of interest to dataframe
   newdf = pd.DataFrame(copyData)

      #initialising a dataframe column to parse the labels (Anormaly/not anormaly)
   Label = pd.DataFrame(columns=['Label'])

      #merging the columns (series containing the variable of interest, higher bounds data, lower bounds data, and Label) 
   newdf  =  pd.concat( [newdf,TH,TL,Label] , axis = 1)

      #defining the columns headers (series containing the variable of interest, higher bounds data, lower bounds data, and Label) 
   newdf.columns = ['Power','TH','TL','Label']

      #appending the appropriate label for each record 
   for i in range(newdf.shape[0]):
      if i in index_of_outliers:
         newdf.at[i,'Label'] = "A" 
      else:
         newdf.at[i,'Label'] = "N" 

   newdf = newdf.dropna()
      #tranforming the result in the dataframe to a json file
   result = newdf.to_dict('records')
   finalData = []
   counter = 0
   for data in result:
      time_change = datetime.timedelta(seconds=(counter*1))
      counter = counter + 1
      tempData = {
         "objectid": "WP_SF_MVPS1.INV1",
         "attributeserviceid":"LPFA",
         "measurementtimestamp": datetime.datetime.now()+time_change,
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