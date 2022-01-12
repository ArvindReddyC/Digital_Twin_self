
import warnings
import time
import os
import pandas as pd
import glob
import itertools
import requests
import numpy as np
from scipy import linalg
from sklearn import mixture
warnings.filterwarnings("ignore")
from json import JSONEncoder
import datetime
import json
import matplotlib as mpl
import matplotlib.pyplot as plt


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()




def get_data(id,attri,start_date , end_date, period = 'D'):
    ''' 
    Edit this if your running a cron Job 
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    '''
    [id1 , id2] = id.split(",")
    [attri1 , attri2] = attri.split(",")

    #Change the start data and End Data for longer duration 
    query = {'id':id,'attributes':attri,'startDate':start_date,'endDate':end_date}
    
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    
    response = requests.get('http://54.206.42.58:8006/api/v2/historicalData/getObjectAttributeHistoricalData', params=query)
           
    data1 = response.json()
    exportedData1 = data1["data"]["ObjectData"]
    df = pd.DataFrame(exportedData1)


    pivoted = df.pivot( index= 'measurementtimestamp' , columns='attributeserviceid' , values= 'value' )
    pivoted = pivoted[[attri1,attri2]]
    pivoted = pivoted.dropna(how='any',axis=0) 
    if( period  == '1H' ):
        pivoted =  resample_to_hour(pivoted).dropna(how='any',axis=0) 
    elif(period == 'D'): 
        pivoted =  resample_to_day(pivoted).dropna(how='any',axis=0) 
    elif(period == 'M'):
        pivoted =  resample_to_Month(pivoted).dropna(how='any',axis=0) 
    xaxis, yaxis =  pivoted.columns
    return pivoted.to_numpy() , xaxis, yaxis  

def resample_to_hour( df ):
  temp = df.astype('float')    
  temp = temp.reset_index()
  temp['measurementtimestamp']  = pd.to_datetime(temp['measurementtimestamp'],utc=True)
  temp = temp.resample('1H', on='measurementtimestamp').mean()
  return temp 

def resample_to_day( df ):
  temp = df.astype('float')    
  temp = temp.reset_index()
  temp['measurementtimestamp']  = pd.to_datetime(temp['measurementtimestamp'],utc=True)
  temp = temp.resample('D', on='measurementtimestamp').mean()
  return temp 

def resample_to_Month( df ):
  temp = df.astype('float')    
  temp = temp.reset_index()
  temp['measurementtimestamp']  = pd.to_datetime(temp['measurementtimestamp'],utc=True)
  temp = temp.resample('M', on='measurementtimestamp')
  return temp 

#Function to PRODUCE the results
def att_ellipse(means, covariances, index, title, nclusters):
    ell_centers = [None]*nclusters
    ell_axislengths = [None]*nclusters
    ell_angles = [None]*nclusters
    # Find the attributes of the ellipse
    for i, (mean, covar) in enumerate(zip(
            means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Produce an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        
        ##print(mean, v[0], v[1], 180. + angle)
        ell_centers[i] = [i, mean[0], mean[1]]
        ell_axislengths[i] = [i, v[0], v[1]]
        ell_angles[i] = [i,  angle]
    return ell_centers, ell_axislengths, ell_angles


def boot( id, attri,  start_date, end_date  , number_of_clusters,period = 'raw'):    
    
    X  , xaxis, yaxis = get_data(id, attri,  start_date, end_date,period)
    cluster_numbers = number_of_clusters;
    dpgmm = mixture.BayesianGaussianMixture(n_components=cluster_numbers,
                                            covariance_type='full',random_state=12).fit(X)
    XY_VALUES = X
    CLUSTERID = dpgmm.predict(X)

    ELL_CENTERS, ELL_AXISLENGTHS, ELL_ANGLES = att_ellipse(dpgmm.means_, dpgmm.covariances_, 1,
                'Bayesian Gaussian Mixture clustering', cluster_numbers)
    #print(ELL_CENTERS)
    #print(ELL_AXISLENGTHS)
    #print(ELL_ANGLES)
    lists = []
    xy_value_data = XY_VALUES.astype(float)
    
    for i,y in zip(xy_value_data,CLUSTERID):
        d_ = { 'value':list(i) , 'name':str(y) }
        lists.append(d_)

  
    return {'ELL_CENTERS': ELL_CENTERS , 'ELL_AXISLENGTHS' : ELL_AXISLENGTHS , 'ELL_ANGLES' : ELL_ANGLES , 'XY_VALUES': lists}
  
#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Irradiance Global (W/m^2),Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )
#boot( 'WP_SF_MVPS4.PM1,WP_SF_MVPS4.WS1', 'Active Power,Irradiance Global (W/m^2)', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )
#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Back-of-Module Temperature 2 (deg C),Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )
#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'AVG in-plane irradiance,Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )
#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Irradiance Global (W/m^2),Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )

print(boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.WS1', 'Ambient Temperature Celsius,Weather Atmospheric Pressure', '2021-11-01 00:00:00' , '2021-11-30 00:00:00' , 4 , period = '1H'))



