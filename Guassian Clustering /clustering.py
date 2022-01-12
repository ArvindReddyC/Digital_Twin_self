
import warnings
import os
import pandas as pd
import glob
import itertools
import requests
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
warnings.filterwarnings("ignore")
from json import JSONEncoder

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

    print(df.head(5))
    pivoted = df.pivot( index= 'measurementtimestamp' , columns='attributeserviceid' , values= 'value' )
    pivoted = pivoted.dropna(how='any',axis=0) 
    print( pivoted.columns)
    xaxis, yaxis =  pivoted.columns
    return pivoted.to_numpy() , xaxis, yaxis

def preprocess_(df_year):
    # Load irradiance data
    rslt_irradiance = df_year[df_year['PME_MeasurementName'] == 'Irradiance Global (W/m^2)']
    rslt_irradiance['TimestampUTC'] = pd.to_datetime(rslt_irradiance['TimestampUTC'], format='%d/%m/%Y %I:%M:%S.%f %p')

    # Pre-processing: fill-up the NaN values of Irradiance by '0' at Night and 'Mean Value' at Day
    irradiance_day_mean = rslt_irradiance[(rslt_irradiance['TimestampUTC'].dt.hour <= 18) & (rslt_irradiance['TimestampUTC'].dt.hour >=6 )]['PME_Value'].mean()
    rslt_irradiance.loc[(rslt_irradiance['PME_Value'].isnull()) & ((rslt_irradiance['TimestampUTC'].dt.hour > 18) | (rslt_irradiance['TimestampUTC'].dt.hour < 6)), 'PME_Value'] = 0
    rslt_irradiance.loc[(rslt_irradiance['PME_Value'].isnull()) & ((rslt_irradiance['TimestampUTC'].dt.hour <= 18) & (rslt_irradiance['TimestampUTC'].dt.hour >= 6)), 'PME_Value'] = irradiance_day_mean

    # Load active power data
    rslt_activepower = df_year[(df_year['PME_SourceName'] == 'WP_SF_MVPS4.PM1') & (df_year['PME_MeasurementName'] == 'Active Power')]
    rslt_activepower['TimestampUTC'] = pd.to_datetime(rslt_activepower['TimestampUTC'], format='%d/%m/%Y %I:%M:%S.%f %p')

    # Renaming the columns by readable identifiers
    rslt_irradiance.rename(columns={'PME_Value': 'irradiance'}, inplace=True)
    rslt_activepower.rename(columns={'PME_Value': 'activepower'}, inplace=True)

    # Merging the data
    rslt_irradiance_select = rslt_irradiance[['TimestampUTC', 'irradiance']]
    rslt_activepower_select = rslt_activepower[['TimestampUTC','activepower']]
    df_merged = pd.merge(rslt_irradiance_select, rslt_activepower_select, on='TimestampUTC')

    # BEFORE THIS POINT, LOAD THE DATA FOR THE CHOICE OF ATTRIBUTES AND THE TIMERANGE BY THE USER

    # Convert the dataframe to array of values
    X = df_merged[['irradiance', 'activepower']].to_numpy()

    # Returning the values for further computations
    return X

# Function to PRODUCE the results
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
        
        #print(mean, v[0], v[1], 180. + angle)
        ell_centers[i] = [i, mean[0], mean[1]]
        ell_axislengths[i] = [i, v[0], v[1]]
        ell_angles[i] = [i, 180. + angle]
    return ell_centers, ell_axislengths, ell_angles


def boot( id, attri,  start_date, end_date  , number_of_clusters):    
    
    X  , xaxis, yaxis = get_data(id, attri,  start_date, end_date)
    cluster_numbers = number_of_clusters;
    dpgmm = mixture.BayesianGaussianMixture(n_components=cluster_numbers,
                                            covariance_type='full').fit(X)
    XY_VALUES = X
    CLUSTERID = dpgmm.predict(X)

    ELL_CENTERS, ELL_AXISLENGTHS, ELL_ANGLES = att_ellipse(dpgmm.means_, dpgmm.covariances_, 1,
                'Bayesian Gaussian Mixture clustering', cluster_numbers)
    # print(ELL_CENTERS)
    # print(ELL_AXISLENGTHS)
    # print(ELL_ANGLES)
    #pd.Dataframe( data = [XY_VALUES]  )
    xy =  XY_VALUES.astype(float)
    ddf =  pd.DataFrame( data = CLUSTERID , columns=['Values'] )
    print(ddf.head(5))
    print(xy)
    lists = []
    for i,y in zip(xy,CLUSTERID):
        d_ = { 'v':list(i) , 'n':y }
        lists.append(d_)
    print(len(lists),len(ELL_AXISLENGTHS) ,len(ELL_CENTERS) )
    return {'ELL_CENTERS': ELL_CENTERS , 'ELL_AXISLENGTHS' : ELL_AXISLENGTHS , 'ELL_ANGLES' : ELL_ANGLES , 'cv':lists }
    #additionally we can pass the  xaxis label  and  y-axis label for graphing 
 
#Example of how to call this function 
print(boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Active Power,Irradiance Global (W/m^2),', '2021-03-01 00:00:00' , '2021-03-04 00:00:00' , 4 ))
