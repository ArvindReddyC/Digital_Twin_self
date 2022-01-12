import warnings
import os
import pandas as pd
import glob
import itertools
import requests
import numpy as np
from scipy import linalg
# import matplotlib.pyplot as plt
# import matplotlib as mpl
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


def plot_results(X, Y_, means, covariances, index, title):
    # Creating subplots
    plt.figure(figsize=(20,20))
    splot = plt.subplot(2, 1, 1 + index)
    # Plot the points and ellipse
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle, color=color)
        #print(mean, v[0], v[1], angle)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
        
    # Define the plot properties 
    plt.xlim(-30, 2000)
    plt.ylim(-5, 250)
    plt.xticks(np.arange(-30, 2000, 100))
    plt.yticks(np.arange(-5, 250, 5))
    plt.title(title)
    plt.show()

def get_data(id,attri,start_date , end_date):
    ''' 
    Edit this if your running a cron Job 
    queryTimeDiff = datetime.timedelta(minutes=90)
    endDate = datetime.datetime.now()
    startDate = datetime.datetime.now() - queryTimeDiff
    '''
    [id1 , id2] = id.split(",")
    [attri1 , attri2] = attri.split(",")

    #Change the start data and End Data for longer duration 
    query = {'id':id1,'attributes':attri1,'startDate':start_date,'endDate':end_date}
    # id1+att1 ; id2+att2
    # merge
    query = DateTimeEncoder().encode(query)
    query = eval(query)
    response = requests.get('http://54.206.42.58:8006/api/v2/historicalData/getObjectAttributeHistoricalData', params=query)
    data1 = response.json()
    exportedData1 = data1["data"]["ObjectData"]
    df1 = pd.DataFrame(exportedData1)
    

    #Change the start data and End Data for longer duration 
    query2 = {'id':id2,'attributes':attri2,'startDate':start_date,'endDate':end_date}  
    # id1+att1 ; id2+att2
    # merge
    query2 = DateTimeEncoder().encode(query2)
    query2 = eval(query2)
    response2 = requests.get('http://54.206.42.58:8006/api/v2/historicalData/getObjectAttributeHistoricalData', params=query2)
    data2 = response2.json()
    exportedData2 = data2["data"]["ObjectData"]
    df2 = pd.DataFrame(exportedData2)
    #print(df2.head(5))

 
    df = pd.concat([df1, df2])


    #print(df.head(5))
    pivoted = df.pivot( index= 'measurementtimestamp' , columns='attributeserviceid' , values= 'value' )
    pivoted = pivoted[[attri1,attri2]]
    pivoted = pivoted.dropna(how='any',axis=0) 
    #print( pivoted.columns)
    xaxis, yaxis =  pivoted.columns
    #print(pivoted.head(5))
    return pivoted.to_numpy() , xaxis, yaxis


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
        
        ##print(mean, v[0], v[1], 180. + angle)
        ell_centers[i] = [i, mean[0], mean[1]]
        ell_axislengths[i] = [i, v[0], v[1]]
        ell_angles[i] = [i, angle]
    return ell_centers, ell_axislengths, ell_angles


def boot( id, attri,  start_date, end_date  , number_of_clusters):    
    
    X  , xaxis, yaxis = get_data(id, attri,  start_date, end_date)
    print(X)
    cluster_numbers = number_of_clusters;
    dpgmm = mixture.BayesianGaussianMixture(n_components=cluster_numbers,
                                            covariance_type='full').fit(X)
    XY_VALUES = X
    CLUSTERID = dpgmm.predict(X)

    ELL_CENTERS, ELL_AXISLENGTHS, ELL_ANGLES = att_ellipse(dpgmm.means_, dpgmm.covariances_, 1,
                'Bayesian Gaussian Mixture clustering', cluster_numbers)
    #print(ELL_CENTERS)
    #print(ELL_AXISLENGTHS)
    #print(ELL_ANGLES)
    lists = []
    xy_value_data = XY_VALUES.astype(float)
    # xy_value_data = xy_value_data.tolist()
    for i,y in zip(xy_value_data,CLUSTERID):
        d_ = { 'value':list(i) , 'name':str(y) }
        lists.append(d_)
    print(CLUSTERID)
    #plot_results(X, CLUSTERID, dpgmm.means_, dpgmm.covariances_, 1,'Bayesian Gaussian Mixture clustering')

    #plt.show()
  
    #return {'ELL_CENTERS': ELL_CENTERS , 'ELL_AXISLENGTHS' : ELL_AXISLENGTHS , 'ELL_ANGLES' : ELL_ANGLES , 'XY_VALUES': lists}
  
#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Irradiance Global (W/m^2),Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )

#boot( 'WP_SF_MVPS4.PM1,WP_SF_MVPS4.WS1', 'Active Power,Irradiance Global (W/m^2)', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )

#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'Back-of-Module Temperature 2 (deg C),Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )

#boot( 'WP_SF_MVPS4.WS1,WP_SF_MVPS4.PM1', 'AVG in-plane irradiance,Active Power', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )


boot( 'WP_SF_MVPS4.PM1,WP_SF_MVPS4.WS1', 'Active Power,Back-of-Module Temperature 2 (deg C)', '2021-10-01 00:00:00' , '2021-11-01 00:00:00' , 4 )
