#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary Libraries

# In[24]:


import numpy as np #Linear Algebra
import pandas as pd #data processing
import geopandas as gpd
import seaborn as sns #Data Visualisation 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime as dt

import warnings


warnings.filterwarnings("ignore")


# ### Importing Dataset

# In[25]:


df = pd.read_csv('NewTaxiProject.csv')


# In[26]:


del df['total_amount']


# In[27]:


get_ipython().system('pip install dataframe_image')
import dataframe_image as dfi
#dfi.export(data_df, "table.png")


# In[22]:


pd.set_option("display.max_column", None)
pd.set_option("display.max_colwidth", None)

import dataframe_image as dfi



dfi.export(df.describe(), "Datadescribe.png")


# ### Exploring the Dataset 

# In[28]:


df.head()


# 

# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[29]:


df.shape


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df.nunique()


# In[11]:


#let's change datatype
df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
df['dropoff_datetime'] = df['dropoff_datetime'].astype('datetime64[ns]')
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')


# In[12]:


df.dtypes


# In[13]:


#visualise taxi rides
import seaborn as sns
def showrides(df, numlines):
  lats = []
  lons = []
  goodrows = df[df['pickup_longitude'] < -70]
  for iter, row in goodrows[:numlines].iterrows():
    lons.append(row['pickup_longitude'])
    lons.append(row['dropoff_longitude'])
    lons.append(None)
    lats.append(row['pickup_latitude'])
    lats.append(row['dropoff_latitude'])
    lats.append(None)

  plt.plot(lons, lats)

showrides(df, 10)

##Some taxi trips are small and some are long 


# In[14]:


plt.figure(figsize = (14, 4))
n, bins, patches = plt.hist(df.fare_amount, 1000, facecolor='red', alpha=0.75)
plt.xlabel('Fare amount')
plt.title('Histogram of fare amount')
plt.xlim(0, 200)
plt.show();


# In[15]:


df.groupby('fare_amount').size().nlargest(10)


# In[15]:


plt.style.use("dark_background")
sns.distplot(df['passenger_count'],kde=False)
plt.title('Distribution of Passenger Count')
plt.show()


# In[16]:


df.groupby('passenger_count').size()


# In[17]:


sns.barplot(x = 'passenger_count',y ='trip_distance',data=df)


# In[18]:


x=df['store_and_fwd_flag'].value_counts()
x


# In[19]:


plt.style.use("classic")
plt.figure(figsize=(8,8))
plt.pie(x, colors=['lightgreen', 'lightcoral'], shadow=True, explode=[0.5,0], autopct='%1.2f%%', startangle=200)
plt.legend(labels=['Y','N'])
plt.title("Store and Forward Flag")


# In[20]:


df['store_and_fwd_flag']=df['store_and_fwd_flag'].apply(lambda x : 0 if x=='N' else 1)


# In[21]:


#Simple Rate calculation
#!pip install -U scikit-learn scipy matplotlib
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=42)


# In[22]:


import numpy as np
import shutil

def distance_between(lat1, lon1, lat2, lon2):
     # Haversine formula to compute distance 
  dist = np.degrees(np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1)))) * 60 * 1.515 * 1.609344
  return dist

def estimate_distance(df):
  return distance_between(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

def compute_rmse(actual, predicted):
  return np.sqrt(np.mean((actual - predicted)**2))

def print_rmse(df, rate, name):
  print("{1} RMSE = {0}".format(compute_rmse(df['fare_amount'], rate * estimate_distance(df)), name))


# In[23]:


rate = train['fare_amount'].mean() / estimate_distance(train).mean()

print("Rate = ${0}/km".format(rate))
print_rmse(train, rate, 'Train')
print_rmse(test, rate, 'Test')


# This clear shows us RootMeanSquareDeviation(RMSE) for test set is $276.997.

# ### Let us Explore the Date time and expand into weekday, month ,hour, minute to get a bigger picture while doing Analysis in our Data
# 

# In[ ]:





# In[24]:


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])


# In[25]:


#extracting day of week (mon-sun)
df['pickup_day'] = df['pickup_datetime'].dt.day_name()
df['dropoff_day'] = df['dropoff_datetime'].dt.day_name()
#extracting day of the weekday
df['pickup_day_no'] = df['pickup_datetime'].dt.weekday
df['dropoff_day_no'] = df['dropoff_datetime'].dt.weekday
#Extracting month of the month
df['p_month'] = df['pickup_datetime'].dt.month_name()
df['d_month'] = df['dropoff_datetime'].dt.month_name()
# Creating features based on Hour
df['pickup_by_hour'] = df['pickup_datetime'].dt.hour
df['dropoff_by_hour'] = df['dropoff_datetime'].dt.hour


# In[26]:


df['pickup_day'].value_counts()


# In[27]:


df['dropoff_day'].value_counts()


# In[29]:


def part_of_day (t):
    if t in range (6,12):
        return "Morning"
    elif t in range (12,16):
        return "Afternoon"
    elif t in range (16,22):
        return "Evening"
    else:
        return "Night"


# In[30]:


df['pickup_partofday'] = df['pickup_by_hour'].apply(part_of_day)
df['dropoff_partofday'] = df['dropoff_by_hour'].apply(part_of_day)


# In[31]:


df['pickup_partofday'].value_counts()


# In[32]:


df.head()


# In[23]:


pd.set_option("display.max_column", None)
pd.set_option("display.max_colwidth", None)

import dataframe_image as dfi



dfi.export(df.head(), "Datahead_newdataadded.png")


# In[33]:


#df['dropoff_partofday'].value_counts(normalize = True).plot(kind = 'bar')
df['pickup_partofday'].value_counts(normalize = True).plot(kind = 'bar')


#  Evening time is the most busiest Time followed by Night and then Morning which makes sense as most people leave from office in evening and and few in Night and people have to reach office in morning and very less people travel in afternoon that's in afternoon it's less busy.

# ### Trips Per day

# In[34]:


figure,(ax1,ax2)=plt.subplots(ncols=2,figsize=(20,5))
ax1.set_title('Pickup Days')
ax=sns.countplot(x="pickup_day",data=df,ax=ax1)
ax2.set_title('Dropoff Days')
ax=sns.countplot(x="dropoff_day",data=df,ax=ax2)


# In[35]:


# Creating two new features called pickup_part_of_day and dropoff_part_of_day.

df['pickup_part_of_day']=df['pickup_by_hour'].apply(part_of_day)
df['dropoff_part_of_day']=df['dropoff_by_hour'].apply(part_of_day)


# In[36]:


# Check to see if the formula has been applied correctly 

df[['pickup_part_of_day','dropoff_part_of_day']].head()
df


# In[38]:


plt.style.use("dark_background")
figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
sns.countplot(x='pickup_by_hour',data=df,ax=ax[0])
ax[0].set_title('The distribution of number of pickups on each hour of the day')
sns.countplot(x='dropoff_by_hour',data=df,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs on each hour of the day')
plt.tight_layout()
     


# In[41]:


plt.style.use("dark_background")
plt.figure(figsize=(20,10))
sns.heatmap(df.corr()*100, annot=True, cmap='inferno')
plt.title('Correlation Plot')


# ### TripDistance

# In[114]:


#a function is created to calculate the distance from latitudes and longitudes
from math import radians, cos, sin, asin, sqrt
def haversine(df):
    lat1, lon1, lat2, lon2 = df.pickup_latitude,df.pickup_longitude,df.dropoff_latitude,df.dropoff_longitude 
    R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c


# In[115]:


df['distance'] = df.apply(lambda x: haversine(x), axis = 1)


# In[116]:


sns.scatterplot(x='distance',y='trip_distance',data=df,color = 'black')


# In[120]:


print('The no of rows with distance =0 are {}'.format(len(df[df.distance==0])))


# That’s quite a number! We will not drop these rows. Rather we will replace these datas with the average distance

# In[121]:


mean_dist=df['distance'].mean()
df.loc[df['distance']==0,'distance']=mean_dist


# We will now create a new feature called speed. This will help us in identifying data points where time taken and distance covered does not match up. We will also have a look at the distribution of trip speed.

# In[122]:


df.columns
#del df['speed']


# In[123]:


#!pip install geopy
from geopy.distance import great_circle
def distance_trip(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
    start_coordinates = (pickup_longitude,pickup_latitude)
    stop_coordinates = (dropoff_longitude,dropoff_latitude)
    return great_circle(start_coordinates,stop_coordinates).km


# In[124]:


df['distance'] = df.apply(lambda x: distance_trip(x['pickup_latitude'],x['pickup_longitude'],x['dropoff_latitude'],x['dropoff_longitude']), axis=1)


# In[125]:


df.dtypes


# ### Let us Now see the Correlation b/w different variables in our Dataset
# 
# 

# In[61]:


numerical = df.select_dtypes(include =['int64','float64','Int64'])[:]
numerical.dtypes


# In[62]:


correlation = numerical.dropna().corr()
correlation


# In[66]:


c = numerical.corr().abs()
s = c.unstack()
so = s.sort_values(kind = 'quicksort', ascending = False)
so = pd.DataFrame(so,columns=['Pearson Coeficient'])


# In[67]:


so[so['Pearson Coeficient']<1].head(25)


# In[73]:


plt.figure(figsize=(20,6),dpi=140)
for j, i in enumerate(['pearson','kendall','spearman']):
    plt.subplot(1,3,j+1)
    correlation = numerical.dropna().corr(method=i)
    sns.heatmap(correlation, linewidth = 2)
    plt.title(i, fontsize = 18)


# Kendall,& Spearman correlation seem to have very similar pattern between them, except the slight variation in magnitude of correlation.
# * Too many variables with insignificant correlation.
# * Major correlation lies between the drop off hour and pickup hour.

# ## the fare of a taxi trip given information about pickup and drop off locations, the pick up date time and the number of the passengers travelling in New York.

# In[33]:


df.loc[df['fare_amount']>200].shape


# In[43]:


#New york latitude , longitude range
boundary={'min_lng':-74.5,
              'min_lat':40.7,
              'max_lng':-72.8, 
              'max_lat':41.8}


# In[45]:


##  consider locations within New York City
for long_value in ['pickup_longitude', 'dropoff_longitude']:
  df = df[(df[long_value] > boundary['min_lng']) & (df[long_value] < boundary['max_lng'])]
for lat_value in ['pickup_latitude', 'dropoff_latitude']:
  df = df[(df[lat_value] > boundary['min_lat']) & (df[lat_value] <boundary['max_lat'])]


# In[46]:


# plot histogram of fare
df[df['fare_amount']>100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare Amount $USD')
plt.title('Histogram');


# In[47]:


def distance_trip(latitude1,longitude1,latitude2,longitude2):
  r = 6373 # earth's radius
  latitude1 = np.deg2rad(latitude1)
  longitude1 = np.deg2rad(longitude1)
  latitude2 = np.deg2rad(latitude2)
  longitude2= np.deg2rad(longitude2)
  dlat = latitude2 - latitude1
  dlon = longitude2 - longitude1
  a = np.sin(dlat/2)**2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(dlon/2)**2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  distance = r*c
  return distance
def direction_angle(latitude1,longitude1,latitude2,longitude2):
  dlon = longitude2 - longitude1
  x = np.cos(latitude2)* np.sin(dlon)
  y= np.cos(latitude1)* np.sin(latitude2) - np.sin(latitude1)*np.cos(latitude2) * np.cos(dlon)
  beta_en_radians = np.arctan2(x,y)
  beta_en_degres = np.rad2deg(beta_en_radians)
  return beta_en_degres


# In[49]:


#distance away from  three major airports, Central park and Manhattan

airports = {'JFK': (-73.78,40.643),'LGA': (-73.87, 40.77),'EWR' : (-74.18, 40.69),'MNT':(-73.97,40.7831),'Cenpark':(-73.96,40.77)}
for airport in airports:
  df['pickup_dist_' + airport] = distance_trip(df['pickup_latitude'], df['pickup_longitude'], airports[airport][1], airports[airport][0])
  df['dropoff_dist_' + airport] = distance_trip(df['dropoff_latitude'], df['dropoff_longitude'], airports[airport][1], airports[airport][0])


# In[51]:


#trip distance
df['trip_distance'] = distance_trip(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])


# In[53]:


#trip direction north south east west
df['direction'] = direction_angle(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])


# In[54]:


df.head()


# In[55]:


df.trip_distance.hist(bins=50, figsize=(12,4))
plt.xlabel('distance per miles')
plt.title('Histogram ride distances in miles')


# In[56]:


df.groupby('passenger_count')['trip_distance', 'fare_amount'].mean()


# In[57]:


print("Average $USD/Mile(Km) : {:0.2f}".format(df.fare_amount.sum()/df.trip_distance.sum()))


# In[58]:


df.head()


# In[1]:


get_ipython().system('pip install pivottablejs')
from pivottablejs import pivot_ui
pivot_ui(df)


# In[32]:


sns.barplot(y='distance',x='vendor_id',data=df,estimator=np.mean)


# In[ ]:




