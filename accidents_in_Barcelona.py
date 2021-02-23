'''
Students:       Karina Jonina - 10543032
                Pauline Lloyd - 10525563
Module:         B8IT107
Module Name:    Data Visualisation
Assignment:     60% 

The dataset used was collected from the following website:
https://www.kaggle.com/xvivancos/barcelona-data-sets

This datasets is created by the local police in the city of Barcelona. 
Incorporates the number of injuries by severity, the number of vehicles
and the point of impact.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import HeatMap
import datetime
import matplotlib.ticker

# read the CSV file
df = pd.read_csv('accidents_2017.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking the df shape
print('The df has {} rows and {} columns.'.format(*df.shape))
#The df has 10339 rows and 15 columns.

# prints out names of columns
print(df.columns)

# This tells us which variables are object, int64 and float 64. This would mean that 
# some of the object variables might have to be changed into a categorical variables and int64 to float64 
# depending on our analysis.
print(df.info())

# checking for counts data and gives Mean, Sd and quartiles for all columns
print(df.describe())

# checking for missing data
pd.isna(df)
# not very useful as it is not possible to see all the columns

# checking for missing data
print('Nan in each columns' , df.isna().sum(), sep='\n')

# Looking for all the unique values in all the columns
column = df.columns
for i in column:
    print('\n',i,'\n',df[i].unique(),'\n','-'*80)
    
# making object into categorical variables
df['District Name'] = df['District Name'].astype('category')
df['Neighborhood Name'] = df['Neighborhood Name'].astype('category')
df['Street'] = df['Street'].astype('category')
df['Month'] = df['Month'].astype('category')
df['Weekday'] = df['Weekday'].astype('category')
df['Part of the day'] = df['Part of the day'].astype('category')

# checking data to check that all objects have been changed to categorical variables.
df.info()

# =============================================================================
# Creating a Date column from Day, Month and Year
# =============================================================================
#creating a column with the correct year
df['Year'] = 2017

# concating the columns
df['Date'] = df['Year'].astype(str) + ' ' + df['Month'].astype(str).str.zfill(2) + ' ' + df['Day'].astype(str).str.zfill(2)

# parsing using datetime so that Python knows this column is a date. 
df['Date'] = pd.to_datetime(df['Date'], format='%Y %B %d')

#checking date 
print(df['Date'])

#checking the type of the dataset
df.info()

# =============================================================================
# Creating a Date column from Day, Month, Year, and Hour
# =============================================================================
# concating the columns
df['Full_Date'] = df['Year'].astype(str) + ' ' + df['Month'].astype(str).str.zfill(2) + ' ' + df['Day'].astype(str).str.zfill(2)+ ' ' + df['Hour'].astype(str).str.zfill(2) + ':00'

# parsing using datetime so that Python knows this column is a date. 
df['Full_Date'] = pd.to_datetime(df['Full_Date'])

print(df['Full_Date'].head())

# =============================================================================
# Creating a new dataset with dates as index
# =============================================================================
# PERSONAL PREFERENCE: I prefer the columns' names to be in lower case and connect words with '_'
# takes too long to type capitals and hard to read without '_'
# copying and pasting is easier this way too

#renaming of the columns
df = pd.DataFrame({'date': df['Date'],
                   'full_date': df['Full_Date'],
                   'id': df['Id'],
                   'district_name': df['District Name'],
                   'neighborhood_name': df['Neighborhood Name'],
                   'street': df['Street'],
                   'weekday': df['Weekday'],
                   'month': df['Month'],
                   'day': df['Day'],
                   'hour': df['Hour'],
                   'part_day': df['Part of the day'],
                   'mild_inj': df['Mild injuries'],
                   'ser_inj': df['Serious injuries'],
                   'victims': df['Victims'],
                   'vehicles': df['Vehicles involved'],
                   'longitude': df['Longitude'],
                   'latitude': df['Latitude']})

#checking new columns' names  
df.info()

#sorting the dataset by full_date. 
#This will ensure the dataset is in the chronological order
df.sort_values(by=['full_date'], inplace=True, ascending=True)

# making the full_date the index
df = df.set_index(pd.DatetimeIndex(df['full_date']))

print(df.index)

# Dropping full_date variables due to the fact that it is an index
df.drop('full_date',axis=1,inplace=True)

# =============================================================================
# Dropping rows with Unknown in the District Name
# =============================================================================
#dropping rows where the district name is unknown
df = df[df['district_name'] != 'Unknown']

#dropping rows where there are no vehicles involved
df = df[df['vehicles'] != 0]

print('After dropping Unknown District Names and Zero Vehicles Involved, the df has {} rows and {} columns.'.format(*df.shape))
#After dropping Unknown District Names and Zero Vehicles Involved, the df has 10307 rows and 16 columns.

df['district_name'] = df['district_name'].cat.remove_categories(['Unknown'])
df['district_name'].unique()

# =============================================================================
# Creating new variable: Low / High Season
# =============================================================================
# changing the date into just date
df['date'] =  df['date'].dt.date

# finding the low / high season by date
df.loc[df['date'] < datetime.date(2017,4,1), 'season'] = 'Low Season'
df.loc[df['date'] >= datetime.date(2017,4,1), 'season'] = 'High Season'
df.loc[df['date'] >= datetime.date(2017,9,30), 'season'] = 'Low Season'

# making object into categorical variables
df['season'] = df['season'].astype('category')


# =============================================================================
# Creating new variable: Low / High Season
# =============================================================================
# trying to create a new variable stating whether it was the weekend or weekday

df.loc[df['weekday'].str.contains('Monday'), 'weekend?'] = 'Weekday'
df.loc[df['weekday'].str.contains('Tuesday'), 'weekend?'] = 'Weekday'
df.loc[df['weekday'].str.contains('Wednesday'), 'weekend?'] = 'Weekday'
df.loc[df['weekday'].str.contains('Thursday'), 'weekend?'] = 'Weekday'
df.loc[df['weekday'].str.contains('Friday'), 'weekend?'] = 'Weekday'

df.loc[df['weekday'].str.contains('Saturday'), 'weekend?'] = 'Weekend'
df.loc[df['weekday'].str.contains('Sunday'), 'weekend?'] = 'Weekend'

# =============================================================================
# Cleaning Streets Data
# =============================================================================
df['street'].unique()

# making sure the streets are all Title Case
df['street'] = df['street'].str.title()

# splitting the string data with ' / '
df['street_split_1'] = df['street'].str.split(' / ').str[0]
df['street_split_1'] = df['street_split_1'].str.split('  ').str[0]


# =============================================================================
# Creating variable called Road Type: Intersection or Road
# =============================================================================
# trying to create the all the names with Intersectiuon and create  a new variable called Road Type
df.loc[df['street'].str.contains('/'), 'road_type'] = 'Intersection'

# Trying to create  Straight Road in Road Type
df.loc[df['road_type'].isnull(), 'road_type'] = 'Straight Road'

# making object into categorical variables
df['road_type'] = df['road_type'].astype('category')

# =============================================================================
# Creating variable called Accident Type
# =============================================================================
# Trying to create accident type
df.loc[df['ser_inj'] > 0 , 'accident_type'] = 'One or More Serious Injuries'

# Trying to create accident type
df.loc[df['accident_type'].isnull(), 'accident_type'] = 'No Serious Injuries'

# making object into categorical variables
df['accident_type'] = df['accident_type'].astype('category')

# =============================================================================
# Reordering Months
# =============================================================================
#looking at the month type and order
df['month'].dtype

# Reordering months so that it starts with 'January' and proceed in the correct order
df['month'] = df['month'].cat.reorder_categories(['January', 'February', 'March', 'April', 'May', 'June',
'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)

df['month'] = df['month'].replace('January','Jan')
df['month'] = df['month'].replace('February','Feb')
df['month'] = df['month'].replace('March','Mar')
df['month'] = df['month'].replace('April','Apr')
df['month'] = df['month'].replace('June','Jun')
df['month'] = df['month'].replace('July','Jul')

df['month'] = df['month'].replace('August','Aug')
df['month'] = df['month'].replace('September','Sep')
df['month'] = df['month'].replace('October','Oct')
df['month'] = df['month'].replace('November','Nov')
df['month'] = df['month'].replace('December','Dec')

# =============================================================================
# Reordering weekend
# =============================================================================    
df['weekday'].dtype
# Reordering weekdays so that it starts with 'Monday' and proceed in the correct order
df['weekday'] = df['weekday'].cat.reorder_categories(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
'Sunday'], ordered=True)  

    
df['weekday'] = df['weekday'].replace('Monday','Mon')
df['weekday'] = df['weekday'].replace('Tuesday','Tue')
df['weekday'] = df['weekday'].replace('Wednesday','Wed')
df['weekday'] = df['weekday'].replace('Thursday','Thu')
df['weekday'] = df['weekday'].replace('Friday','Fri')
df['weekday'] = df['weekday'].replace('Saturday','Sat')
df['weekday'] = df['weekday'].replace('Sunday','Sun')
    
df['hour'].unique()


# Putting hour in the correct order
df['hour'] = pd.Categorical(df['hour'], 
       categories = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
       ordered = True)


df['part_day'] = pd.Categorical(df['part_day'], 
       categories = ['Morning','Afternoon','Night'],
       ordered = True)
# =============================================================================
# Default Sizing
# =============================================================================
titlefont = 14
labelfont = 12
default_size = plt.figure(figsize = (12, 8))

# =============================================================================
# Setting Color for each variable
# =============================================================================
pal = sns.color_palette('Set2',5)
pal.as_hex()

c_ser_inj = '#e78ac3'
c_mild_inj = '#8da0cb'
c_vehicle = '#fc8d62'
c_victim = '#66c2a5'
c_accident = '#a6d854'

c_high_season = '#a6cee3'
c_low_season = '#b2df8a' 

c_straight = '#fccde5'
c_intersection = '#cab2d6'

c_no_serious = '#66c2a5'
c_serious = '#fc8d62'

c_weekday = '#fb9a99'
c_weekend = '#fdbf6f'
# =============================================================================
# Examining the data
# =============================================================================
victims = df['victims'].sum()
mild_inj = df['mild_inj'].sum()
seri_inj = df['ser_inj'].sum()
vehi = df['vehicles'].sum()

unknown = victims - (mild_inj+seri_inj)
total = mild_inj+seri_inj

print('Total Number of vehicles involved: ',vehi)
print('Total Number of Victims: ',victims)
print('Total Number of Victims with mild injuries: ',mild_inj)
print('Total Number of Victims with serious injuries: ',seri_inj)
print('Total Number of Victims with any injuries: ',total)
print('Total Number of Victims with unknown information ',unknown)


# recalculating victims column
df['new_victims'] = df['ser_inj'] + df['mild_inj']
victims = df['new_victims'].sum()

new_unknown = victims - (mild_inj+seri_inj)
new_total = mild_inj+seri_inj

print('Checking the new Total Number of vehicles involved: ',vehi)
print('Checking the new Total Number of Victims: ',victims)
print('Checking the Number of Victims with any injuries: ', new_total )
print('Checking the new Total Number of Victims with unknown information ',new_unknown )



## =============================================================================
## Save the Dataset in a new CSV as we have new columns 
## =============================================================================
#df.to_csv('df_barcelona.csv')

## =============================================================================
## simple Maps with Longitude and Latitude
## =============================================================================
## Plot the locations of all accidents in Barcelona
#plt.figure(figsize = (20, 20))
#sns.relplot(x = 'longitude', 
#            y = 'latitude', 
#            hue = 'district_name', 
#            data = df, 
#            kind = 'scatter', 
#            palette = 'Set2')
#plt.title('Accidents by District')
#
## Show the plot
#plt.show()



# =============================================================================
# Looking at heatmap of accidents
# =============================================================================

coordinates = [41.406141, 2.168594]

map_acc = folium.Map(location=coordinates,
                    zoom_start = 13)

df_cor = df[['latitude','longitude']]
cor = [[row['latitude'],row['longitude']] for index,row in df_cor.iterrows()]

HeatMap(cor, min_opacity=0.5, radius=14).add_to(map_acc)
map_acc

#saving the map as a html
map_acc.save('map_acc.html') 

# =============================================================================
# Examining The Accident Trend for Each Day (4,1)
# =============================================================================

#create dataset examining how many times each date was mentions
df_ac= df.groupby('date').size().to_frame('size').reset_index()
print(df_ac)

#create dataset examining vehicles, mild and serious injuriesby date
df_date = df.groupby(['date'])['vehicles','mild_inj','ser_inj'].sum().reset_index()
print(df_date)

#adding the size column to DATE  summary 
df_date['size'] = df_ac['size']

#order columns
df_date = df_date[['date',  'vehicles','mild_inj','size', 'ser_inj']]

# viewing the date
print(df_date)


sns.set_style('darkgrid')

fig, ax = plt.subplots(4,figsize=(8, 10))
sns.lineplot(ax = ax[0], 
             x = 'date', 
             y = 'vehicles', 
             data = df_date, 
             color = c_vehicle)
ax[0].set_ylabel(' # of Vehicles', fontsize = labelfont)
ax[0].set_title('Vehicles', fontsize = titlefont)
ax[0].set_xlabel(' ')
ax[0].set(xticklabels=[])

sns.lineplot(ax = ax[1], 
             x ='date', 
             y = 'size', data = df_date, 
             color = c_accident)
ax[1].set_ylabel('# of Accidents', fontsize = labelfont)
ax[1].set_title( 'Accidents', fontsize = titlefont)
ax[1].set_xlabel(' ')
ax[1].set(xticklabels=[])

sns.lineplot(ax = ax[2], 
             x ='date', 
             y = 'mild_inj', 
             data = df_date, 
             color = c_mild_inj)
ax[2].set_ylabel('# of mild injuries', fontsize = labelfont)
ax[2].set_title('Mild Injuries', fontsize = titlefont)
ax[2].set_xlabel(' ')
ax[2].set(xticklabels=[])

sns.lineplot(ax = ax[3],
             x ='date', 
             y = 'ser_inj', 
             data = df_date, 
             color = c_ser_inj)
ax[3].set_ylabel('#  of serious injuries', fontsize = labelfont)
ax[3].set_title( 'Serious Injuries ', fontsize = titlefont)
ax[3].set_xlabel(' ')

plt.suptitle('Accident Trend',size=16)
plt.show()


# =============================================================================
# LOGARITHMIC SCALE  - > Examining Distributions of Numeric Variables'
# =============================================================================
print(df.groupby('mild_inj').size())

print(df.groupby('ser_inj').size())

print(df.groupby('vehicles').size())

print(df.groupby('new_victims').size())



sns.set_style('darkgrid')
fig, ax = plt.subplots(2,2, figsize=(12, 8))

# Plot the distribution of vehicles
ax[0,0].hist(df['vehicles'], bins = 15, color = c_vehicle)
ax[0,0].set_title('Distribution of Vehicles', fontsize = 14)
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel('# of Accidents', fontsize = 12)
ax[0,0].set_xticks(np.arange(0, 15, 1))
ax[0,0].yaxis.grid(True) # Hide the horizontal gridlines
ax[0,0].xaxis.grid(False)


# Plot the distribution of victims
ax[0,1].hist(df['new_victims'], bins = 10, color = c_victim)
ax[0,1].set_title ('Distribution of Victims', fontsize = 14)
ax[0,1].set_yscale('log')
ax[0,1].set_ylabel('# of Accidents', fontsize = 12)
ax[0,1].set_xticks(np.arange(0, 11, 1))
ax[0,1].yaxis.grid(True) # Hide the horizontal gridlines
ax[0,1].xaxis.grid(False)


# Plot the distribution of mildinjuries
ax[1,0].hist(df['mild_inj'],bins = 11, color = c_mild_inj)
ax[1,0].set_title('Distribution of Mild Injuries', fontsize = 14)
ax[1,0].set_ylabel('# of Accidents', fontsize = 12)
ax[1,0].set_yscale('log')
ax[1,0].set_xticks(np.arange(0, 11, 1))
ax[1,0].yaxis.grid(True) # Hide the horizontal gridlines
ax[1,0].xaxis.grid(False)

# Plot the distribution of serious injuries
ax[1,1].hist(df['ser_inj'], bins = 4, color = c_ser_inj)
ax[1,1].set_title('Distribution of Serious Injuries', fontsize = 14)
locator = matplotlib.ticker.MultipleLocator(2)
ax[1,1].xaxis.set_major_locator(locator)
ax[1,1].set_ylabel('# of Accidents', fontsize = 12)
ax[1,1].set_yscale('log')
ax[1,1].set_xticks(np.arange(0, 5, 1))
ax[1,1].yaxis.grid(True) # Hide the horizontal gridlines
ax[1,1].xaxis.grid(False)

plt.suptitle('Distributions of Accident Trends', fontsize = titlefont)
# Display the plot
plt.show()

''' showing example without logarithmic scale '''
#fig, ax = plt.subplots(figsize=(10, 6))
## Plot the distribution of victims
#plt.hist(df['vehicles'], bins = 15, color = c_victim)
#plt.title('Distribution of Vehicles', fontsize = 14)
#plt.ylabel('# of Accidents (Normal Scale)', fontsize = 14) 
## Display the plot
#plt.show()


# =============================================================================
# VIOLIN PLOTS OF VEHICLES, VICTIM, SERIOUS AND MILD INJURIES 
# =============================================================================
sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(3,1, figsize=(18,12))
sns.violinplot(ax = ax[0], data = df,
         x='district_name',
         y='vehicles',
         palette='Set2')
ax[0].set_ylabel('# of Vehicles', fontsize = labelfont)
ax[0].set_xlabel('', fontsize = labelfont)
ax[0].set_title('Vehicles per Accident', fontsize = titlefont)
ax[0].set(xticklabels=[])

sns.violinplot(ax = ax[1], data = df,
         x='district_name',
         y='mild_inj',
         palette='Set2')
ax[1].set_ylabel('# of Mild Injuries', fontsize = labelfont)
ax[1].set_xlabel('', fontsize = labelfont)
ax[1].set_title('Mild Injuries per Accident', fontsize = titlefont)
ax[1].set(xticklabels=[])

sns.violinplot(ax = ax[2], data = df,
         x='district_name',
         y='ser_inj',
         palette='Set2')
ax[2].set_ylabel('# of Serious Injuries', fontsize = labelfont)
ax[2].set_xlabel('', fontsize = labelfont)
ax[2].set_title('Serious Injuries per Accident', fontsize = titlefont)
ax[2].set_xticklabels(df['district_name'].unique(), rotation = 45, fontsize = 12)
ax[2].set(xticklabels=[])
locator = matplotlib.ticker.MultipleLocator(2)
ax[2].yaxis.set_major_locator(locator)
ax[2].set_xticklabels(df['district_name'].unique(),  rotation = 45, fontsize = labelfont)



plt.suptitle('Trends per Accident by District')
plt.show() 

# =============================================================================
# Accidents in each Districts
# =============================================================================
district_ac = df.groupby('district_name').size().to_frame('size').reset_index()
print(district_ac)

district_sum = df.groupby(['district_name'])['vehicles' ,'mild_inj','ser_inj'].sum().reset_index()

#adding the size column to district  summary 
district_sum['size'] = district_ac['size']

#order columns
district_sum = district_sum[['district_name', 'vehicles','mild_inj','size', 'ser_inj']]

#viewing new dataset
print(district_sum)



'''the barchart below was viewed as uninformative
Feel free to unhash and view'''
#setting style
#creating a 2x2 BAR CHARTS of Weekday
#fig, ax = plt.subplots(2,2, figsize=(12, 10))
#
#sns.barplot(ax = ax[0,0],
#            x = 'district_name', 
#            y = 'vehicles', 
#            data = district_sum, 
#            color = c_vehicle)
#ax[0,0].set_ylabel(' # of vehicles', fontsize = labelfont)
#ax[0,0].set_title( 'Vehicles', fontsize = titlefont)
#ax[0,0].set_xticklabels(district_sum['district_name'].unique(), rotation = 25, fontsize = labelfont)
#ax[0,0].set_xlabel(' ')
#
#sns.barplot(ax = ax[0,1], 
#            x = 'district_name', 
#            y = 'size', 
#            data = district_sum, 
#            color = c_accident)
#ax[0,1].set_ylabel('# of Accidents', fontsize = labelfont)
#ax[0,1].set_title( 'Accidents', fontsize = titlefont)
#ax[0,1].set_xticklabels(district_sum['district_name'].unique(), rotation = 25, fontsize = labelfont)
#ax[0,1].set_xlabel(' ')
#
#sns.barplot(ax = ax[1,0],
#            x = 'district_name', 
#            y = 'mild_inj', 
#            data = district_sum,  
#            color = c_mild_inj)
#ax[1,0].set_ylabel('# of mild injuries', fontsize = labelfont)
#ax[1,0].set_title( 'Mild Injuries', fontsize = titlefont)
#ax[1,0].set_xlabel(' ')
#ax[1,0].set_xticklabels(district_sum['district_name'].unique(), rotation = 25, fontsize = labelfont)
#
#plt.suptitle('Accident Trend by Weekday in Eixample',size=16)
#
#
#sns.barplot(ax = ax[1,1],
#            x = 'district_name', 
#            y = 'ser_inj',
#            data = district_sum,  
#            color = c_ser_inj)
#ax[1,1].set_ylabel('# of serious injuries', fontsize = labelfont)
#ax[1,1].set_title( 'Serious Injuries', fontsize = titlefont)
#ax[1,1].set_xticklabels(district_sum['district_name'].unique(), rotation = 25, fontsize = labelfont)
#
#ax[1,1].set_xlabel(' ')
#plt.show()
#
#


'''the graph below was viewed as uninformative
Feel free to unhash and view'''
## setting
#sns.set_style('darkgrid')
#fig, ax = plt.subplots(figsize = (12, 8))
#ax.plot('district_name', 'vehicles', data = district_sum, color = c_vehicle)
#ax.plot('district_name', 'mild_inj', data = district_sum,color = c_mild_inj)
#ax.plot('district_name', 'size', data = district_sum,color= c_accident)
#ax.plot('district_name', 'ser_inj', data = district_sum, color = c_ser_inj)
#ax.set_ylabel('Quantity.', fontsize = labelfont)
#ax.set_xlabel('', fontsize = labelfont)
#ax.set_title('Accident Trends By District', fontsize = titlefont)
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#L = ax.legend()
#L.get_texts()[0].set_text('Vehicles')
#L.get_texts()[1].set_text('Mild Injuries')
#L.get_texts()[2].set_text('Accidents')
#L.get_texts()[3].set_text('Serious Injuries')
#ax.set_xticklabels(district_sum['district_name'].unique(), rotation = 45, fontsize = labelfont)
#plt.show()


#creating a new dataset 
district_sum1 = pd.melt(district_sum, id_vars = "district_name")
print(district_sum1)


''' THE HORIZONTAL LINE WAS NOT PREFERED
BECAUSE IT IS EASIER TO READ DISTRICTS'''
## setting
#sns.set_style('darkgrid')
##building 
#fig, ax = plt.subplots(figsize = (12, 8))
#g = sns.barplot(x = 'district_name', y = 'value', 
#               hue = 'variable',
#               data = district_sum1,
#               palette = [c_vehicle, c_mild_inj, c_accident, c_ser_inj])
#L = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#L.get_texts()[0].set_text('Vehicles')
#L.get_texts()[1].set_text('Mild Injuries')
#L.get_texts()[2].set_text('Accidents')
#L.get_texts()[3].set_text('Serious Injuries')
#g.set_xticklabels(district_sum['district_name'].unique(), rotation = 45, fontsize = labelfont)
#g.set_xlabel(' ')
#g.set_ylabel('Quantity', fontsize = labelfont)
#g.set_title('Accident Trends by District' , fontsize = titlefont)
#plt.show()


# setting
sns.set_style('darkgrid')

#building 
fig, ax = plt.subplots(figsize = (8, 12))
g = sns.barplot(y = 'district_name', x = 'value', 
               hue = 'variable',
               data = district_sum1,
               palette = [c_vehicle, c_mild_inj, c_accident, c_ser_inj])
L = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
L.get_texts()[0].set_text('Vehicles')
L.get_texts()[1].set_text('Mild Injuries')
L.get_texts()[2].set_text('Accidents')
L.get_texts()[3].set_text('Serious Injuries')
g.set_ylabel(' ')
g.set_xlabel('Quantity', fontsize = labelfont)
g.set_title('Accident Trends by District' , fontsize = 18)
plt.show()




# =============================================================================
# Accidents in each Districts by Month
# =============================================================================
#creating a new dataset for SEABORN to operate quickly
month_district_ac = df.groupby(['month','district_name']).size().to_frame('size').reset_index()
print(month_district_ac)

month_district_sum = df.groupby(['month','district_name'])['vehicles','mild_inj','ser_inj'].sum().reset_index()

#adding the size column to district  summary 
month_district_sum['size'] = month_district_ac['size']

#order columns
month_district_sum = month_district_sum[['month', 'district_name', 'vehicles','mild_inj','size', 'ser_inj']]

print(month_district_sum)

# setting
sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18, 10))
sns.lineplot(ax = ax[0,0], 
             x = 'month', 
             y = 'vehicles', 
             data = month_district_sum,
             hue = 'district_name',
             palette = 'Set2')
ax[0,0].set_ylabel('# of Vehicles', fontsize = labelfont)
ax[0,0].set_xlabel('', fontsize = labelfont)
ax[0,0].set_title('Vehicles', fontsize = titlefont)
ax[0,0].legend([],[], frameon=False)
ax[0,0].yaxis.grid(False) # Hide the horizontal gridlines
ax[0,0].xaxis.grid(True)



sns.lineplot(ax = ax[0,1],
             x = 'month', 
             y = 'size', 
             data = month_district_sum,
             hue = 'district_name',
             palette = 'Set2')
ax[0,1].set_ylabel('# of Accidents', fontsize = labelfont)
ax[0,1].set_xlabel('', fontsize = labelfont)
ax[0,1].set_title('Accidents', fontsize = titlefont)
ax[0,1].legend([],[], frameon=False)
ax[0,1].yaxis.grid(False) # Hide the horizontal gridlines
ax[0,1].xaxis.grid(True)


sns.lineplot(ax = ax[1,0],
             x = 'month', 
             y = 'mild_inj',
             data = month_district_sum,
             hue ='district_name',
             palette = 'Set2')
ax[1,0].set_ylabel('# of Mild Injuries', fontsize = labelfont)
ax[1,0].set_xlabel('', fontsize = labelfont)
ax[1,0].set_title('Mild Injuries', fontsize = titlefont)
ax[1,0].legend([],[], frameon=False)
ax[1,0].yaxis.grid(False) # Hide the horizontal gridlines
ax[1,0].xaxis.grid(True)


sns.lineplot(ax = ax[1,1],
             x = 'month', 
             y = 'ser_inj',
             data = month_district_sum,
             hue ='district_name',
             palette = 'Set2')
ax[1,1].set_ylabel('# of Serious Injuries', fontsize = labelfont)
ax[1,1].set_xlabel('', fontsize = labelfont)
ax[1,1].set_title('Serious Injuries', fontsize = titlefont)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax[1,1].yaxis.grid(False) # Hide the horizontal gridlines
ax[1,1].xaxis.grid(True)



plt.suptitle('Accident Trends by Month and District', fontsize = titlefont)

plt.show()

# =============================================================================
# COMPARING TO DISTRICTS TO ANNUAL AVERAGE
# =============================================================================

#creating a copy of district_sum so that the graph can work without issues
district_sum_copy = district_sum

district_sum_copy['mild_diff'] = district_sum['mild_inj'] - district_sum['mild_inj'].mean()
district_sum_copy['ser_diff'] = district_sum['ser_inj'] - district_sum['ser_inj'].mean()
district_sum_copy['veh_diff'] = district_sum['vehicles'] - district_sum['vehicles'].mean()
district_sum_copy['size_diff'] = district_sum['size'] - district_sum['size'].mean()

print(district_sum_copy)


sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18,12), sharey = True)
sns.barplot(ax = ax[0,0], 
            y = 'district_name', x = 'veh_diff', 
            data = district_sum_copy,
            palette='Set2')
ax[0,0].set_xlabel('# of Vehicles', fontsize = labelfont)
ax[0,0].set_ylabel('', fontsize = labelfont)
ax[0,0].set_title('Comparing Vehicles to the Average', fontsize = titlefont)

sns.barplot(ax = ax[0,1], y = 'district_name', x = 'size_diff', 
               data = district_sum_copy,
               palette='Set2')
ax[0,1].set_xlabel('# of Accidents')
ax[0,1].set_ylabel(' ')
ax[0,1].set_title('Comparing Accidents to the Average', fontsize = titlefont)


sns.barplot(ax = ax[1,0], 
            data = district_sum_copy,
            y='district_name',
            x='mild_diff',
            palette='Set2')
ax[1,0].set_xlabel('# of Mild Injuries', fontsize = labelfont)
ax[1,0].set_ylabel(' ', fontsize = labelfont)
ax[1,0].set_title('Comparing  Mild Injuries to the Average', fontsize = titlefont)


sns.barplot(ax = ax[1,1], 
            data = district_sum_copy,
            y='district_name',
            x='ser_diff',
            palette='Set2')
ax[1,1].set_xlabel('# of Serious Injuries', fontsize = labelfont)
ax[1,1].set_ylabel(' ', fontsize = labelfont)
ax[1,1].set_title('Comparing Serious Injuries to the Average', fontsize = titlefont)
plt.suptitle('Comparing Districts to Averages', fontsize = 18)

plt.show()


# =============================================================================
# TABLEAU - > Replicate where  Eixample is in orange and 
# other graphs are in greay
# =============================================================================

sns.set_style('white')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18, 10))
sns.lineplot(ax = ax[0,0], 
             x = 'month', 
             y = 'vehicles', 
             data = month_district_sum,
             hue = 'district_name',
             palette = ['grey', c_vehicle, 'grey','grey','grey','grey','grey','grey','grey','grey'])
ax[0,0].set_ylabel('# of Vehicles', fontsize = labelfont)
ax[0,0].set_xlabel('', fontsize = labelfont)
ax[0,0].set_title('Vehicles', fontsize = titlefont)
ax[0,0].legend([],[], frameon=False)
ax[0,0].yaxis.grid(False) # Hide the horizontal gridlines
ax[0,0].xaxis.grid(False)



sns.lineplot(ax = ax[0,1],
             x = 'month', 
             y = 'size', 
             data = month_district_sum,
             hue = 'district_name',
             palette = ['grey', c_accident, 'grey','grey','grey','grey','grey','grey','grey','grey'])
ax[0,1].set_ylabel('# of Accidents', fontsize = labelfont)
ax[0,1].set_xlabel('', fontsize = labelfont)
ax[0,1].set_title('Accidents', fontsize = titlefont)
ax[0,1].legend([],[], frameon=False)
ax[0,1].yaxis.grid(False) # Hide the horizontal gridlines
ax[0,1].xaxis.grid(False)


sns.lineplot(ax = ax[1,0],
             x = 'month', 
             y = 'mild_inj',
             data = month_district_sum,
             hue ='district_name',
             palette = ['grey', c_mild_inj, 'grey','grey','grey','grey','grey','grey','grey','grey'])
ax[1,0].set_ylabel('# of Mild Injuries', fontsize = labelfont)
ax[1,0].set_xlabel('', fontsize = labelfont)
ax[1,0].set_title('Mild Injuries', fontsize = titlefont)
ax[1,0].legend([],[], frameon=False)
ax[1,0].yaxis.grid(False) # Hide the horizontal gridlines
ax[1,0].xaxis.grid(False)


sns.lineplot(ax = ax[1,1],
             x = 'month', 
             y = 'ser_inj',
             data = month_district_sum,
             hue ='district_name',
             palette = ['grey', c_ser_inj, 'grey','grey','grey','grey','grey','grey','grey','grey'])
ax[1,1].set_ylabel('# of Serious Injuries', fontsize = labelfont)
ax[1,1].set_xlabel('', fontsize = labelfont)
ax[1,1].set_title('Serious Injuries', fontsize = titlefont)

ax[0,0].legend([],[], frameon=False)
ax[0,1].legend([],[], frameon=False)
ax[1,0].legend([],[], frameon=False)
ax[1,1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax[1,1].yaxis.grid(False) # Hide the horizontal gridlines
ax[1,1].xaxis.grid(False)

plt.suptitle('Accident Trends by Month and District', fontsize = titlefont)

plt.show()

# =============================================================================
# TABLEAU -> COMPARING DISTRICTS TO ANNUAL MEAN 
# =============================================================================

#creating a copy of district_sum so that the graph can work without issues
district_sum_copy = district_sum

district_sum_copy['mild_diff'] = district_sum['mild_inj'] - district_sum['mild_inj'].mean()
district_sum_copy['ser_diff'] = district_sum['ser_inj'] - district_sum['ser_inj'].mean()
district_sum_copy['veh_diff'] = district_sum['vehicles'] - district_sum['vehicles'].mean()
district_sum_copy['size_diff'] = district_sum['size'] - district_sum['size'].mean()

print(district_sum_copy)


sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18,12), sharey = True)
sns.barplot(ax = ax[0,0], 
            y = 'district_name', x = 'veh_diff', 
            data = district_sum_copy,
            palette=['grey', 'orange', 'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey'])
ax[0,0].set_xlabel('# of Vehicles', fontsize = labelfont)
ax[0,0].set_ylabel('', fontsize = labelfont)
ax[0,0].set_title('Comparing Vehicles to the Average', fontsize = titlefont)
#ax[0,0].set(xticklabels=[])
#ax[0,0].set_yticklabels(district_sum['district_name'].unique(), fontsize = 12)

sns.barplot(ax = ax[0,1], y = 'district_name', x = 'size_diff', 
               data = district_sum_copy,
               palette=['grey', 'orange', 'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey'])
ax[0,1].set_xlabel('# of Accidents')
ax[0,1].set_ylabel(' ')
ax[0,1].set_title('Comparing Accidents to the Average', fontsize = titlefont)

sns.barplot(ax = ax[1,0], 
            data = district_sum_copy,
            y='district_name',
            x='mild_diff',
            palette=['grey', 'orange', 'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey'])
ax[1,0].set_xlabel('# of Mild Injuries', fontsize = labelfont)
ax[1,0].set_ylabel(' ', fontsize = labelfont)
ax[1,0].set_title('Comparing  Mild Injuries to the Average', fontsize = titlefont)


sns.barplot(ax = ax[1,1], 
            data = district_sum_copy,
            y='district_name',
            x='ser_diff',
            palette=['grey', 'orange', 'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey'])
ax[1,1].set_xlabel('# of Serious Injuries', fontsize = labelfont)
ax[1,1].set_ylabel(' ', fontsize = labelfont)
ax[1,1].set_title('Comparing Serious Injuries to the Average', fontsize = titlefont)
#ax[1,1].set_yticklabels(district_sum['district_name'].unique(), fontsize = 12)

#ax[1,1].set(yticklabels=[])
plt.suptitle('Comparing Districts to Averages', fontsize = 18)

plt.show()


# =============================================================================
'''FOCUSING IN EIXAMPLE'''
# =============================================================================



# =============================================================================
# Creating a new dataset  -> Examining only EIXAMPLE 
# =============================================================================
df_Eixample = df[df['district_name'] == 'Eixample']

print('The df has {} rows and {} columns.'.format(*df_Eixample.shape))
#The df has 3029 rows and 21 columns.




# =============================================================================
# Looking at heatmap of accidents
# =============================================================================

coordinates = [41.406141, 2.168594]

df_Eixample_map_acc = folium.Map(location=coordinates,
                    zoom_start = 13)

df_Eixample_cor = df_Eixample[['latitude','longitude']]
cor = [[row['latitude'],row['longitude']] for index,row in df_Eixample_cor.iterrows()]

HeatMap(cor, min_opacity=0.5, radius=14).add_to(df_Eixample_map_acc)
df_Eixample_map_acc

#saving the map as a html
df_Eixample_map_acc.save('map_acc.html') 



# =============================================================================
# Examining The Accident Trend for Each Day (4,1)
# =============================================================================

df_Eixample_ac= df_Eixample.groupby('date').size().to_frame('size').reset_index()
print(df_Eixample_ac)
df_Eixample_date = df_Eixample.groupby(['date'])['vehicles','mild_inj','ser_inj', 'new_victims'].sum().reset_index()
print(df_Eixample_date)
#adding the size column to DATE  summary 
df_Eixample_date['size'] = df_Eixample_ac['size']
#order columns
df_Eixample_date = df_Eixample_date[['date',  'vehicles', 'new_victims','mild_inj','size', 'ser_inj']]

#Examining the relationship between weekend and road_type on number of accidents
print(df_Eixample_date)

sns.set_style('darkgrid')
fig, ax = plt.subplots(4,figsize=(8, 10))
sns.lineplot(ax = ax[0], 
             x = 'date', 
             y = 'vehicles', 
             data = df_Eixample_date, 
             color = c_vehicle)
ax[0].set_ylabel(' # of Vehicles', fontsize = labelfont)
ax[0].set_title('Vehicles', fontsize = titlefont)
ax[0].set_xlabel(' ')
ax[0].set(xticklabels=[])

sns.lineplot(ax = ax[1], 
             x ='date', 
             y = 'size', 
             data = df_Eixample_date, 
             color = c_accident)
ax[1].set_ylabel('# of Accidents', fontsize = labelfont)
ax[1].set_title( 'Accidents', fontsize = titlefont)
ax[1].set_xlabel(' ')
ax[1].set(xticklabels=[])

sns.lineplot(ax = ax[2], 
             x ='date', 
             y = 'mild_inj', 
             data = df_Eixample_date, 
             color = c_mild_inj)
ax[2].set_ylabel('# of mild injuries', fontsize = labelfont)
ax[2].set_title('Mild Injuries', fontsize = titlefont)
ax[2].set_xlabel(' ')
ax[2].set(xticklabels=[])

sns.lineplot(ax = ax[3],
             x ='date', 
             y = 'ser_inj', 
             data = df_Eixample_date, 
             color = c_ser_inj)
ax[3].set_ylabel('#  of serious injuries', fontsize = labelfont)
ax[3].set_title( 'Serious Injuries ', fontsize = titlefont)
ax[3].set_xlabel(' ')
plt.suptitle('Accident Trend in Eixample',size=16)
plt.show()

# =============================================================================
# STREETS IN EIXAMPLE
# =============================================================================

Eixample_street_split_1s = df_Eixample.groupby('street_split_1').size().sort_values(ascending=False).head(10).to_frame('size').reset_index()
print(Eixample_street_split_1s)

#finding the top 10 streets for the most serious injuries
Eixample_ser_street = df_Eixample.groupby(['street_split_1'])['ser_inj'].sum().sort_values(ascending=False).head(10).reset_index()
print(Eixample_ser_street)

#finding the top 10 streets for the most mild injuries
Eixample_mil_street = df_Eixample.groupby(['street_split_1'])['mild_inj'].sum().sort_values(ascending=False).head(10).reset_index()
print(Eixample_mil_street)

#finding the top 10 streets for the most vehicles
Eixample_veh_street = df_Eixample.groupby(['street_split_1'])['vehicles'].sum().sort_values(ascending=False).head(10).reset_index()
print(Eixample_veh_street)

#finding the top 10 streets for the most victims
Eixample_vic_street = df_Eixample.groupby(['street_split_1'])['new_victims'].sum().sort_values(ascending=False).head(10).reset_index()
print(Eixample_vic_street)


sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18,12))
sns.barplot(ax = ax[0,0], 
            y = 'street_split_1',
            x = 'vehicles', 
            data = Eixample_veh_street,
            palette = [c_vehicle])
ax[0,0].set_xlabel(' ', fontsize = labelfont)
ax[0,0].set_ylabel('', fontsize = labelfont)
ax[0,0].set_title('Number of Vehicles Involved in an Accident by Street', fontsize = titlefont)

sns.barplot(ax = ax[0,1], 
            y = 'street_split_1', 
            x = 'size', 
            data = Eixample_street_split_1s,
            palette = [c_accident])
ax[0,1].set_xlabel(' ')
ax[0,1].set_ylabel(' ')
ax[0,1].set_title('Number of Accidents by Street', fontsize = titlefont)


sns.barplot(ax = ax[1,0], 
            data = Eixample_mil_street,
            y='street_split_1',
            x='mild_inj',
            palette = [c_mild_inj])
ax[1,0].set_xlabel(' ', fontsize = labelfont)
ax[1,0].set_ylabel(' ', fontsize = labelfont)
ax[1,0].set_title('Number of Mild Injuries by Street', fontsize = titlefont)
plt.suptitle('Worst Streets in Eixample for Accidents', fontsize = 18)


sns.barplot(ax = ax[1,1], 
            data = Eixample_ser_street,
            y='street_split_1',
            x='ser_inj',
            palette = [c_ser_inj])
ax[1,1].set_xlabel(' ', fontsize = labelfont)
ax[1,1].set_ylabel(' ', fontsize = labelfont)
ax[1,1].set_title('Number of Serious Injuries by Street', fontsize = titlefont)
#ax[1,0].set_yticklabels(district_sum['district_name'].unique(), fontsize = 12)

plt.show()


# =============================================================================
# EIXAMPLE -> Examing by Months
# =============================================================================
#creating a summary of months
Eixample_month_ac = df_Eixample.groupby('month').size().to_frame('size').reset_index()

#creating a summary of months
Eixample_month_sum = df_Eixample.groupby(['month'])['vehicles','mild_inj','ser_inj'].sum().reset_index()

#adding an new column to month summary
Eixample_month_sum['size'] = Eixample_month_ac['size']

#order columns
Eixample_month_sum = Eixample_month_sum[['month', 'vehicles','mild_inj','size', 'ser_inj']]
print(Eixample_month_sum)

#creating a new dataaset
Eixample_month_sum1 = pd.melt(Eixample_month_sum, id_vars = "month")
print(Eixample_month_sum1)



fig, ax = plt.subplots(figsize = (12, 8))
ax.plot('month',  'vehicles', data = Eixample_month_sum, color = c_vehicle)
ax.plot('month', 'mild_inj',  data = Eixample_month_sum,color = c_mild_inj)
ax.plot('month', 'size', data = Eixample_month_sum, color= c_accident)
ax.plot('month', 'ser_inj', data = Eixample_month_sum, color = c_ser_inj)
ax.set_ylabel('Quantity.', fontsize = labelfont)
ax.set_xlabel('', fontsize = labelfont)
#ax.set_title('Accident Trends in Eixample By Month', fontsize = titlefont)
L = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
L.get_texts()[0].set_text('Vehicles')
L.get_texts()[1].set_text('Mild Injuries')
L.get_texts()[2].set_text('Accidents')
L.get_texts()[3].set_text('Serious Injuries')
plt.show()

'''the grouped bar graph below was viewed as uninformative
Feel free to unhash and view
'''
#fig, ax = plt.subplots(figsize = (12, 8))
#g = sns.barplot(x = 'month', y = 'value', 
#               hue = 'variable',
#               data = Eixample_month_sum1,
#               palette = [c_vehicle, c_mild_inj, c_accident, c_ser_inj])
#L = g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#L.get_texts()[0].set_text('Vehicles')
#L.get_texts()[1].set_text('Mild Injuries')
#L.get_texts()[2].set_text('Accidents')
#L.get_texts()[3].set_text('Serious Injuries')
#g.set_xticklabels(Eixample_month_sum['month'].unique(), fontsize = labelfont)
#g.set_xlabel(' ')
#g.set_ylabel('Quantity')
##g.set_title('Accident Trends in Eixample by Month', fontsize = titlefont)
#plt.show()


#setting Style
sns.set_style('darkgrid')
#creating a line plot with month and type of Injuries
fig, ax = plt.subplots(figsize = (12, 8))
sns.pointplot(ax=ax, 
             x = 'month', 
             y = 'mild_inj', 
             data = Eixample_month_sum,
             color = c_mild_inj)
ax.set_ylabel('# of Mild Injuries', fontsize = 14,color = c_mild_inj)
ax.set_xlabel(' ')
#ax.set_title('Injuries by Months in Example', fontsize =  16)
ax.tick_params('y', colors = c_mild_inj)
ax.yaxis.grid(False) # Hide the horizontal gridlines
ax.xaxis.grid(True) # Show the vertical gridlines
ax2 = ax.twinx()
sns.pointplot(ax = ax2, 
                x = 'month', 
                y = 'ser_inj',
                data = Eixample_month_sum,
                color = c_ser_inj) 
ax2.set_ylabel('# of Serious Injuries', fontsize = 14, color = c_ser_inj)
ax2.set_xlabel(' ')
ax2.tick_params('y', colors = c_ser_inj)
ax2.yaxis.grid(False) # Hide the horizontal gridlines
ax2.yaxis.grid(False) # Hide the horizontal gridlines


# =============================================================================
# EIXAMPLE -> COMPARING MONTHS AGAINST AVERAGE OF ITSELF
# =============================================================================
# making a copy of the dataset to ensure the previous map works
Eixample_month_sum_copy = Eixample_month_sum

Eixample_month_sum_copy['mild_diff'] = Eixample_month_sum['mild_inj'] - Eixample_month_sum['mild_inj'].mean()
Eixample_month_sum_copy['ser_diff'] = Eixample_month_sum['ser_inj'] - Eixample_month_sum['ser_inj'].mean()
Eixample_month_sum_copy['veh_diff'] = Eixample_month_sum['vehicles'] - Eixample_month_sum['vehicles'].mean()
Eixample_month_sum_copy['size_diff'] = Eixample_month_sum['size'] - Eixample_month_sum['size'].mean()

print(Eixample_month_sum_copy)


sns.set_style('darkgrid')
# creating a graph using a SEABORN
fig, ax = plt.subplots(2,2, figsize=(18,12))
sns.barplot(ax = ax[0,0], 
            y = 'month', 
            x = 'veh_diff',  palette='Set2',
            data = Eixample_month_sum_copy)
ax[0,0].set_xlabel(' ', fontsize = labelfont)
ax[0,0].set_ylabel('', fontsize = labelfont)
ax[0,0].set_title('Comparing Vehicles to the Annual Mean', fontsize = titlefont)
ax[0,0].set_yticklabels(Eixample_month_sum_copy['month'].unique(), fontsize = 12)

sns.barplot(ax = ax[0,1],
            y = 'month', 
            x = 'size_diff', palette='Set2',
            data = Eixample_month_sum_copy)
ax[0,1].set_xlabel(' ', fontsize = labelfont)
ax[0,1].set_ylabel('', fontsize = labelfont)
ax[0,1].set_title('Comparing Number of Accidents to the Annual Mean', fontsize = titlefont)

sns.barplot(ax = ax[1,0], 
            data = Eixample_month_sum_copy,
            y='month',palette='Set2',
            x='mild_diff')
ax[1,0].set_xlabel(' ', fontsize = labelfont)
ax[1,0].set_ylabel('', fontsize = labelfont)
ax[1,0].set_title('Comparing  Mild Injuries to the Annual Mean', fontsize = titlefont)
#plt.suptitle('Accident Trends by Month in Eixample', fontsize = 18)

sns.barplot(ax = ax[1,1], 
            data = Eixample_month_sum_copy,
            y='month',palette='Set2',
            x='ser_diff')
ax[1,1].set_xlabel(' ', fontsize = labelfont)
ax[1,1].set_ylabel('', fontsize = labelfont)
ax[1,1].set_title('Comparing Serious Injuries to the Annual Mean', fontsize = titlefont)
ax[1,1].set_yticklabels(Eixample_month_sum_copy['month'].unique(), fontsize = 12)


plt.show()

# =============================================================================
# EIXAMPLE -> Examining the Weekday
# =============================================================================

Eixample_weekday_ac = df_Eixample.groupby('weekday').size().to_frame('size').reset_index()
print(Eixample_weekday_ac)


#creating summary weekday summary 
Eixample_weekday_sum = df_Eixample.groupby(['weekday'])['vehicles','mild_inj','ser_inj'].sum().reset_index()

#adding the size column to weekday summary 
Eixample_weekday_sum['size'] = Eixample_weekday_ac['size']

#order columns
Eixample_weekday_sum = Eixample_weekday_sum[['weekday',  'vehicles','mild_inj','size', 'ser_inj']]

#viewing the new dataset
print(Eixample_weekday_sum)


# creating a linegraph to examine the weekday trends
fig, ax = plt.subplots(figsize = (12, 8))
ax.plot('weekday', 'vehicles', data = Eixample_weekday_sum,  color = c_vehicle)
ax.plot('weekday', 'mild_inj', data = Eixample_weekday_sum, color = c_mild_inj)
ax.plot('weekday', 'size', data = Eixample_weekday_sum, color= c_accident)
ax.plot('weekday', 'ser_inj', data = Eixample_weekday_sum, color = c_ser_inj)
ax.set_ylabel('Quantity.', fontsize = labelfont)
ax.set_xlabel('', fontsize = labelfont)
ax.set_title('Accident Trends By Weekday by Eixample', fontsize = titlefont)
ax.set_xticklabels(Eixample_weekday_sum['weekday'].unique(), rotation = 45, fontsize = 12)
L = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
L.get_texts()[0].set_text('Vehicles')
L.get_texts()[1].set_text('Mild Injuries')
L.get_texts()[2].set_text('Accidents')
L.get_texts()[3].set_text('Serious Injuries')
plt.show()


#setting Style
sns.set_style('darkgrid')
#creating a line plot ;
fig, ax = plt.subplots(figsize = (12, 8))
sns.pointplot(ax=ax, 
             x = 'weekday', 
             y = 'mild_inj', 
             data= Eixample_weekday_sum,
             color = c_mild_inj)
ax.set_ylabel('Number of Mild Injuries', fontsize = 14,color = c_mild_inj)
ax.set_xlabel(' ')
#ax.set_title('Injuries by Weekdays in Eixample', fontsize =  16)
ax.tick_params('y', colors = c_mild_inj)
ax.yaxis.grid(False) # Hide the horizontal gridlines
ax.xaxis.grid(True) # Show the vertical gridlines
ax2 = ax.twinx()
sns.pointplot(ax = ax2, 
                x = 'weekday', 
                y = 'ser_inj', 
                data = Eixample_weekday_sum,
                color = c_ser_inj) 
ax2.set_ylabel('Number of Serious Injuries', fontsize = 14, color = c_ser_inj)
ax2.set_xlabel(' ')
ax2.tick_params('y', colors = c_ser_inj)
ax2.yaxis.grid(False) # Hide the horizontal gridlines
ax2.xaxis.grid(False) # Show the vertical gridlines


# =============================================================================
# EIXAMPLE - > Heat map by Hour
# =============================================================================

Eixample_map = folium.Map(location=[41.395425, 2.169141],zoom_start = 12) 

Eixample_coordinates_data = df_Eixample[['latitude', 'longitude']]
Eixample_coordinates_data['Weight'] = df_Eixample['hour']
Eixample_coordinates_data['Weight'] = Eixample_coordinates_data['Weight'].astype(float)
Eixample_coordinates_data = Eixample_coordinates_data.dropna(axis=0, subset=['latitude','longitude', 'Weight'])

coordinates_list = [[[row['latitude'],row['longitude']] for index, 
                     row in Eixample_coordinates_data[Eixample_coordinates_data['Weight'] == i].iterrows()] for i in range(0,24)]

hm = plugins.HeatMapWithTime(coordinates_list,auto_play=True,max_opacity=0.8)
hm.add_to(Eixample_map)

Eixample_map.save('Eixample_map heat map.html')






















# =============================================================================
# UNUSED INSIGHTS
# =============================================================================



# =============================================================================
# BREAKDOWN OF Accident Trends
# =============================================================================
Eixample_road_types = df_Eixample.groupby('road_type').size().to_frame('size')
Eixample_accident_types = df_Eixample.groupby('accident_type').size().to_frame('size')
Eixample_seasons = df_Eixample.groupby('season').size().to_frame('size')
Eixample_weekends = df_Eixample.groupby('weekend?').size().to_frame('size')


print(Eixample_road_types)
print(Eixample_accident_types)
print(Eixample_seasons)
print(Eixample_weekends)


sns.set_style('whitegrid')
fig, ax = plt.subplots(2,2, figsize=(12, 12))
# ACCIDENT TYPE
ax[0,0].pie(x = df_Eixample['accident_type'].value_counts().tolist(), #size
    colors = [c_no_serious, c_serious], 
    labels = ('No Serious Injuries', 'One or More Serious Injury'),
    autopct = '%1.1f%%', 
    shadow = False, counterclock = False,  
    textprops={'fontsize': 12},
    startangle = 90)
ax[0,0].set_title('..by Serious Injuries', fontsize = 16)

# INTERSECTION
ax[0,1].pie(x = df_Eixample['road_type'].value_counts().tolist(), #size
    colors = [c_straight, c_intersection],
    labels = ('Straight Road', 'Intersection'),
    autopct = '%1.1f%%', counterclock = False,  
    shadow = False, textprops={'fontsize': 12},
    startangle = 90)
ax[0,1].set_title('..by Street Type', fontsize = 16)
    
# SEASON
ax[1,0].pie(x = df_Eixample['season'].value_counts().tolist(), #size
    colors = [c_high_season, c_low_season], 
    labels = ('Low Season', 'High Season'),
    autopct = '%1.1f%%',counterclock = False,   
    shadow = False, textprops={'fontsize': 12},
    startangle = 90)
ax[1,0].set_title('...by Season', fontsize = 16)

# WEEKEND
ax[1,1].pie(x = df_Eixample['weekend?'].value_counts().tolist(), #size
    colors = [c_weekday, c_weekend], 
    labels = ('Weekday', 'Weekend'),
    autopct = '%1.1f%%', counterclock = False,  
    shadow = False, textprops={'fontsize': 12},
    startangle = 90)
ax[1,1].set_title('...by Weekday / Weekend', fontsize = 16)
plt.suptitle('Breakdown of Accidents in Eixample', fontsize = 18)
# Display the plot
plt.show()


# =============================================================================
# Eixample -> Part of Day
# =============================================================================
Eixample_part_day_ac = df_Eixample.groupby('part_day').size().to_frame('size').reset_index()
print(Eixample_part_day_ac)

#creating a summary of months
Eixample_part_day_sum = df_Eixample.groupby(['part_day'])['vehicles','victims' ,'mild_inj','ser_inj'].sum().reset_index()
print(Eixample_part_day_sum)

#adding an new column to month summary
Eixample_part_day_sum['size'] = Eixample_part_day_ac['size']

#order columns
Eixample_part_day_sum = Eixample_part_day_sum[['part_day', 'vehicles', 'victims','mild_inj','size', 'ser_inj']]

#viewing new dataset
print(Eixample_part_day_sum)


Eixample_part_day_sum1 = pd.melt(Eixample_part_day_sum, id_vars = "part_day")
print(Eixample_part_day_sum1)

sns.set_style('darkgrid')
#building 
fig, ax = plt.subplots(figsize = (12, 8))
g = sns.barplot(x = 'part_day', y = 'value', 
               hue = 'variable',
               data = Eixample_part_day_sum1,
               palette = [c_vehicle, c_victim, c_mild_inj, c_accident, c_ser_inj])
g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
L = g.legend()
L.get_texts()[0].set_text('Vehicles')
L.get_texts()[1].set_text('Victims')
L.get_texts()[2].set_text('Mild Injuries')
L.get_texts()[3].set_text('Accidents')
L.get_texts()[4].set_text('Serious Injuries')
g.set_xticklabels(Eixample_part_day_sum1['part_day'].unique(), fontsize = labelfont)
g.set_xlabel(' ')
g.set_ylabel('Quantity')
g.set_title('Examining the Accidents Trends Part of Day', fontsize = titlefont)
plt.show()



# =============================================================================
# Accidents by District and hour of day 
# =============================================================================
df['hour_string'] = df['hour'].astype(str)


df.loc[df['hour_string'].str.contains('8'), 'traffic_type'] = 'Rush Hour'
df.loc[df['hour_string'].str.contains('9'), 'traffic_type'] = 'Rush Hour'
df.loc[df['hour_string'].str.contains('13'), 'traffic_type'] = 'Siesta'
df.loc[df['hour_string'].str.contains('14'), 'traffic_type'] = 'Siesta'
df.loc[df['hour_string'].str.contains('19'), 'traffic_type'] = 'Rush Hour'
df.loc[df['hour_string'].str.contains('20'), 'traffic_type'] = 'Rush Hour'

df.loc[df['traffic_type'].isnull(), 'traffic_type'] = 'Regular'


hour_ac = df.groupby(['hour', 'district_name', 'traffic_type']).size().to_frame('size').reset_index()
print(hour_ac)

#creating a summary of months
hour_sum = df.groupby(['hour', 'district_name', 'traffic_type'])['vehicles','victims' ,'mild_inj','ser_inj'].sum().reset_index()
print(hour_sum)

#adding an new column to month summary
hour_sum['size'] = hour_ac['size']

#order columns
hour_sum = hour_sum[['hour', 'district_name','traffic_type','vehicles', 'victims','mild_inj','size', 'ser_inj']]


print('Nan in each columns' , hour_sum.isna().sum(), sep='\n')

hour_sum = hour_sum.dropna()
hour_sum  = hour_sum.reset_index()


#viewing new dataset
print(hour_sum)

sns.catplot(x = 'hour', y = 'size', 
               col = 'district_name',
               kind= 'bar', hue = 'traffic_type',
               col_wrap = 3,
               data = hour_sum)
plt.show()

