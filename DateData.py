import numpy as np
import pandas as pd
import collections
import datetime
import matplotlib.pyplot as plt

data_init = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_date.csv", 
                        nrows = 10, dtype=float, usecols = list(range(0,1157)))
#cols = list(data_init.columns)
#cols.remove('Id')

#we will create a list of dictionaries with the following structure
# list[id: id, time: time, station: station]
# we are going to drop the NaNs
#data = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_date.csv",
#                   chunksize = 100, dtype=float, usecols = list(range(0,1157)))

date_cols = data_init.drop('Id', axis=1).count().reset_index().sort_values(by=0,ascending = False)

date_cols['Station'] = date_cols['index'].apply(lambda x: x.split('_')[1])

date_cols = date_cols.drop_duplicates('Station', keep = 'first')['index'].tolist()

l_date_cols = list(date_cols)

train_date = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_date.csv", usecols = date_cols)

#we want to get the data into the format of 
# ID time station
def times_by_station(data, cols):
    times = pd.DataFrame(columns = ['times','station'])
    #every dictionary has a key value pair therefore for each key value pair we want to update the value 
    for c in range(0,len(cols)):
        x = np.array([cols[c].split('_')[1]])
        d = {'times': train_date.iloc[:,c].dropna(), 'station': np.repeat(x,len(train_date.iloc[:,c].dropna()))}
        p1 = pd.DataFrame(d)
        times = times.append(p1, ignore_index=True)
    return(times)



train_station_times = times_by_station(train_date,l_date_cols)

train_station_cnt = train_station_times.groupby('times').count()[['station']].reset_index()

train_station_cnt.columns = ['times','cnt']

plt.plot(train_station_cnt['times'].values, train_station_times['cnt'].values, 'g.', alpha=0.1, label='Train')

# ok so this is a bit funny and took me a little while to understand (4 seconds) but I believe this is to make the data
# somewhat continuous, there may be some missing data in the orginal train_station_cnt which would cause problems
# when doing the autocorrelation anlaysis
time_ticks = np.arange(train_station_cnt['times'].min(), train_station_cnt['times'].max()+0.01,0.01)

time_ticks = pd.DataFrame({'times': time_ticks})

time_ticks = pd.merge(time_ticks, train_station_cnt, how = 'left', on = 'times')

time_ticks = time_ticks.fillna(0)

#Autocorrelation analysis

x = time_ticks['cnt']

max_lag = 8000

#some note on np slicing and indexation. 
# x[i:j:k] is the general syntax 
# i = starting index
# j = stopping index
# k = step size
# if j is blank then the default is the length of the array
# so in the case of autocrrelation we would like to see how each of the points in the time series
# correlate to each for the frequency at the time interval
auto_corr_k = range(1, max_lag)
auto_corr = np.array([1]+[np.corrcoef(x[:-k],x[k:])[0,1] for k in auto_corr_k])
pyplot.plot(auto_corr,'k.',label='autocorrelation by 0.01')
pyplot.title('Train Sensor Time Autocorrelation')
period = 25

#once the autocorrelation can be determined then we can set the period and look at what is happening
#within that period

train_station_cnt['week_part'] = ((train_station_cnt['times'].values*100)%1679).astype(np.int64)

train_week_part = train_station_cnt.groupby(['week_part'])[['cnt']].sum().reset_index()

pyplot.plot(train_week_part.week_part.values, train_week_part.cnt.values, 'b.', alpha = 0.1, label= 'Week part')




