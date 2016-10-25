import plotly.plotly as py
from plotly.graph_objs import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import re


# example from adjacency matrix

# Ad=np.array([[0,1,1,1,0,0,0,1], # Adjacency matrix
#             [1,0,1,0,1,1,1,0],
#             [1,1,0,0,0,0,1,1],
#             [1,0,0,0,1,1,1,1],
#             [0,1,0,1,0,1,1,0],
#             [0,1,0,1,1,0,1,0],
#             [0,1,1,1,1,1,0,1],
#             [1,0,1,1,0,0,1,0]], dtype=float)

# Gr = nx.from_numpy_matrix(Ad)

# nx.draw(Gr, node_color='g',with_labels =True, alpha = 0.5)

# another example

# G2 = nx.Graph() #G is an empty graph
# Nodes=range(51)
# G2.add_nodes_from(Nodes)
# Edges=[(0,1),(1,2),(1,3)]
# G2.add_edges_from(Edges)
# nx.draw(G2, node_color='c',edge_color='k', with_labels=True)

# This next piece of script is to visualise the network throughput data

def create_file_reference(initial_dir, file_name):
    s = path.join(initial_dir, file_name)
    return s

def create_training_data(train_input_data_path, numeric_input_data_path):
    data_init = pd.read_csv(train_input_data_path,
                            nrows=10, dtype=float, usecols=list(range(0, 1157)))
    date_cols = data_init.count().reset_index().sort_values(by=0, ascending=False)
    date_cols['Station'] = date_cols['index'].apply(lambda x: x.split('_')[1] if x != 'Id' else x)
    date_cols = date_cols.drop_duplicates('Station', keep='first')['index'].tolist()
    train_date = pd.read_csv(train_input_data_path, usecols=date_cols, nrows=100000)
    train_numeric = pd.read_csv(numeric_input_data_path, nrows=100)
    return train_date, train_numeric, date_cols

# we want to get the data into the format of
# ID time station
def times_by_station(data, cols):
    times = pd.DataFrame(columns=['Id', 'times', 'station'])
    # every dictionary has a key value pair therefore for each key value pair we want to update the value
    for c in range(1,len(cols)):
        x = np.array([cols[c].split('_')[1]])
        d = {'Id': data.iloc[:, 0], 'times': data.loc[:,cols[c]], 'station': np.repeat(x, len(data.loc[:, cols[c]]))}
        p1 = pd.DataFrame(d)
        times = times.append(p1, ignore_index=True)
    return times

def create_sorted_station_times(train_date, date_cols):
    train_station_times = times_by_station(train_date, date_cols).dropna().sort_values(by=['Id', 'times'],
                                                                                       ascending=[1, 1])
    return train_station_times

# create a numpy array [matrix] to hold the values that will become the adjacency matrix
# The coordinates will be the S29-230 split using the following syntax
# re.split(r'(\d+)',train_station_times.loc[0]['station'])

# even if we do this step though we are still going to have a problem. because the data set is so large we need an
# efficient way of selecting indices. Basically we need to get the table into the format of i,j index of the adjacency
# matrix.

# Id station times station_next validation_id
# 4    S1    82.24     S2          4

# In this way we can get the appropriate results. We included the validation_id because we need a nice little tag
# to tell us if we are still on the same product or not we know this is either going to be the minimum time or the 
# maximum time and therefore we can adjust the values accordingly

# The appropriate python function to get all of this in order:
# use the shift function df['col_name'].shift(i) i in -R -> +R
def split_stations_into_numeric(train_station_ordered_pandas):
    v = train_station_ordered_pandas['station'].apply(lambda s: re.split(r'(\d+)', s)[1])
    train_station_ordered_pandas['numeric_station_from'] = v
    return train_station_ordered_pandas
