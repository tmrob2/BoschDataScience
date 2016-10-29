import plotly.plotly as py
from plotly.graph_objs import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import re

class StationNetwork:
    """This class is used to visualise the station network contained as a sequence of dates in the data 
       on class initialisation the class will take:
        - option: 0 Represents constructing data but not the graph, 1 represents constructing the data and the supporting graph and adjacency matrix
        - ide: {'vs', 'pc'}. 'vs' represents visual studio, 'pc' represents pycharm 
    """

    def __init__(self, ide_option, option=0):
        """ On initialisation of the class, all of the data will be constructed. If option 1 has been selected then the graph will be created as well"""
        if ide_option in ['vs','pc']: 
            self.train_station_timestamp, self.train_date, self.train_numeric, self.date_cols = self.which_ide(ide_option)
            self.train_station_timestamp = self.adjust_timestamp_df()
            if option == 1:
                self.adj = self.define_adj(self.train_station_timestamp, self.date_cols)
                self.d, self.labels, self.coords_x, self.coords_y= self.create_list_of_pairs(self.date_cols[1:len(self.date_cols)])
                self.Gr = self.drawGraph(self.adj, self.d, self.labels)
        else:
            print('Ide option does not exist, see help(class_name) for more information on input args')

    def create_file_reference(self, initial_dir, file_name):
        s = path.join(initial_dir, file_name)
        return s

    def create_training_data(self, train_input_data_path, numeric_input_data_path):
        data_init = pd.read_csv(train_input_data_path,
                                nrows=10, dtype=float, usecols=list(range(0, 1157)))
        date_cols = data_init.count().reset_index().sort_values(by=0, ascending=False)
        # Initial string of the cols: e.g. L1_S32_F2343 i.e. the vector will be 'split' into 3 with i = 0,1,2
        date_cols['Station'] = date_cols['index'].apply(lambda x: x.split('_')[1] if x != 'Id' else x)
        date_cols = date_cols.drop_duplicates('Station', keep='first')['index'].tolist()
        train_date = pd.read_csv(train_input_data_path, usecols=date_cols, nrows=100000)
        train_numeric = pd.read_csv(numeric_input_data_path, nrows=100000)
        return train_date, train_numeric, date_cols

    # we want to get the data into the format of
    # ID time station
    def times_by_station(self, data, cols):
        times = pd.DataFrame(columns=['Id', 'times', 'station'])
        # every dictionary has a key value pair therefore for each key value pair we want to update the value
        for c in range(1,len(cols)):
            x = np.array([cols[c].split('_')[1]])
            d = {'Id': data.iloc[:, 0], 'times': data.loc[:,cols[c]], 'station': np.repeat(x, len(data.loc[:, cols[c]]))}
            p1 = pd.DataFrame(d)
            times = times.append(p1, ignore_index=True)
        return times

    def create_sorted_station_times(self, train_date, date_cols):
        train_station_times = self.times_by_station(train_date, date_cols).dropna().sort_values(by=['Id', 'times'],
                                                                                           ascending=[1, 1])
        return train_station_times

    # create a numpy array [matrix] to hold the values that will become the adjacency matrix
    # The coordinates will be the S29-230 split using the following syntax
    # re.split(r'(\d+)',train_station_times.loc[0]['station'])

    # Id station times station_next validation_id
    # 4    S1    82.24     S2          4

    # use the shift function (df['col_name'].shift(i) i in -R -> +R) to do this
    def split_stations_into_numeric(self, train_station_ordered_pandas):
        v = train_station_ordered_pandas['station'].apply(lambda s: re.split(r'(\d+)', s)[1])
        train_station_ordered_pandas['numeric_station_from'] = v
        u = train_station_ordered_pandas['numeric_station_from'].shift(-1)
        train_station_ordered_pandas['numeric_station_to'] = u
        w = train_station_ordered_pandas['Id'].shift(-1)
        train_station_ordered_pandas['ref_id'] = w
        return train_station_ordered_pandas

    #script so i don't have to keep sending to interactive

    def which_ide(self, ide):
        if ide=='vs':
            # do some visual studio stuff here
            initial_directory = "C:/Users/Thomas/Documents/Kaggle/Bosch"
        elif ide=='pc':
            # do some pycharm stuff here
            initial_directory = '~/PycharmProjects/BoschDataScience'

        train_date_path = self.create_file_reference(initial_directory, 'train_date.csv')
        train_numeric_path = self.create_file_reference(initial_directory, 'train_numeric.csv')
        train_date, train_numeric, date_cols = self.create_training_data(train_date_path, train_numeric_path)
        train_station_timestamp = self.create_sorted_station_times(train_date, date_cols)
        train_station_timestamp = self.split_stations_into_numeric(train_station_timestamp)
        # need to adjust the shifted station times to match the correct id
        train_station_timestamp['numeric_station_to'] = \
            train_station_timestamp[['Id', 'ref_id', 'numeric_station_from', 'numeric_station_to']].apply(
            lambda x: x[2] if x[0] != x[1] else x[3], axis=1)

        return train_station_timestamp, train_date, train_numeric, date_cols

    # We can now create a weighted adjacency matrix which contains the necessary data to visualise our graph
    def adjust_timestamp_df(self):
        df = self.train_station_timestamp
        df2 = self.train_numeric
        a = df[lambda df: df.numeric_station_from == df.numeric_station_to].index.values
        for i in range(0, len(a)):
            id = df.get_value(a[i], 'Id')
            response = (df2.loc[df2.Id == id]['Response']).values[0]
            if response == 1:
                df.set_value(a[i], 'numeric_station_to', -99)
            else:
                df.set_value(a[i], 'numeric_station_to', 99)
        return df

    def define_adj(self, data, cols, response_fail=False):
        l = len(cols) - 1 + 2 #This may seem silly but it lets us keep track of the fact that we
        # have id and response to consider i.e. r in {0,1}
        A = np.repeat(0, l**2).reshape(l, l)
        # we are just entering the counts of the edges which are technically just groups of from to values
        if response_fail==False:
            data_count = data.groupby(['numeric_station_from', 'numeric_station_to']).count()
        else:
            data_count = data.isin
        # Time to loop over this should be fairly good
        for i, r in data_count.iterrows():
            if i[1] == -99:
                A[int(i[0])][52] = r['Id']
            elif i[1] == 99:
                A[int(i[0])][53] = r['Id']
            else:
                A[int(i[0])][int(i[1])] = r['Id']
        return A

    # Now that the data is aligned in the format that we need it to be in we need to determine which values end up in
    # failure

    def drawGraph(self, Adjacency, pos, labels):
        Gr = nx.from_numpy_matrix(Adjacency)
        plt.figure(figsize=(22, 8))
        nx.draw_networkx_nodes(Gr, pos, node_size=500)
        eVeryLarge = [(u, v) for (u, v, d) in Gr.edges(data=True) if d['weight'] > 50000]
        eLarge = [(u, v) for (u, v, d) in Gr.edges(data=True) if d['weight'] > 1000]
        eLess = [(u, v) for (u, v, d) in Gr.edges(data=True) if d['weight'] < 1000]
        nx.draw_networkx_edges(Gr, pos, edgelist=eLarge, width=2, edge_color='b')
        nx.draw_networkx_edges(Gr, pos, edgelist=eLess, width=2, style='dashed')
        nx.draw_networkx_edges(Gr, pos, edgelist=eVeryLarge, width=5, edge_color='r')
        nx.draw_networkx_labels(Gr, pos, labels=labels)
        plt.savefig('Station_Network.png')
        return Gr

    def create_list_of_pairs(self, machine_feats):
        labels = {}
        for i in range(0, len(machine_feats)):
            labels[i] = 'S' + str(i)

        labels[53] = 'R_0'
        labels[52] = 'R_1'

        y = [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 6, 6,
             7, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 13, 14, 8, 9, 10, 11, 11, 12, 13, 14, 15, 15, 16, 17, 17, 18]

        x = [5, 9, 9, 10, 8, 10, 8, 10, 8, 7, 10, 9, 8, 5, 5, 6, 4, 6, 4, 6, 4, 5, 6, 5, 4, 2,
             0, 2, 1, 0, 7, 7, 8, 7, 6, 8, 6, 7, 7, 7, 3, 3, 3, 4, 2, 3, 3, 3, 4, 2, 3, 7, 3, 7]

        d = {}
        for i in range(0,len(x)):
            d[i] = (y[i], x[i])
        return d, labels, x, y




class CreateLanlGraph:

    # In this class we want to represent the machine sequences that end in response = 1, i.e. this will be the numerical
    # data with id's
    def __init__(self, data):
        self.InputData = data
        self.filter_response_failure()

    def filter_response_failure(self):
        self.Failures = pd.DataFrame(columns=['Id', 'Response'])
        # With numerical training data input lets have a look at the sequences with repsonse 1
        df = self.InputData
        self.test = df.loc[lambda df: df.Response == 1, ['Id', 'Response']]






