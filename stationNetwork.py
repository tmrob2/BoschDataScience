import plotly.plotly as py
from plotly.graph_objs import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import datetime

G = nx.random_geometric_graph(200,0.125)
pos=nx.get_node_attributes(G,'pos')

dmin=1
ncenter=0

for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

p = nx.single_source_shortest_path_length(G,ncenter)

#creating edges
#add edges as disconnected lines in a single trace and nodes as a scatter trace

edge_trace = Scatter(
    x = [],
    y = [],
    line=Line(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0,y0 = G.node[edge[0]]['pos']
    x1,y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += [x0,x1,None]
    edge_trace['y'] += [y0,y1,None]

node_trace = Scatter(
    x=[], 
    y=[], 
    text=[],
    mode='markers', 
    hoverinfo='text',
    marker=Marker(
        showscale=True,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        colorscale='YIGnBu',
        reversescale=True,
        color=[], 
        size=10,         
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))

for node in G.nodes():
    x,y = G.node[node]['pos']
    node_trace['x'].append(x)
    node_trace['y'].append(y)


#Colour Node points by the number fo connections

for node, adjacencies in enumerate(G.adjacency_list()):
    node_trace['marker']['color'].append(len(adjacencies))
    node_info = '# of connections: '+ str(len(adjacencies))
    node_trace['text'].append(node_info)

fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Network graph made with Python',
                titlefont=dict(size=16),
                showlegend=False, 
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

plotly.offline.plot(fig)

#example from adjacency matrix

Ad=np.array([[0,1,1,1,0,0,0,1], # Adjacency matrix
             [1,0,1,0,1,1,1,0],
             [1,1,0,0,0,0,1,1],
             [1,0,0,0,1,1,1,1],
             [0,1,0,1,0,1,1,0],
             [0,1,0,1,1,0,1,0],
             [0,1,1,1,1,1,0,1],
             [1,0,1,1,0,0,1,0]], dtype=float)

Gr = nx.from_numpy_matrix(Ad)

nx.draw(Gr, node_color='g',with_labels =True, alpha = 0.5)

#another example 

G2 = nx.Graph() #G is an empty graph
Nodes=range(51)
G2.add_nodes_from(Nodes)
Edges=[(0,1),(1,2),(1,3)]
G2.add_edges_from(Edges)
nx.draw(G2, node_color='c',edge_color='k', with_labels=True)

#This next piece of script is to visulaise the network throughput data

data_init = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_date.csv", 
                        nrows = 10, dtype=float, usecols = list(range(0,1157)))

date_cols = data_init.count().reset_index().sort_values(by=0,ascending = False)

date_cols['Station'] = date_cols['index'].apply(lambda x: x.split('_')[1] if x != 'Id' else x)

date_cols = date_cols.drop_duplicates('Station', keep = 'first')['index'].tolist()

train_date = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_date.csv", usecols = date_cols, nrows = 100000)
train_numeric = pd.read_csv("C:/Users/Thomas/Documents/Kaggle/Bosch/train_numeric.csv", nrows = 100)
#we want to get the data into the format of 
# ID time station
def times_by_station(data, cols):
    times = pd.DataFrame(columns = ['Id','times','station'])
    #every dictionary has a key value pair therefore for each key value pair we want to update the value 
    for c in range(1,len(cols)):
        x = np.array([cols[c].split('_')[1]])
        d = {'Id': data.iloc[:,0], 'times': data.loc[:,cols[c]], 'station': np.repeat(x,len(data.loc[:,cols[c]]))}
        p1 = pd.DataFrame(d)
        times = times.append(p1, ignore_index=True)
    return(times)

train_station_times = times_by_station(train_date,date_cols).dropna().sort_values(by= ['Id','times'],ascending=[1,1])

#create a numpy array [matrix] to hold the values that will become the adjacency matrix
#The coordinates will be the S29-230 split using the following syntax
#re.split(r'(\d+)',train_station_times.loc[0]['station'])

#even if we do this step though we are still going to have a problem. because the data set is so large we need an
#efficient way of selecting indices. Basically we need to get the table into the format of i,j index of the adjacency
#matrix. 

# Id station times station_next validation_id
# 4    S1    82.24     S2          4

# In this way we can get the appropriate results. We included the validation_id because we need a nice little tag
# to tell us if we are still on the same product or not we know this is either going to be the minimum time or the 
# maximum time and therefore we can adjust the values accordingly

# The appropriate python function to get all of this in order is the shift function df['col_name'].shift(i) i in -R -> +R
