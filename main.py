import numpy as np
import operator
import matplotlib.pyplot as plt
import networkx as nx
import math
from concurrent.futures import ThreadPoolExecutor
import networkx.algorithms.community as nxc
is_oriented=True
filename = 'musae_git_edges.csv'
attrib_filename = 'musae_git_target.csv'
num_nodes = 37699
def create_graph():
    delimiter = ','
    g = None
    if(is_oriented == True):
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header =1, dtype='int')
    attribs = np.genfromtxt(attrib_filename, delimiter=delimiter, skip_header =1, dtype='str')
    for i in range(0, num_nodes):
        g.add_node(i)
    for i in range(0, data.shape[0]):
        g.add_edge(data[i,0], data[i,1])
        if(is_oriented == False):
            g.add_edge(data[i,1], data[i,0])
    #ml_target = attribs[:,2].astype('int')
    ml_target = attribs[:,2]
    ids = attribs[:,0].astype('int')
    types1 = {}
    types2 = {}
    name = attribs[:,1]
    for i in range(0, len(ids)):
        types1[ids[i]] = name[i]
        types2[ids[i]] = ml_target[i]
    nx.set_node_attributes(g, types1, 'name')
    nx.set_node_attributes(g, types2, 'developer')
    return g

def print_graph(g):
    for i in g.nodes:
        print(str(i) + "-> ", end= '')
        for j in g.edges(i):
            print (str(j[1]) + ', ', end = '')
        print('')

def plot_degree_dist(g):
    size = g.nodes
    proportion = dict()
    for i in range(0, size):
        if g.degree[i] not in proportion:
            ind = g.degree[i]
            proportion[ind] = 0.0
    for i in range(0, size):
        ind = g.degree[i]
        proportion[ind] = proportion[ind] + 1  
    degrees = list(proportion.keys())
    values = list(proportion.values())
    for i in range(0, len(values)):
        values[i] = values[i]/size
    plt.title("Degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(degrees, values, marker='.', c='#e36387')
    plt.ylabel('P')
    plt.xlabel('Degree')
    #plt.grid(True)
    plt.show()

def plot_bar_char(x, y):
    plt.bar(x, y, width=1, color='#e36387')
    plt.show()

def calc_plot_strongly_connected_components(g):
    sizes = [len(c) for c in sorted(g.calc_connected_components(), key=len, reverse=True)]
    dat = {}
    for i in range(0, len(sizes)):
        if i not in dat:
            dat[i] = sizes[i]
    
    plt.scatter(dat.keys(), dat.values(), color='#e36387')
    plt.show()


def calc_plot_weakly_connected_components(g):
    sizes = [len(c) for c in sorted(g.calc_weakly_connected_components(), key=len, reverse=True)]
    dat = {}
    for i in range(0, len(sizes)):
        if i not in dat:
            dat[i] = sizes[i]
    
    plt.scatter(dat.keys(), dat.values(), color='#e36387')
    plt.show()

def calc_connected_components(g):
    if is_oriented == True:
        print('Number of connected components: ' + str(nx.number_strongly_connected_components(g)))
        return nx.strongly_connected_components(g)
    else:
        return nx.connected_components(g)
def calc_weakly_connected_components(g):
    if(is_oriented == True):
        print('Number of weakly connected components: ' + str(nx.number_weakly_connected_components(g)))
        return nx.weakly_connected_components(g)
    else:
        return None
def calc_in_degree_centrality(g):
    if is_oriented == True:
        return nx.in_degree_centrality(g)
    else:
        return None
def calc_out_degree_centrality(g):
    if is_oriented == True:
        return nx.out_degree_centrality(g)
    else:
        return None


def plot_subgraph_from_vertex(g, node_index):
    vertices = []
    colors = []
    node_sizes = []
    vertices.append(node_index)
    for e in g.edges(node_index):
        vertices.append(e[1])
    for e in g.in_edges(node_index):
        vertices.append(e[0])
    sub_g = g.subgraph(vertices)
    for i in sub_g.nodes():
        if(i == node_index):
            node_sizes.append(10)
            colors.append('#00bcd4')
        else:
            node_sizes.append(0.5)
            colors.append('#b52b65')
    nx.draw(sub_g, node_size=node_sizes, node_color=colors, width=0.1, edge_color='#dfd3c3', arrows=True, arrowsize=2)
    plt.show()

def main():
    g = create_graph()
    run = True
    while(run == True):
        print('Options\n',
        '\t01 - Calculate the vertex with maximum eigen vector centrality\n',
        '\t02 - Calculate the vertex with maximum degree centrality\n',
        '\t03 - Calculate the vertex with maximum betweeness centrality\n',
        '\t04 - Calculate the degree of a vertex\n',
        '\t05 - Check if the graph is weakly connected\n',
        '\t06 - Calculate the indegree of a vertex\n',
        '\t07 - Plot the degree distribution\n',
        '\t08 - Plot the subgraph of a vertex and it\'s adjacents\n',
        '\t09 - Calculate the attribute assortativity coefficient\n',
        '\t10 - Calculate the attribute mixing matrix\n',
        '\t11 - Calculate the average clustering coefficient\n',
        '\t12 - Calculate the global clustering coefficient\n',
        '\t13 - Calculate the local clustering coefficient of a vertex\n',
        '\t14 - Calculate the reciprocity of a vertex\n',
        '\t15 - Calculate the number of triangles\n',
        '\t16 - Calculate the maximal independent set\n',
        '\t17 - Calculate reciprocity\n'
        '\t18 - Exit\n'
        )
        op = int(input('Type the option you want: '))
        if(op==1):
            eigen = nx.eigenvector_centrality(g, max_iter=1000)
            print(max(eigen.items(), key=operator.itemgetter(1)))
        elif(op==2):
            centrality = nx.degree_centrality(g)
            print(max(centrality.items(), key=operator.itemgetter(1)))
        elif(op==3):
            betweeness = nx.betweenness_centrality(g, normalized=True)
            print(max(betweeness.items(), key=operator.itemgetter(1)))
        elif(op==4):
            vertex_id = int(input('Vertex id: '))
            print(vertex_id, ": ", g.in_degree(vertex_id))
        elif(op==5):
            if( nx.is_weakly_connected(g) == True):
                print("It's weakly connected")
            else:
                print("It isn't weakly connected")
        elif(op==6):
            vertex_id = int(input('Vertex id: '))
            print(vertex_id, ": ", g.degree(vertex_id))
        elif(op==7):
            plot_degree_dist(g)
        elif(op==8):
            vertex_id = int(input('Vertex id: '))
            plot_subgraph_from_vertex(g,vertex_id)
        elif(op==9):
            print(nx.attribute_assortativity_coefficient(g,'developer'))
        elif(op==10):
            print(nx.attribute_mixing_matrix(g,'developer'))
        elif(op==11):
            print("Average clustering coefficient: " + str(nx.average_clustering(g, count_zeros=True)))
        elif(op==12):
            print("Global clustering coefficient: " + str(nx.transitivity(g)))
        elif(op==13):
            vertex_id = int(input('Vertex id: '))
            coef = nx.clustering(g, coef)
            print("Local clustering coefficient of vertex ",vertex_id, ": ",coef)
        elif(op==14):
            vertex_id = int(input('Vertex id: '))
            print("Reciprocity of the vertex ", vertex_id, ": ",nx.reciprocity(g, vertex_id))
        elif(op==15):
            print('Number of triangles: ',nx.triangles(g))
        elif(op==16):
            print('Maximal independent set:')
            print(nx.maximal_independent_set(g))
        elif(op==17):
            print("Reciprocity: ", nx.reciprocity(g))
        elif(op==18):
            run = False

    #calc_plot_strongly_connected_components(g)
    #calc_plot_weakly_connected_components(g)
    #print(nx.k_nearest_neighbors(g))
    '''
    comp = nxc.girvan_newman(g)
    comp_list = list(comp)
    for i in comp_list:
        print(i)
    '''
    #nx.draw_networkx(g, arrows=True, with_labels =False, width = 0.5 )
    #plt.savefig("graph.pdf")
    #
main()