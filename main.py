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
    print(sizes)

def count_triangles_number(g):
    return nx.triangles(g)

def calc_clustering_coeff(g):
    return nx.clustering(g)

def calc_global_clustering_coeff(g):
    return nx.transitivity(g)

def calc_average_clustering_coeff(g):
    return nx.average_clustering(g, count_zeros=True)
def calc_max_independent_set(g):
    if is_oriented == True:
        return nx.maximal_independent_set(g)
    else:
        return None
def calc_reciprocity(g):
    return nx.reciprocity(g)
def calc_connected_components(g):
    if is_oriented == True:
        print('Number of connected components: ' + str(nx.number_strongly_connected_components(g)))
        return nx.strongly_connected_components(g)
    else:
        return nx.connected_components(g)
def check_if_weakly_connected(g):
    if is_oriented == True:
        return nx.is_weakly_connected(g)
    else:
        return False
def calc_weakly_connected_components(g):
    if(is_oriented == True):
        print('Number of weakly connected components: ' + str(nx.number_weakly_connected_components(g)))
        return nx.weakly_connected_components(g)
    else:
        return None
def calc_degree_centrality(g):
    return nx.degree_centrality(g)
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

def calc_eigenvector_centrality(g):
    return nx.eigenvector_centrality(g, max_iter=1000)

def calc_betweeness_centrality(g):
    return nx.betweenness_centrality(g, normalized=True)

def print_subgraph_from_vertex(g, node_index):
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

    #g.plot_degree_dist()
    #
    '''
    if(g.check_if_weakly_connected() == True):
        print("fracamente conectado")
    else:
        print("nao eh fracamente conectado")
    '''
    #centrality = calc_degree_centrality(g)
    #print(max(centrality.items(), key=operator.itemgetter(1)))
    #eigen = g.calc_eigenvector_centrality()
    #print(max(eigen.items(), key=operator.itemgetter(1)))
    #betweeness = g.calc_betweeness_centrality()
    #print(max(betweeness.items(), key=operator.itemgetter(1)))
    #print("2347: " + str(g.get_node_degree(2347)))
    #print("31890: " + str(g.get_node_degree(31890)))
    #print("31890: " + str(g.get_node_indegree(31890)))
    #g.print_subgraph_from_vertex(31890)
    #g.print_subgraph_from_vertex(31890)
    #calc_plot_strongly_connected_components(g)
    #calc_plot_weakly_connected_components(g)
    #print(nx.attribute_assortativity_coefficient(g,'developer'))
    #print(nx.degree_assortativity_coefficient(g))
    #print(nx.attribute_mixing_matrix(g,'developer'))
    #print(nx.k_nearest_neighbors(g))
    #comp = nxc.girvan_newman(g)
    #tuple(sorted(c) for c in next(comp))
    #nx.draw_networkx(g, arrows=True, with_labels =False, width = 0.5 )
    #plt.savefig("graph.pdf")
    #print("Coeff de clusterizacao medio: " + str(calc_average_clustering_coeff(g)))
    #print("Coeff de clusterizacao global: " + str(calc_global_clustering_coeff(g)))
    #print_graph(g)
    #coef = nx.clustering(g, 31890)
    #print("Coeff de clusterizacao vertice " + str(31890) +": "  + str(coef))
    #coef = nx.clustering(g, 2347)
    #print("Coeff de clusterizacao vertice " + str(2347) +": "  + str(coef))
main()