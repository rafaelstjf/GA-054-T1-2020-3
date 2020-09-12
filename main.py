import numpy as np
import operator
import matplotlib.pyplot as plt
import networkx as nx
import math
from concurrent.futures import ThreadPoolExecutor
from networkx.algorithms.community import greedy_modularity_communities
from infomap import Infomap
is_oriented=True
filename = 'musae_git_edges.csv'
attrib_filename = 'musae_git_target.csv'
num_nodes = 37699
def run_infomap(g):
    im = Infomap("--two-level --directed --tree --ftree")
    print("Creating graph")
    for node in g.nodes:
        im.add_node(node)
    for edge in g.edges:
        im.add_link(edge[0], edge[1])
    print("Running!")
    im.run()
    im.write_flow_tree('arvore.txt')
    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
    print("Result")
    print("\n#node module")
    #for node in im.tree:
    #    if node.is_leaf:
    #        print(node.node_id, node.module_id)

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
    size_ = len(g.nodes)
    proportion_in = dict()
    proportion_out = dict()
    proportion = dict()
    #get in degrees
    for i in range(0, size_):
        if g.in_degree(i) not in proportion_in:
            ind_in = g.in_degree(i)
            proportion_in[ind_in] = 0.0
        if g.out_degree(i) not in proportion_out:
            ind = g.out_degree(i)
            proportion_out[ind] = 0.0
        if g.degree[i] not in proportion:
            proportion[g.degree[i]] = 0.0
    #get out degrees
    for i in range(0, size_):
        ind_in = g.in_degree(i)
        ind_out = g.out_degree(i)
        proportion_in[ind_in] += 1
        proportion_out[ind_out] += 1
        proportion[g.degree[i]] += 1
    degrees = list(proportion.keys())
    values = list(proportion.values())
    for i in range(0, len(values)):
        values[i] = values[i]/size_
    plt.clf()
    plt.ylabel('P')
    plt.xlabel('Degree')
    plt.title("Degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(degrees, values, marker='.', c='#e36387')
    plt.show()
    #indegree
    in_degrees = list(proportion_in.keys())
    in_values = list(proportion_in.values())
    for i in range(0, len(in_values)):
        in_values[i] = in_values[i]/size_
    plt.clf()
    plt.ylabel('P')
    plt.xlabel('Degree')
    plt.title("In degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0,size_)
    plt.scatter(in_degrees, in_values, marker='.', c='#e36387')
    plt.show()
    #outdegree
    out_degrees = list(proportion_out.keys())
    out_values = list(proportion_out.values())
    for i in range(0, len(out_values)):
        out_values[i] = out_values[i]/size_
    plt.clf()
    plt.title("Out degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0,size_)
    plt.ylabel('P')
    plt.xlabel('Degree')
    plt.scatter(out_degrees, out_values, marker='.', c='#e36387')
    plt.show()
def plot_degree_dist_binned(g):
    size_ = len(g.nodes)
    proportion_in = dict()
    for i in range(0, size_):
        if g.in_degree(i) not in proportion_in:
            ind_in = g.in_degree(i)
            proportion_in[ind_in] = 0.0
    biggest_degree = max(proportion_in.keys())
    num_bins_in = math.ceil(math.log2(biggest_degree) + 1)
    print(num_bins_in)
    keys = list(proportion_in.keys())
    it = keys[1]
    it_ind = 1
    values = [proportion_in[0]/size_]
    degrees  = [0]
    for i in range(1, num_bins_in+1):
        bin_size = 0
        num_nodes = 0
        av = 0
        while(it >= (math.pow(2, i-1)) and it <= (math.pow(2, i) - 1) and it < len(proportion_in)):
            num_nodes+=it
            bin_size+=1
            av+=it
            it_ind+=1
            it=keys[it_ind]
        if(bin_size>0):
            av = av/bin_size
            values.append(num_nodes/bin_size)
            degrees.append(av)
    plt.clf()
    plt.title("In degree distribution")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1,size_)
    plt.ylabel('P')
    plt.xlabel('Degree')
    plt.scatter(degrees, values, marker='.', c='#e36387')
    plt.show()
def plot_bar_char(x, y):
    plt.bar(x, y, width=1, color='#e36387')
    plt.show()
def plot_graph_by_dev(g):
    colors=[]
    elements = nx.get_node_attributes(g, 'developer')
    for i in elements.items():
        if(i[1] == '0'): #web
            colors.append('#f09ae9')
        else:
            colors.append('#ffd36b') #machine learning
    nx.draw(g, node_size_=10, node_color=colors, width=0.1, edge_color='#dfd3c3', arrows=True, arrowsize_=2)
    plt.clf()
    plt.show()
def calc_plot_strongly_connected_components(g):
    components = nx.number_strongly_connected_components(g)
    print('Size of the strongly connected components: ',components)
    '''
    size_s = [len(c) for c in sorted(g.calc_connected_components(), key=len, reverse=True)]
    dat = {}
    for i in range(0, len(size_s)):
        if i not in dat:
            dat[i] = size_s[i]
    
    plt.scatter(dat.keys(), dat.values(), color='#e36387')
    plt.show()
    '''


def calc_plot_weakly_connected_components(g):
    components = nx.number_weakly_connected_components(g)
    print('Size of the weakly connected components: ',components)
    '''
    size_s = [len(c) for c in sorted(g.calc_weakly_connected_components(), key=len, reverse=True)]
    dat = {}
    for i in range(0, len(size_s)):
        if i not in dat:
            dat[i] = size_s[i]
    
    plt.scatter(dat.keys(), dat.values(), color='#e36387')
    plt.show()
    '''

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
    node_size_s = []
    vertices.append(node_index)
    for e in g.edges(node_index):
        vertices.append(e[1])
    for e in g.in_edges(node_index):
        vertices.append(e[0])
    sub_g = g.subgraph(vertices)
    for i in sub_g.nodes():
        if(i == node_index):
            node_size_s.append(10)
            colors.append('#00bcd4')
        else:
            node_size_s.append(0.5)
            colors.append('#b52b65')
    nx.draw(sub_g, node_size_=node_size_s, node_color=colors, width=0.1, edge_color='#dfd3c3', arrows=True, arrowsize_=2)
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
        '\t18 - Calculate the vertex with maximum indegree centrality\n'
        '\t19 - Calculate the vertex with maximum outdegree centrality\n'
        '\t20 - Find communities using infomap\n'
        '\t21 - Exit\n'
        )
        op = int(input('Type the option you want: '))
        if(op==1):
            eigen = nx.eigenvector_centrality(g, max_iter=1000)
            print(max(eigen.items(), key=operator.itemgetter(1)))
        elif(op==2):
            centrality = nx.degree_centrality(g)
            print('Degree centrality')
            print(max(centrality.items(), key=operator.itemgetter(1)))
        elif(op==3):
            betweeness = nx.betweenness_centrality(g, normalized=True)
            print(max(betweeness.items(), key=operator.itemgetter(1)))
        elif(op==4):
            vertex_id = int(input('Vertex id: '))
            print(vertex_id, ": ", g.degree(vertex_id))
            print(vertex_id, " In degree: ", g.in_degree(vertex_id))
            print(vertex_id, " Out degree: ", g.out_degree(vertex_id))
        elif(op==5):
            if( nx.is_weakly_connected(g) == True):
                print("It's weakly connected")
            else:
                print("It isn't weakly connected")
        elif(op==6):
            vertex_id = int(input('Vertex id: '))
            print(vertex_id, ": ", g.in_degree(vertex_id))
        elif(op==7):
            #plot_degree_dist(g)
            plot_degree_dist_binned(g)
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
            print('In degree centrality')
            centrality = nx.in_degree_centrality(g)
            print(max(centrality.items(), key=operator.itemgetter(1)))
        elif(op==19):
            print('Out degree centrality')
            centrality = nx.out_degree_centrality(g)
            print(max(centrality.items(), key=operator.itemgetter(1)))
        elif(op==20):
            #undir_g = g.to_undirected()
            #c = list(greedy_modularity_communities(undir_g))
            #sorted(c[0])
            run_infomap(g)
        elif(op==21):
            run = False

    calc_plot_strongly_connected_components(g)
    calc_plot_weakly_connected_components(g)
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