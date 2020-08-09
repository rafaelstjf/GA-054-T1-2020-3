from modules.graph import Graph
import operator
import matplotlib.pyplot as plt
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


def main():
    g = Graph(filename="github_edges.txt")
    #g.plot_degree_dist()
    #
    '''
    if(g.check_if_weakly_connected() == True):
        print("fracamente conectado")
    else:
        print("nao eh fracamente conectado")
    '''
    #centrality = g.calc_degree_centrality()
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
    calc_plot_weakly_connected_components(g)
main()