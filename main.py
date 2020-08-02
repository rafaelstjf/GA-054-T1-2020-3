from modules.graph import Graph
import matplotlib.pyplot as plt
def plot_bar_char(x, y):
    plt.bar(x, y, width=1, color='#e36387')
    plt.show()

def test():
    g = Graph(filename="github_edges.txt")
    #g.plot_degree_dist()
    #a = (g.calc_connected_components())
    #print([len(c) for c in sorted(a, key=len, reverse=True)])
    '''
    if(g.check_if_weakly_connected() == True):
        print("fracamente conectado")
    else:
        print("nao eh fracamente conectado")
    '''
    #centrality = g.calc_degree_centrality()
    eigen = g.calc_eigenvector_centrality()
    print(eigen)
test()
