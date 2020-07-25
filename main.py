from modules.graph import Graph
def test():
    g = Graph(filename="github_edges.txt")
    g.print_graph()
    g.plot_degree_dist()
    #print(g.calc_clustering_coeff())
test()
