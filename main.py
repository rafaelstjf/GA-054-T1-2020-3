from modules.graph import Graph
def test():
    g = Graph(filename="github_edges.txt")
    g.plot_degree_dist()
    print("Number of triangles: " + str(g.count_triangles_number()))
    #g.print()
    #g.calc_clustering_coeff()
    #print(g.calc_average_clustering_coeff())
test()
