from modules.graph import Graph



def test():
    g = Graph(filename="g1.txt")
    g.print_adjmatrix()
    g.get_allnode_degree()
    g.add_edge(0,2, 3)
    g.print_adjmatrix()

test()
