from modules.graph import Graph
import matplotlib as plt
def test():
    g = Graph(filename="github_edges.txt")
    print(g.get_allnode_degree())

test()
