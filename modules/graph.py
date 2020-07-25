import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

class Graph:
    def __init__(self, size=0, is_oriented=False, filename=None):
        self.is_oriented = False
        self.nodes_degree = []
        self.cluster_coeff = []
        self.g = None
        if(filename != None):
            # first line: 1 if oriented 0 otherwise
            # second line: number of nodes
            # other lines Vi Vj weight
            try:
                graph_file = open(filename, 'r')
            except IOError:
                pass
            lines = graph_file.readlines()
            if(int(lines[0]) == 1):
                self.is_oriented = True
                self.g = nx.DiGraph()
            else:
                self.g = nx.Graph()
            self.size = int(lines[1])
            for i in range(0, self.size):
                self.g.add_node(i)
            for i in range(2, len(lines)):
                curr_line = lines[i].split(' ')
                v_i = int(curr_line[0])
                v_j = int(curr_line[1])
                w = 1
                if(len(curr_line) >= 3):
                    w = float(curr_line[2])
                self.g.add_edge(v_i, v_j, weight=w)
        else:
            # size of the adjacency matrix (number of vertices)
            self.size = size
            self.adj_list = {}
            self.is_oriented = is_oriented
            if(self.is_oriented == True):
                self.g = nx.DiGraph()
            else:
                self.g = nx.Graph()
            for i in range(0, self.size):
                self.g.add_node(i)

    def get_node_degree(self, node_index):
        if(node_index >= 0 and node_index < self.size):
            return self.nodes_degree[node_index]
        else:
            return -1


    def print_graph(self):
        print(self.g)

    def plot_degree_dist(self):
        proportion = dict()
        for i in range(0, self.size):
            if self.g.degree[i] not in proportion:
                ind = self.g.degree[i]
                proportion[ind] = 0.0
        for i in range(0, self.size):
            ind = self.g.degree[i]
            proportion[ind] = proportion[ind] + 1
            
        degrees = list(proportion.keys())
        values = list(proportion.values())
        for i in range(0, len(values)):
            values[i] = values[i]/self.size
        plt.title("Degree distribution")
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(degrees, values, marker='.', c='r')
        plt.ylabel('P')
        plt.xlabel('Degree')
        #plt.grid(True)
        plt.show()

    def count_triangles_number(self):
        return nx.triangles(self.g)
    
    def calc_clustering_coeff(self):
        return nx.clustering(self.g)
    
    def calc_average_clustering_coeff(self):
        return nx.average_clustering(self.g)
