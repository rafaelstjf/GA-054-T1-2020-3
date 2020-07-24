import numpy as np
import matplotlib.pyplot as plt
import math
from concurrent.futures import ThreadPoolExecutor

class Graph:
    def __init__(self, size=0, is_oriented=False, filename=None):
        self.is_oriented = False
        self.nodes_degree = []
        self.cluster_coeff = []
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
            self.size = int(lines[1])
            self.adj_list = []
            for i in range(0, self.size):
                self.adj_list.append([])
            for i in range(2, len(lines)):
                curr_line = lines[i].split(' ')
                v_i = int(curr_line[0])
                v_j = int(curr_line[1])
                w = 1
                if(len(curr_line) >= 3):
                    w = float(curr_line[2])
                self.adj_list[v_i].append((v_j, w))
                if(self.is_oriented == False):
                    self.adj_list[v_j].append((v_i, w))
            self.calc_nodes_degree()
        else:
            # size of the adjacency matrix (number of vertices)
            self.size = size
            self.adj_list = {}
            self.is_oriented = is_oriented
            for i in range(0, self.size):
                self.adj_list.append([])
                self.nodes_degree.append(0)
        for i in range(0, self.size):
            self.cluster_coeff.append(0)

    def calc_nodes_degree(self):
        in_degree = []
        out_degree = []
        self.nodes_degree = []
        for i in range (0, self.size):
            in_degree.append(0)
            out_degree.append(0)
            self.nodes_degree.append(0)
        for i in range(0, self.size):
            out_degree[i] = len(self.adj_list[i])
            if self.is_oriented == True:
                for j in range(0, len(self.adj_list[i])):
                    in_degree[self.adj_list[i][j][0]] = in_degree[self.adj_list[i][j][0]] + 1
        for i in range(0, self.size):
            self.nodes_degree[i] = in_degree[i] + out_degree[i]
        return self.nodes_degree

    def get_node_degree(self, node_index):
        if(node_index >= 0 and node_index < self.size):
            return self.nodes_degree[node_index]
        else:
            return -1

    def get_allnode_degree(self):
        return self.nodes_degree

    def print(self):
        for i in range(0, self.size):
            print(str(i) + "-> " + str(self.adj_list[i]))

    def add_edge(self, v1, v2, weight=1):
        if((v1 >= 0 and v1 < self.size) and (v2 >= 0 and v2 < self.size)):
            self.adj_list[v1][1].append(v2, weight)
            if(not self.is_oriented):
                self.adj_list[v2][1].append(v1, weight)

    def plot_degree_dist(self):
        self.calc_nodes_degree()
        proportion = dict()
        for i in range(0, len(self.nodes_degree)):
            if self.nodes_degree[i] not in proportion:
                ind = self.nodes_degree[i]
                proportion[ind] = 0.0
        for i in range(0, len(self.nodes_degree)):
            ind = self.nodes_degree[i]
            proportion[ind] = proportion[ind] + 1
            
        degrees = list(proportion.keys())
        values = list(proportion.values())
        for i in range(0, len(values)):
            values[i] = values[i]/self.size
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(degrees, values, marker='.', c='r')
        plt.ylabel('P')
        plt.xlabel('Degree')
        #plt.grid(True)
        plt.show()

    def count_triangles_number(self):
        num = 0
        for i in range (0, self.size):
            for j in range (0, len(self.adj_list[i])):
                ind = self.adj_list[i][j][0]
                for k in range(0, len(self.adj_list[ind])):
                    if self.adj_list[ind][k][0] == i:
                        num = num + 1
                        break
        if self.is_oriented == True:
            num = math.floor(num /3)
        else:
            num = math.floor(num/6)
        return num
    
    def __calc_clustering_coeff_aux(self, index):
        summation = 0.0
        temp = self.adj_list[index]
        for j in range(0, len(temp)):
            ind = temp[j][0]
            for k in range(0, len(self.adj_list[ind])):
                for l in range(j, len(temp)):
                    if(self.adj_list[ind][k][0] == temp[l][0]):
                        summation+=1
                        break
        coeff = 0
        if(self.nodes_degree[index] > 1):
            if(self.is_oriented == False):
                coeff = (2* summation)/(self.nodes_degree[index] * (self.nodes_degree[index] - 1))
            else:
                coeff = (summation)/(self.nodes_degree[index] * (self.nodes_degree[index] - 1))
        return coeff

    def calc_clustering_coeff(self):
        executor = ThreadPoolExecutor(5)
        for i in range(0, self.size):
            res = executor.submit(self.__calc_clustering_coeff_aux, (i))
            self.cluster_coeff[i] = res.result()
        executor.shutdown(wait=True)
        return self.cluster_coeff
    
    def calc_average_clustering_coeff(self):
        summation = 0.0
        for i in range(0, self.size):
            summation+= self.cluster_coeff[i]
        return summation/self.size
