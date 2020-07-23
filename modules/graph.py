import numpy as np


class Graph:
    def __init__(self, size=0, is_oriented=False, filename=None):
        self.is_oriented = False
        self.all_degree = []
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
            self.calc_allnode_degree()
        else:
            # size of the adjacency matrix (number of vertices)
            self.size = size
            self.adj_list = {}
            self.is_oriented = is_oriented
            for i in range(0, self.size):
                self.adj_list.append([])
                self.all_degree.append(0)

    def calc_node_degree(self, node_index):
        in_degree = 0
        out_degree = 0
    #    if(self.is_oriented):
   #         in_degree = len(self.adj_list[node_index][1][1])
  #      for i in range(0, self.size):
 #           for j in range (0, len(self.adj_list[i][1]if(self.adj_matrix.item((node_index, i)) > 0.0):
#                in_degree += self.adj_matrix.item((node_index, i))
        return in_degree + out_degree

    def calc_allnode_degree(self):
        in_degree = []
        out_degree = []
        self.all_degree = []
        for i in range (0, self.size):
            in_degree.append(0)
            out_degree.append(0)
            self.all_degree.append(0)
        for i in range(0, self.size):
            out_degree[i] = len(self.adj_list[i])
            if self.is_oriented == True:
                for j in range(0, len(self.adj_list[i])):
                    in_degree[self.adj_list[i][j][0]] = in_degree[self.adj_list[i][j][0]] + 1
        for i in range(0, self.size):
            self.all_degree[i] = in_degree[i] + out_degree[i]
        return self.all_degree

    def get_node_degree(self, node_index):
        if(node_index >= 0 and node_index < self.size):
            return self.all_degree[node_index]
        else:
            return -1

    def get_allnode_degree(self):
        return self.all_degree

    def print_adjmatrix(self):
        for i in range(0, self.size):
            print(str(i) + "-> " + str(self.adj_list[i]))


    def add_edge(self, v1, v2, weight=1):
        if((v1 >= 0 and v1 < self.size) and (v2 >= 0 and v2 < self.size)):
            self.adj_list[v1][1].append(v2, weight)
            if(not self.is_oriented):
                self.adj_list[v2][1].append(v1, weight)
