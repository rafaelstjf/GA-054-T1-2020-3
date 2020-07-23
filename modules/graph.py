import numpy as np
class Graph:
    def __init__(self, size = 0, is_oriented = False, filename = None):
        self.is_oriented = False
        if(filename!= None):
            #first line: 1 if oriented 0 otherwise
            #second line: number of nodes
            #other lines Vi Vj weight
            try:
                graph_file = open(filename, 'r')
            except IOError:
                pass
            lines = graph_file.readlines()
            if(int(lines[0])==1):
                self.is_oriented = True
            self.size = int(lines[1])
            self.adj_matrix = np.zeros((self.size, self.size))
            for i in range (2, len(lines)):
                curr_line = lines[i].split(' ')
                v_i = int(curr_line[0])
                v_j = int(curr_line[1])
                w = 1
                if(len(curr_line)>=3):
                    w = float(curr_line[2])
                self.adj_matrix[v_i, v_j] =  w
                if(self.is_oriented == False):
                    self.adj_matrix[v_j, v_i] = w
        else:
            self.size = size #size of the adjacency matrix (number of vertices)
            self.adj_matrix = np.zeros((size, size))
            self.is_oriented = is_oriented
        self.all_degree = np.zeros(size)

    def calc_node_degree(self, node_index):
        in_degree = 0
        out_degree = 0
        if(self.is_oriented):
            for i in range (0, self.size):
                if(self.adj_matrix.item((i, node_index)) > 0.0):
                    out_degree+=self.adj_matrix.item((i, node_index))
        for i in range (0, self.size):
                if(self.adj_matrix.item((node_index, i)) > 0.0):
                    in_degree+=self.adj_matrix.item((node_index, i))
        return in_degree + out_degree

    def calc_allnode_degree(self):
        self.all_degree = np.array(self.size)
        for k in range (0, self.size):
            for ind in range (0, self.size):
                in_degree = 0
                out_degree = 0
                if(self.is_oriented):
                    for i in range (0, self.size):
                        if(self.adj_matrix.item((i, k)) > 0.0):
                            out_degree+=self.adj_matrix.item((i, k))
                for i in range (0, self.size):
                    if(self.adj_matrix.item((k, i)) > 0.0):
                        in_degree+=self.adj_matrix.item((k, i))
                self.all_degree[ind] = in_degree + out_degree
        return self.all_degree

    def get_node_degree(self, node_index):
        if(node_index >= 0 and node_index < self.size):
            return self.all_degree[node_index]
        else:
            return -1

    def get_allnode_degree(self):
        return self.all_degree

    def print_adjmatrix(self):
        print (self.adj_matrix)
    
    def add_edge(self, v1, v2, weight = 1):
        if((v1 >= 0 and v1 < self.size) and (v2 >= 0 and v2 < self.size)):
            if(self.is_oriented):
                self.adj_matrix[v1, v2] = weight
            else:
                self.adj_matrix[v1, v2] = weight
                self.adj_matrix[v2, v1] = weight
    