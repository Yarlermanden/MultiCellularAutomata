import grispy as gsp
import numpy as np
import torch
import time
from typing import Union, Tuple, Optional
from torch import Tensor

class FixedRadiusNearestNeighbors(object):
    def __init__(self, nodes, radius, batch_size):
        Lbox = 1.0
        #blocks = int(1.0/radius)*batch_size # 10x10 - more efficient to create a bit larger boxes than 
        blocks = int(2.0/radius) # 10x10 - more efficient to create a bit larger boxes than 
        periodic = {0: (-Lbox, Lbox), 1: (-Lbox, Lbox), 2: None}
        self.grid = gsp.GriSPy(nodes.detach().cpu().numpy(), periodic=periodic, N_cells=blocks)

    def get_neighbors(self, node, radius):
        bubble_dist, bubble_ind = self.grid.bubble_neighbors(
            node.detach().cpu().numpy(), distance_upper_bound=radius
        )
        return bubble_dist, bubble_ind


class DataStructure(object):
    def __init__(self, radius, device, wrap_around, batch_size):
        self.radius = radius
        self.device = device
        self.wrap_around = wrap_around
        self.batch_size = batch_size
        self.radius_food = radius*5

    def add_edges(self, graph):

        def update_dist1(dist):
            if dist > 1.0:
                return -2.0 + dist
            elif dist < -1.0:
                return 2.0 + dist
            return dist
        vupdate = np.vectorize(update_dist1)

        edges = []
        attributes = []

        x = graph.x.detach().cpu().numpy()
        s_idx = 0
        for i in range(self.batch_size): #TODO could we simply just vectorize this entire thing?
            time1 = time.perf_counter()
            e_idx = s_idx + graph.subsize[i].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            if len(nodes) == 0:
                continue
            frnn = FixedRadiusNearestNeighbors(nodes, self.radius_food, self.batch_size)
            cell_indices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1).flatten()
            cells = nodes[cell_indices]
            cell_indices += s_idx
            cell_indices = cell_indices.detach().cpu().numpy()
            if len(cells) != 0:
                dists, indices = frnn.get_neighbors(cells, self.radius_food)
                indices = [x + s_idx for x in indices]
                time2 = time.perf_counter()

                tup = [([j,i], [dists[ii][jj], x[i][0]-x[j][0], x[i][1]-x[j][1], 1]) 
                       for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices[ii]) 
                       if i!=j and (dists[ii][jj] < self.radius or graph.x[j,4] == 0)]
                l = [list(t) for t in zip(*tup)]
                if len(l) > 0:
                    edges.extend(l[0])
                    attributes.extend(l[1])
                time3 = time.perf_counter()
            s_idx = e_idx
         
        if len(edges) == 0:
            graph.edge_index = torch.tensor([[]], dtype=torch.long, device=self.device)
            graph.edge_attr = torch.tensor([[]], dtype=torch.float, device=self.device)
            return False
        graph.edge_index = torch.tensor(np.array(edges), dtype=torch.long, device=self.device).T
        edge_attributes = np.array(attributes)
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3]) #restrict to match wraparound
        graph.edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=self.device)
        graph.edge_attr[:, 3] = graph.x[graph.edge_index[0, :], 4] #change edge attr to match whether it connects cells or food

        #print('find: ', time2-time1)
        #print('add: ', time3-time2)
        #print('old: ', time3-time1)
        #print('new: ', time4-time3)
        return True