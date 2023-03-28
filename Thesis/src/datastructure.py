import grispy as gsp
import numpy as np
import torch
import time
from typing import Union, Tuple, Optional
from torch import Tensor
from graphUtils import *

from sklearn.neighbors import KDTree

class FixedRadiusNearestNeighbors(object):
    def __init__(self, nodes, radius, batch_size, scale):
        Lbox = float(scale)
        #blocks = int(2.0*scale/radius) # 10x10 - more efficient to create a bit larger boxes than 
        #blocks = int(10) # 10x10 - more efficient to create a bit larger boxes than 
        blocks = int(1) # 10x10 - more efficient to create a bit larger boxes than 
        periodic = {0: (-Lbox, Lbox), 1: (-Lbox, Lbox)}
        self.grid = gsp.GriSPy(nodes.detach().cpu().numpy(), periodic=periodic, N_cells=blocks)

    def get_neighbors(self, node, radius):
        bubble_dist, bubble_ind = self.grid.bubble_neighbors(
            #TODO could experiment with changing distance_upper_bound radius to 3D and low z so it doesn't combine batches
            node.detach().cpu(), distance_upper_bound=radius
        )
        return bubble_dist, bubble_ind

class FixedRadiusNearestNeighbors2(object):
    def __init__(self, nodes, radius, batch_size, scale, dense):
        nodes = nodes.detach().cpu().numpy()
        if dense: self.tree = KDTree(nodes, leaf_size=40)
        else: self.tree = KDTree(nodes, leaf_size=20)

    def get_neighbors(self, node, radius):
        return self.tree.query_radius(node.detach().cpu().numpy(), radius, return_distance=True)

class DataStructure(object):
    def __init__(self, radius, device, wrap_around, batch_size, scale):
        self.radius = radius
        self.device = device
        self.wrap_around = wrap_around
        self.batch_size = batch_size
        self.radius_food = radius*5
        self.scale = scale

    def update_dist1(self, dist):
        if dist > self.scale:
            return -2*self.scale + dist
        elif dist < -self.scale:
            return 2*self.scale + dist
        return dist

    def update_norm_dist(self, norm):
        if norm > 1.4142135624:
            return -2.828427 + norm
        elif norm < -1.4142135624:
            return 2.828427 + norm
        return norm

    def add_edges(self, graph):
        vupdate = np.vectorize(self.update_dist1)
        edges = []

        x = graph.x[:, :2].detach().cpu().numpy()
        s_idx = 0
        for batch_idx in range(self.batch_size): #TODO could we simply just vectorize this entire thing?
            e_idx = s_idx + graph.subsize[batch_idx].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            if len(nodes) == 0: continue
            cell_indices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 1).flatten()
            cells = nodes[cell_indices]
            food_indices = torch.nonzero(graph.x[s_idx:e_idx, 4] == 0).flatten()
            food = nodes[food_indices]

            if len(cells) != 0:
                cell_indices += s_idx
                food_indices += s_idx
                cell_indices = cell_indices.detach().cpu().numpy()
                food_indices = food_indices.detach().cpu().numpy()

                frnn_food = FixedRadiusNearestNeighbors2(food, self.radius_food, self.batch_size, self.scale, False)
                indices_food, dists_food = frnn_food.get_neighbors(cells, self.radius_food)
                indices_food = [food_indices[x] for x in indices_food]
                edges_food = [[j, i, dists_food[ii][jj], *(x[i]-x[j]), 0]
                            for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices_food[ii])]

                frnn_cell = FixedRadiusNearestNeighbors2(cells, self.radius, self.batch_size, self.scale, True)
                indices_cells, dists_cells = frnn_cell.get_neighbors(cells, self.radius)                       
                indices_cells = [cell_indices[x] for x in indices_cells]
                edges_cells = [[j, i, dists_cells[ii][jj], *(x[i]-x[j]), 1]
                       for ii, i in enumerate(cell_indices) for jj, j in enumerate(indices_cells[ii])
                       if i!=j]

                if len(edges_food) > 0:
                    edges.extend(edges_food)
                if len(edges_cells) > 0:
                    edges.extend(edges_cells)
            s_idx = e_idx
         
        if len(edges) == 0:
            graph.edge_index = torch.tensor([[]], dtype=torch.long, device=self.device)
            graph.edge_attr = torch.tensor([[]], dtype=torch.float, device=self.device)
            return False
        edges = np.array(edges)
        graph.edge_index = torch.tensor(edges[:, :2], dtype=torch.long, device=self.device).T
        edge_attributes = edges[:, 2:]
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3]) #restrict to match wraparound
        graph.edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=self.device)
        return True

    def add_edges_global(self, graph):
        edges = []
        attributes = []

        x = graph.x.detach().cpu().numpy()
        s_idx = 0
        for batch_idx in range(self.batch_size):
            e_idx = s_idx + graph.subsize[batch_idx].detach().cpu().numpy()
            nodes = graph.x[s_idx:e_idx, :2]
            if len(nodes) == 0: continue
            food_idxs = set()

            cell_filter = graph.x[s_idx:e_idx, 4] == 1
            cells = nodes[cell_filter]
            if len(cells) != 0:
                foods = nodes[torch.bitwise_not(cell_filter)]
                frnn_food = FixedRadiusNearestNeighbors(foods, self.radius_food, self.batch_size)
                dists, indices = frnn_food.get_neighbors(cells, self.radius_food)
                [food_idxs.add(x) for i in indices for x in i]
                #convert ids to hashset or otherwise make unique

                #Make edges from all cells to all of these food in ids

                #Make edges between all cells and cells

                #simply make the edges as [0, x_dist, y_dist, 0/1] and update dist later


                #... we need to actually have the distance between all of the cell and food.........
                #should propably consider whether all nodes should have the same edge to each food or only their local information - like what should we do with distance...

                #for both cell and food, we could compute the actual distance afterwards, by just adding the x and y distance at first...


    def add_edges_with_global_node(self, graph):
        '''Add edges according to a organism containing a global node for long range communication'''
        #Add edges as usual
        vupdate = np.vectorize(self.update_dist1)
        vcompute_dist = np.vectorize(self.update_norm_dist)
        any = self.add_edges(graph)
        if not any: return False
        edges = []
        attributes = []

        #do all of this pr batch
        s_idx = 0
        for batch_idx in range(self.batch_size):
            e_idx = s_idx + graph.subsize[batch_idx].detach().cpu().numpy()

            nodes = graph.x[s_idx:e_idx]
            cell_idx = torch.nonzero(nodes[:, 4] == 1).flatten()
            cells = nodes[cell_idx, :2]
            global_node_idx = torch.nonzero(nodes[:, 4] == 2).squeeze()
            global_node = nodes[global_node_idx, :2].squeeze()

            cell_idx+=s_idx
            global_node_idx+=s_idx

            #each of these edges could possibly contain a new number, so we know that these nodes come from some form of global information
            #we could then train a new conv for this connection specifically
            #TODO should we add attributes depending on the nodes position as well? and in that case, should we update the node during model as well?
            tup = [([cell_idx[i], global_node_idx], 
                    [0, global_node[0]-cells[i, 0], global_node[1]-cells[i,1], 2],
                    [global_node_idx, cell_idx[i]], 
                    [0, -global_node[0]+cells[i, 0], -global_node[1]+cells[i,1], 2])
                   for i in range(len(cells))]
            l = [list(t) for t in zip(*tup)]
            if len(l) > 0:
                edges.extend(l[0])
                edges.extend(l[2])
                attributes.extend(l[1])
                attributes.extend(l[3])
            s_idx = e_idx

        if len(edges) == 0:
            return False
        edge_attributes = np.array(attributes)
        edge_attributes[:, 0] = np.linalg.norm(edge_attributes[:, 1:3], axis=1)
        edge_attributes[:, 0] = vcompute_dist(edge_attributes[:, 0])
        edge_attributes[:, 1:3] = vupdate(edge_attributes[:, 1:3]) #restrict to match wraparound

        graph.edge_index = torch.concat((torch.tensor(np.array(edges), dtype=torch.long, device=self.device).T, graph.edge_index), dim=1)
        graph.edge_attr = torch.concat((torch.tensor(edge_attributes, dtype=torch.float, device=self.device), graph.edge_attr), dim=0)
        return True
        
        #Ignore this node when plotting and in the other checks... like minimum edges ...
        #ignore when computing shortest longest path
        #ignore when visualizing
        #ignore when computing all metrics
        #ignore when computing minimum edges... who to remove, consume...

        #in general implement as a new type of node, and ensure everything checks for specifically food and cells and not assumes just the opposite

    def add_edges_rigid(self, graph):
        '''Add edges according to a rigid organism, where all cells stay connected'''
        ...
        #don't need to readd edges for cell to cell as these are rigid
        #Only need to add edges for food