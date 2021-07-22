#! /usr/bin/env python
'''
    A Cache Network (Py3)
'''
from abc import ABCMeta, abstractmethod
from Caches import PriorityCache, EWMACache, LMinimalCache
from helpers import uniqueify
from networkx import Graph, DiGraph, shortest_path
import networkx
import random
from cvxopt import spmatrix, matrix
from cvxopt.solvers import lp, qp
from simpy import *
from scipy.stats import rv_discrete
import numpy as np
from numpy.linalg import matrix_rank
import logging
import argparse
import itertools
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import topologies


class CONFIG(object):
    QUERY_MESSAGE_LENGTH = 0.0
    RESPONSE_MESSAGE_LENGTH = 0.0
    EXPLORE_MESSAGE_LENGTH = 0.0
    EXPLORE_RESPONSE_MESSAGE_LENGTH = 0.0


def pp(l):
    return ' '.join(map(str, l))


class Demand:
    """ A demand object. Contains the item requested, the path a request follows, as a list, and the
        rate with which requests are generated. Tallies count various metrics.

        Attributes:
            item: the id of the item requested
            path: a list of nodes to be visited
            rate: the rate with which this request is generated
            query_source: first node on the path
            item_source: last node on the path
    """

    def __init__(self, item, path, rate):
        """ Initialize a new request.
        """
        self.item = item
        self.path = path
        self.rate = rate

        self.query_source = path[0]
        self.item_source = path[-1]

    def __str__(self):
        return Demand.__repr__(self)

    def __repr__(self):
        return 'Demand('+','.join(map(str, [self.item, self.path, self.rate]))+')'

    def succ(self, node):
        """ The successor of a node in the path. 
        """
        path = self.path
        if node not in path:
            return None
        i = path.index(node)
        if i+1 == len(path):
            return None
        else:
            return path[i+1]

    def pred(self, node):
        """The predecessor of a node in the path.
        """
        path = self.path
        if node not in path:
            return None
        i = path.index(node)
        if i-1 < 0:
            return None
        else:
            return path[i-1]


class Message(object):
    """A Message object. 

       Attributes:
           header: the header of the message (e.g., query_message, response_message
           payload: the payload, can be set by the programmer
           length: length, to be used in transmission delay calculations
           stats: statistics collected as the message traverses nodes
           u_bit: indicates if the message is going upstream or downstream
    """

    def __init__(self, header, payload, length, stats, u_bit):
        self.header = header
        self.payload = payload
        self.length = length
        self.u_bit = u_bit
        if stats == None:
            self.stats = {}
            self.stats['delay'] = 0.0
            self.stats['hops'] = 0.0
            self.stats['weight'] = 0.0
            self.stats['downweight'] = 0.0
        else:
            self.stats = stats

    def __str__(self):
        return Message.__repr__(self)

    def __repr__(self):
        return pp(['Message(', self.header, ',', self.payload, ',', self.length, ',', self.stats, ')'])


class QueryMessage(Message):
    """
         A query message.
    """

    def __init__(self, d, query_id, stats=None):
        Message.__init__(self, header=("QUERY", d, query_id), payload=None,
                         length=CONFIG.QUERY_MESSAGE_LENGTH, stats=stats, u_bit=True)


class ResponseMessage(Message):
    """
         A response message.
    """

    def __init__(self, d, query_id, stats=None):
        Message.__init__(self, header=("RESPONSE", d, query_id), payload=None,
                         length=CONFIG.RESPONSE_MESSAGE_LENGTH, stats=stats, u_bit=False)


class CacheNetwork(DiGraph):
    """A cache network. 

      A cache network comprises a weighted graph and a list of demands. Each node in the graph is associated with a cache of finite capacity.
      NetworkCaches must support a message receive operation, that determines how they handle incoming messages.

      The cache networks handles messaging using simpy stores and processes. In partiqular, each cache, edge and demand is associated with a 
      Store object, that receives, stores, and processes messages from simpy processes.

      In more detail:
      - Each demand is associated with two processes, one that generates new queries, and one that monitors and logs completed queries (existing only for logging purposes)
      - Each cache/node is associated with a process that receives messages, and processes them, and produces new messages to be routed, e.g., towards neigboring edges
      - Each edge is associated with a process that receives messages to be routed over the edge, and delivers them to the appropriate target node.
        During "delivery", messages are (a) delayed, according to configuration parameters, and (b) statistics about them are logged (e.g., number of hops, etc.)

      Finally, a global monitoring process computes the social welfare at poisson time intervals.

    """

    def __init__(self, G, cacheGenerator, demands, item_sources, capacities, weights, delays, warmup=0, monitoring_rate=1.0, demand_change_rate=0, demand_min=1.0, demand_max=1.0):
        self.isWireless = False
        self.env = Environment()
        self.warmup = warmup
        self.demandstats = {}
        self.sw = {}
        self.funstats = {}
        self.optstats = {}
        self.monitoring_rate = monitoring_rate
        self.demand_change_rate = demand_change_rate
        self.demand_min = demand_min
        self.demand_max = demand_max

        self.node = {}
        self.edge = {}
        DiGraph.__init__(self, G)
        for x in self.nodes():
            self.node[x] = {}
            self.node[x]['cache'] = cacheGenerator(capacities[x], x)
            self.node[x]['pipe'] = Store(self.env)
        for e in self.edges():
            x = e[0]
            self.edge[x] = {}
        for e in self.edges():
            x = e[0]
            y = e[1]
            self.edge[x][y] = {}
            self.edge[x][y]['weight'] = weights[e]
            self.edge[x][y]['delay'] = delays[e]
            self.edge[x][y]['pipe'] = Store(self.env)

        self.demands = {}
        self.item_set = set()

        for d in demands:
            self.demands[d] = {}
            self.demands[d]['pipe'] = Store(self.env)
            self.demands[d]['queries_spawned'] = 0
            self.demands[d]['queries_satisfied'] = 0
            self.demands[d]['queries_logged'] = 0.0
            self.demands[d]['pending'] = set([])
            self.demands[d]['stats'] = {}
            self.item_set.add(d.item)

        for item in item_sources:
            for source in item_sources[item]:
                # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE
                self.node[source]['cache'].makePermanent(item)

        for d in self.demands:
            self.env.process(self.spawn_queries_process(d))
            self.env.process(self.demand_monitor_process(d))
            if self.demand_change_rate > 0.0:
                self.env.process(self.demand_change_process(d))

        if self.demand_change_rate > 0.0:
            self.env.process(self.compute_opt_process())

        for e in self.edges():
            self.env.process(self.message_pusher_process(e))

        for x in self.nodes():
            self.env.process(self.cache_process(x))

        self.env.process(self.monitor_process())

    def initWireless(self, T, V, SC, gains, cache_type, power_max = 10, sinr = 0.1, noise = 0.1):
        """ Separate init function for wireless scenario.
        
            Sets/constructs the following parameters/variables:
                 noise: Currently sets one constant noise for the entire network (Default value 1)
                 gains: Gets channel gain on each edge from generated HetNet topology (No default value)
                 linkcost: Currently just inverse of channel gain
                 power: Currently distributes maximum per-node power to all outgoing edges from that node equally (Power max default value 100)
                 sinrconst: Currently randomizes (using uniform distribution) the SINR constraint (gamma) of each edge (Default is between 0.1 < sinrconst < 1.0)
                 A and b: The inequality A*s >= b defines the SINR constraint for the entire network.
                          A is a matrix populated with coefficients made of gains and SINR constraints, b is a vector of SINR constraints times noise and s is the vector of power variables, ordered by edge number found in self.edges()
        """
        self.isWireless = True
        # Wireless stats dict
        self.wirelessStats = {}
        self.wirelessStats['power_by_slot'] = []
        self.wirelessStats['frac_by_slot'] = []
        # Additional wireless parameters
        self.T = T
        self.graphsize = V
        #self.PowerCap = [power_max * 3] + [power_max] * int(SC) + [0]*(V-int(SC)-1)
        self.PowerCap = [power_max * 3] + [power_max] * (V-1)
        self.PowerFrac = {}
        self.Power_LP_A = 0.0 * np.identity(self.graphsize**2)
        self.Power_LP_b = 0.0 * np.arange(self.graphsize**2)
        self.A_nonnegative = np.identity(self.graphsize**2)
        self.b_nonnegative = 0.0 * np.arange(self.graphsize **2)
        self.A_powercap = 0.0 * np.arange(self.graphsize **3).reshape(self.graphsize , self.graphsize**2)
        self.b_powercap = -1.0 * np.ones(self.graphsize)
        self.WirelessGain = {}
        self.w_gradient = {}
        self.w_tilde = [] # vector/list
        self.Global_Power_Consume = 0.0 # the averaged global power consumption till the end of last time slot
        self.Current_Power_Consume = 0.0 # used to compute power consumption during last time slot
        self.noise = noise
        
        # Identify the edges that used in a path
        for e in self.edges():
            v = e[0]
            u = e[1]
            self.edge[v][u]['is_in_path'] = 0

        for d in self.demands:
            path_d = d.path
            #print(path_d)
            for k in range(len(path_d)-1):
                v = path_d[k]
                u = path_d[k+1]
                self.edge[v][u]['is_in_path'] = 1
                self.edge[u][v]['is_in_path'] = 1
        
        for e in self.edges():
            v = e[0]
            u = e[1]
            if self.edge[v][u]['is_in_path'] == 1:
                self.edge[v][u]['sinrconst'] = sinr
            else:
                self.edge[v][u]['sinrconst'] = 0
        
        # Count outgoing in-path edges
        for v in self.nodes():
            self.node[v]['out_paths'] = 0
        for e in self.edges():
            v = e[0]
            u = e[1]
            if self.edge[v][u]['is_in_path'] == 1:
                self.node[v]['out_paths'] += 1
        
        # WirelessGain is derived from edge ['gain']
        for v in self.nodes():
            self.WirelessGain[v] = {}
            self.PowerFrac[v] = {}
            for u in self.nodes():
                self.WirelessGain[v][u] = 0.0
                if (v,u) in self.edges() and self.edge[v][u]['is_in_path'] == 1:
                    self.PowerFrac[v][u] = self.PowerCap[v] / self.node[v]['out_paths']
                else:
                    self.PowerFrac[v][u] = 0
        
        # Initially, evenly distributed power to out-paths
        for v in self.nodes():
            self.PowerFrac[v] = {}
            for u in self.nodes():
                if (v,u) in self.edges() and self.edge[v][u]['is_in_path'] == 1:
                    self.PowerFrac[v][u] = 1.0 / self.node[v]['out_paths']
                else:
                    self.PowerFrac[v][u] = 0.0
        for e in self.edges():
            v = e[0]
            u = e[1]
            self.edge[v][u]['power'] = self.PowerCap[v] * self.PowerFrac[v][u]
            self.edge[v][u]['gain'] = gains[e]
            self.WirelessGain[v][u] = gains[e]
            """ if self.edge[v][u]['is_in_path'] == 1:
                self.edge[v][u]['gain'] = 1.0
                self.WirelessGain[v][u] = 1.0
            else:
                self.edge[v][u]['gain'] = 0
                self.WirelessGain[v][u] = 0 """

        # Push power to each node
        self.push_power()
        for i, e in enumerate(self.edges()):
            v = e[0]
            u = e[1]
            self.Power_LP_b[i] = -1.0 * self.edge[v][u]['sinrconst'] * self.noise
            self.Power_LP_A[i][v * self.graphsize + u] += -1.0 * self.WirelessGain[v][u] * self.PowerCap[v]
            for up in self.nodes():
                if up != u:
                    self.Power_LP_A[i][v * self.graphsize + up] += 1.0 * self.edge[v][u]['sinrconst'] * self.WirelessGain[v][u] * self.PowerCap[v]
            for vp in self.nodes():
                if vp != v:
                    for up in self.nodes():
                        self.Power_LP_A[i][vp * self.graphsize + up] += 1.0* self.edge[v][u]['sinrconst'] * self.WirelessGain[vp][u] * self.PowerCap[vp]
        
        for v in self.nodes():
            for u in self.nodes():
                self.A_powercap[v][v * self.graphsize + u] = -1.0

        self.A_nonnegative = -1.0* np.identity(self.graphsize **2)
        self.b_nonnegative = np.zeros(self.graphsize ** 2)
        self.A_powercap = np.kron(np.eye(self.graphsize) , np.ones((1,self.graphsize)))
        self.b_powercap = np.ones(self.graphsize)
        
        #self.b_powercap = -1.0 * np.ones(self.graphsize)
        print("Power initialized at time "+str(self.env.now))
        
        """ for i, e in enumerate(self.edges()):
            v = e[0]
            u = e[1]
            #print((v,u))
            self.edge[v][u]['power'] = power_max / len(self.edge[v])
            self.edge[v][u]['sinrconst'] = random.uniform(sinr_min, sinr_max)
            self.edge[v][u]['gain'] = gains[e]
            #self.edge[v][u]['linkcost'] = 1/gains[e]
        for i, e in enumerate(self.edges()):
            v = e[0]
            u = e[1]
            self.b[i] = self.edge[v][u]['sinrconst'] * self.noise
            for j, ep in enumerate(self.edges()):
                vp = ep[0]
                up = ep[1]
                if vp <= SC:
                    if i == j:
                        self.A[i][j] = self.edge[vp][up]['gain']
                    elif vp != u:
                        self.A[i][j] = -1 * self.edge[v][u]['sinrconst'] * self.edge[vp][u]['gain'] """
    
        # start power update process
        if cache_type == 'LMIN':
            self.env.process(self.power_update_process_LMIN())
        else:
            #self.env.process(self.power_update_process_Priority())
            self.env.process(self.power_update_process_Priority_MinC())
        
    def push_power(self):
        for v in self.nodes():
            power_cap = self.PowerCap[v]
            power_frac_vect = self.PowerFrac[v]
            self.node[v]['cache'].cache._w = power_frac_vect
            self.node[v]['cache'].cache._powercap = power_cap
            for u in self.nodes():
                if (v,u) in self.edges():
                    self.edge[v][u]['power'] = self.PowerFrac[v][u] * self.PowerCap[v]
        print("Power pushed to caches at time "+str(self.env.now))
        
    def power_update_process_LMIN(self,alpha = 5.e-2):
        # Process that pull the subgradients of power from caches, 
        # project and update global power every T 
        while True:
            yield self.env.timeout(self.T)
            
            #print("Power global-updating at time "+str(self.env.now))
            # step0: reset parameters
            self.Current_Power_Consume = 0.0
            for v in self.nodes():
                self.Current_Power_Consume += self.node[v]['cache'].stats['history_power_consume']
                self.w_gradient[v] = {}
                for u in self.nodes():
                    self.w_gradient[v][u] = 0.0
            self.w_tilde = [0.0 for i in range(self.graphsize **2)]
            print("Time-averaged power consumption in last time slot ="+str((self.Current_Power_Consume - self.Global_Power_Consume)/self.T))
            self.wirelessStats['power_by_slot'].append((self.Current_Power_Consume - self.Global_Power_Consume)/self.T)
            self.Global_Power_Consume = self.Current_Power_Consume
            
            # step1: pull n-scores from each node and compute estimated subgradient from 
            for v in self.nodes():
                for u in self.nodes():
                    if u in self.node[v]['cache'].cache._Score_n.keys():
                        self.w_gradient[v][u] = -1.0* self.PowerCap[v] * self.node[v]['cache'].cache._Score_n[u]  / self.T
                    else:
                        #print("not in key")
                        self.w_gradient[v][u] = 0.0
            
            # step2: calculate pre-projection w vector
            for v in self.nodes():
                for u in self.nodes():
                    self.w_tilde[v*self.graphsize+u] = self.PowerFrac[v][u] + alpha* self.w_gradient[v][u]
            #print("w_tilde = "+str(self.w_tilde))    
            
            # step3: project w_tilde vector into constraint set Aw<=b, i.e. min w'Pw + Q'w, s.t. Gw <= h
            Q_qp = matrix(-2.0 * np.transpose(np.array(self.w_tilde)))
            
            P_qp = matrix(np.identity(self.graphsize ** 2))
            
            
            #G_qp = matrix( -1.0 * np.vstack((self.Power_LP_A, self.A_nonnegative, self.A_powercap)))
            #G_qp = matrix( -1.0 * np.vstack((self.Power_LP_A, self.A_nonnegative, self.A_powercap, -1.0*self.A_powercap)))
            #G_qp = matrix( -1.0 * np.vstack((self.A_nonnegative, self.A_powercap)))
            G_qp = matrix( np.vstack((self.Power_LP_A, self.A_nonnegative, self.A_powercap)))  
           
            #h_qp = matrix( -1.0 * np.transpose(np.hstack((self.Power_LP_b, self.b_nonnegative, self.b_powercap))))
            #h_qp = matrix( -1.0 * np.transpose(np.hstack((self.Power_LP_b, self.b_nonnegative, self.b_powercap, -1.0*self.b_powercap))))
            #h_qp = matrix( -1.0 * np.transpose(np.hstack((self.b_nonnegative, self.b_powercap))))
            h_qp = matrix( np.transpose(np.hstack((self.Power_LP_b, self.b_nonnegative, self.b_powercap))))            
            
            sol_qp = qp(P_qp, Q_qp, G_qp, h_qp, options={'show_progress': False})
            Iter = sol_qp['iterations']            
            # step3: update global power variable
            w_bar = sol_qp['x']
            #print(Iter)
            #print(np.max(G_qp*w_bar - h_qp))
            #if Iter >= 200:
            if Iter >= 100 or np.max(G_qp*w_bar - h_qp) > 1.e-9:
                print("Fail to solve projection at time "+str(self.env.now))
                yield self.env.timeout(self.T)
                continue

            epsilon = 1.e-3
            for v in self.nodes():
                for u in self.nodes():
                    if np.abs(w_bar[v * self.graphsize + u]) > epsilon:
                        self.PowerFrac[v][u] = w_bar[v * self.graphsize + u]
                    else:
                        self.PowerFrac[v][u] = 0.0
            #print("self.PowerFrac = "+str(self.PowerFrac))
            self.wirelessStats['frac_by_slot'].append(self.PowerFrac)
            
            # step4: push power to caches and edges
            self.push_power()
            print("...done")
            yield self.env.timeout(self.T)

    def power_update_process_Priority(self):
        # Centralized power updating for other cache types
        while(True):
            # step0: collect power consumption and reset
            self.Current_Power_Consume = 0.0
            for v in self.nodes():
                self.Current_Power_Consume += self.node[v]['cache'].stats['history_power_consume']
                self.w_gradient[v] = {}
                for u in self.nodes():
                    self.w_gradient[v][u] = 0.0
            self.w_tilde = [0.0 for i in range(self.graphsize **2)]
            print("Time-averaged power consumption in last time slot ="+str((self.Current_Power_Consume - self.Global_Power_Consume)/self.T))
            self.wirelessStats['power_by_slot'].append((self.Current_Power_Consume - self.Global_Power_Consume)/self.T)
            self.Global_Power_Consume = self.Current_Power_Consume
            
            # step1: collects the caching states and construct LP elements:
            # max sum_f lamb_f sum_k=1^(p-1) t_{i,p,k}
            # s.t. t_{i.p.k} <= 1; t_{i.p.k} <= 1 - w_{p_k+1 p_k} + sum_l=1^k x_{p_l i}; SINR; Powercap
            size_t = 0
            for d in self.demands:
                size_t += len(d.path)-1
            t_const_1 = np.ones(size_t) # row vector for t <= 1
            t_const_2 = np.zeros(size_t) # row vector for t <= 1-w+sum...
            t_pos = 0
            c_lp = np.zeros(size_t + self.graphsize**2)
            for d in self.demands:
                for k in [1+i for i in range(len(d.path)-1)]:
                    c_lp[t_pos] = -1.0 * d.rate
                    sum_cache = 0.0
                    for l in range(k):
                        sum_cache += d.item in self.node[d.path[l]]['cache'].cache
                    t_const_2[t_pos] = 1 - self.PowerFrac[d.path[k]][d.path[k-1]]
                    t_pos += 1
            #print("size_t = "+str(size_t)+", t_pos = "+str(t_pos))
            
            # step2: solve LP: min cTx , s.t. Gx <= h
            c_lp = matrix(np.transpose(c_lp))
            G1 = np.hstack((np.identity(size_t), np.zeros((size_t,self.graphsize**2))))
            G2 = G1
            G_W = -1.0 * np.vstack((self.Power_LP_A, self.A_nonnegative, self.A_powercap))
            rows,cols = np.shape(G_W)
            G3 = np.hstack(( np.zeros(( rows, size_t)),  G_W ))
            G_lp = matrix(np.vstack((G1,G2,G3)))
            
            h_W = -1.0 * np.hstack((self.Power_LP_b, self.b_nonnegative, self.b_powercap))
            h_lp = matrix(np.transpose(np.hstack((t_const_1,t_const_2,h_W))))
            #print("c_lp = "+str(c_lp))
            #print("G_lp = "+str(G_lp))
            #print("h_lp = "+str(h_lp))
            
            sol_lp = lp(c = c_lp, G = G_lp, h = h_lp, options={'show_progress': False})
            
            # step3: update power variables
            for v in self.nodes():
                for u in self.nodes():
                    self.PowerFrac[v][u] = sol_lp['x'][size_t + v*self.graphsize +u]

            self.wirelessStats['frac_by_slot'].append(self.PowerFrac)            
            self.push_power()
            yield self.env.timeout(self.T)

    def power_update_process_Priority_MinC(self):
        # Centralized power updating for other cache types
        while(True):
            # step0: collect power consumption and reset
            self.Current_Power_Consume = 0.0
            for v in self.nodes():
                self.Current_Power_Consume += self.node[v]['cache'].stats['history_power_consume']
                self.w_gradient[v] = {}
                for u in self.nodes():
                    self.w_gradient[v][u] = 0.0
            self.w_tilde = [0.0 for i in range(self.graphsize **2)]
            print("Time-averaged power consumption in last time slot ="+str((self.Current_Power_Consume - self.Global_Power_Consume)/self.T))
            self.Global_Power_Consume = self.Current_Power_Consume
            
            # step1: collects the caching states and construct LP elements:
            # min sum_(i,p) lambda_(i,p) sum_k s_bar_p_k+1 w_{p_k+1 p_k} prod_l (1 - x_{p_l i})
            # s.t. w >= 0, sum w <= 1, SINR
            w_len = self.graphsize **2
            c_minC = np.zeros(w_len)
            for d in self.demands.keys():
                rate = d.rate
                path = d.path
                item = d.item
                for k in range(1,len(path)-1):
                    cache_product = 1.0
                    for l in range(1,k):
                        cache_product = cache_product * (1.0 - item in self.node[path[l-1]]['cache'].cache)
                    v = path[k]
                    u = path[k-1]
                    c_minC[ v * self.graphsize + u] += rate * self.PowerCap[v] * cache_product
            
            # step2: solve LP: min cTx , s.t. Gx <= h
            G_lp = matrix( np.vstack((self.Power_LP_A, self.A_nonnegative, self.A_powercap)))
            h_lp = matrix( np.transpose(np.hstack((self.Power_LP_b, self.b_nonnegative, self.b_powercap))))
            
            c_lp = matrix(np.transpose(c_minC))
            #print("c_lp = "+str(c_lp))
            #print("G_lp = "+str(G_lp))
            #print("h_lp = "+str(h_lp))
            
            sol_lp = lp(c = c_lp, G = G_lp, h = h_lp, options={'show_progress': False})
            
            # step3: update power variables
            for v in self.nodes():
                for u in self.nodes():
                    self.PowerFrac[v][u] = sol_lp['x'][v*self.graphsize +u]
            
            self.push_power()
            yield self.env.timeout(self.T)

    def run(self, finish_time):

        logging.info('Simulating..')
        self.env.run(until=finish_time)
        logging.info('..done simulating')

    def spawn_queries_process(self, d):
        """ A process that spawns queries.

            Queries are generated according to a Poisson process with the appropriate rate. Queries generated are pushed to the query source node.
        """
        while True:
            logging.debug(
                pp([self.env.now, ':New query for', d.item, 'to follow', d.path]))
            _id = self.demands[d]['queries_spawned']
            # create a new query message at the query_source
            qm = QueryMessage(d, _id)
            self.demands[d]['pending'].add(_id)
            self.demands[d]['queries_spawned'] += 1
            yield self.node[d.query_source]['pipe'].put((qm, (d, d.query_source)))
            yield self.env.timeout(random.expovariate(d.rate))

    def demand_monitor_process(self, d):
        """ A process monitoring statistics about completed requests.
        """
        while True:
            msg = yield self.demands[d]['pipe'].get()

            lab, dem, query_id = msg.header
            stats = msg.stats
            now = self.env.now

            if lab is not "RESPONSE":
                logging.warning(
                    pp([now, ':', d, 'received a non-response message:', msg]))
                continue

            if dem is not d:
                logging.warning(
                    pp([now, ':', d, 'received a message', msg, 'aimed for demand', dem]))
                continue

            if query_id not in self.demands[d]['pending']:
                logging.warning(
                    pp(['Query', query_id, 'of', d, 'satisfied but not pending']))
                continue
            else:
                self.demands[d]['pending'].remove(query_id)

            logging.debug(
                pp(['Query', query_id, 'of', d, 'satisfied with stats', stats]))
            self.demands[d]['queries_satisfied'] += 1

            if now >= self.warmup:
                self.demands[d]['queries_logged'] += 1.0

                for key in stats:
                    if key in self.demands[d]['stats']:
                        self.demands[d]['stats'][key] += stats[key]
                    else:
                        self.demands[d]['stats'][key] = stats[key]

    def demand_change_process(self, d):
        """ A process changing the demand periodically 
        """
        while True:
            yield self.env.timeout(1./self.demand_change_rate)
            new_rate = random.uniform(self.demand_min, self.demand_max)
            logging.info(pp([self.env.now, ':Demand for ', d.item, 'following',
                         d.path, 'changing rate from', d.rate, 'to', new_rate]))
            d.rate = new_rate

    def compute_opt_process(self):
        ''' Process recomputing optimal values after demand changes. Used only if demand changes periodically
        '''
        yield self.env.timeout(0.01*1./self.demand_change_rate)  # Tiny offset, to make sure all demands have been updated by measurement time

        while True:
            Y, res = self.minimizeRelaxation()
            logging.info(
                pp([self.env.now, ': Optimal Relaxation is: ', self.relaxation(Y)]))
            logging.info(
                pp([self.env.now, ': Expected caching gain at relaxation point is: ', self.expected_caching_gain(Y)]))
            optimal_stats = {}
            optimal_stats['res'] = res
            optimal_stats['Y'] = Y
            optimal_stats['L'] = self.relaxation(Y)
            optimal_stats['F'] = self.expected_caching_gain(Y)
            self.optstats[self.env.now] = optimal_stats
            yield self.env.timeout(1./self.demand_change_rate)

    def message_pusher_process(self, e):
        """ A process handling message transmissions over edges.

            It delays messages, accoding to delay specs, and increments stat counters in them.
        """

        while True:
            msg = yield self.edge[e[0]][e[1]]['pipe'].get()
            logging.debug(
                pp([self.env.now, ':', 'Pipe at', e, 'pushing', msg]))
            time_before = self.env.now
            yield self.env.timeout(msg.length*random.expovariate(1/self.edge[e[0]][e[1]]['delay']))
            delay = self.env.now - time_before
            msg.stats['delay'] += delay
            msg.stats['hops'] += 1.0
            msg.stats['weight'] += self.edge[e[0]][e[1]]['weight']
            if not msg.u_bit:
                if self.isWireless:
                    msg.stats['downweight'] += self.edge[e[0]][e[1]]['power']
                else:
                    msg.stats['downweight'] += self.edge[e[0]][e[1]]['weight'] + self.edge[e[1]][e[0]]['weight']
            logging.debug(
                pp([self.env.now, ':', 'Pipe at', e, 'delivering', msg]))
            self.node[e[1]]['pipe'].put((msg, e))

    def cache_process(self, x):
        """A process handling messages sent to caches.

           It is effectively a wrapper for a receive call, made to a NetworkedCache object. 

        """
        while True:
            (msg, e) = yield self.node[x]['pipe'].get()
            # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!!
            generated_messages = self.node[e[1]]['cache'].receive(
                msg, e, self.env.now)
            for (new_msg, new_e) in generated_messages:
                if new_e[1] in self.demands:
                    yield self.demands[new_e[1]]['pipe'].put(new_msg)
                else:
                    yield self.edge[new_e[0]][new_e[1]]['pipe'].put(new_msg)

    def cachesToMatrix(self):
        """Constructs a matrix containing cache information.
        """
        zipped = []
        n = len(self.nodes())
        m = max(len(self.item_set), max(self.item_set)+1)
        for x in self.nodes():
            for item in self.node[x]['cache']:
                zipped.append((1, x, item))

        val, I, J = zip(*zipped)
        J = list(map(int,J))
        return spmatrix(val, I, J, size=(n, m))

    def statesToMatrix(self):
        """Constructs a matrix containing marginal information. This assumes that caches contain a state() function, capturing maginals. Only LMin implements this
        """
        zipped = []
        n = len(self.nodes())
        m = max([d.item for d in self.demands])+1
        Y = matrix()
        for x in self.nodes():
            for item in self.node[x]['cache'].non_zero_state_items():
                zipped.append((self.node[x]['cache'].state(item), x, item))

        val, I, J = zip(*zipped)
        J = list(map(int,J))
        return spmatrix(val, I, J, size=(n, m))

    def social_welfare(self):
        """ Function computing the social welfare.
        """
        dsw = 0.0
        hsw = 0.0
        wsw = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate
            x = d.query_source
            # THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!
            while not (item in self.node[x]['cache'] or x is d.item_source):
                s = d.succ(x)
                dsw += rate*(CONFIG.QUERY_MESSAGE_LENGTH *
                             self.edge[x][s]['delay'] + CONFIG.RESPONSE_MESSAGE_LENGTH*self.edge[s][x]['delay'])
                wsw += rate*(self.edge[x][s]['weight'] +
                             self.edge[s][x]['weight'])
                hsw += rate*(2)
                x = s

        return (dsw/sumrate, hsw/sumrate, wsw/sumrate)

    def caching_gain(self):
        """ Function computing the caching gain under the present caching situation
        """
        X = self.cachesToMatrix()
        return self.expected_caching_gain(X)

    def cost_without_caching(self):
        """ Function computing the  cost of recovering all items demanded from respective sources."""
        cost = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            while s is not None:
                cost += rate*(self.edge[s][x]['weight'] +
                              self.edge[x][s]['weight'])
                x = s
                s = d.succ(x)

        return cost/sumrate

    def expected_caching_gain(self, Y):
        """ Function computing the expected caching gain under marginals Y, presuming product form. Also computes deterministic caching gain if Y is integral.
        """
        ecg = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            prodsofar = 1-Y[x,int(item)]
            while s is not None:
                ecg += rate*(self.edge[s][x]['weight'] +
                             self.edge[x][s]['weight'])*(1-prodsofar)

                x = s
                s = d.succ(x)
                prodsofar *= 1-Y[x,int(item)]

        return ecg/sumrate

    def relaxation(self, Y):
        """ Function computing the relaxation of caching gain under marginals Y. Relaxation equals deterministic caching gain if Y is integral.
        """
        rel = 0.0
        sumrate = 0.0
        for d in self.demands:
            item = d.item
            rate = d.rate
            sumrate += rate

            x = d.query_source
            s = d.succ(x)
            sumsofar = Y[x,int(item)]
            while s is not None:
                rel += rate*(self.edge[s][x]['weight'] +
                             self.edge[x][s]['weight'])*min(1.0, sumsofar)

                x = s
                s = d.succ(x)
                sumsofar += Y[x,int(item)]

        return rel/sumrate

    def demand_stats(self):
        """ Computed stats across demands.
        """
        stats = {}
        queries_logged = 0.0
        rate = 0.0
        for d in self.demands:
            queries_logged += self.demands[d]['queries_logged']
            for key in self.demands[d]['stats']:
                if key in stats:
                    stats[key] += self.demands[d]['stats'][key]
                else:
                    stats[key] = self.demands[d]['stats'][key]
        for key in stats:
            stats[key] = stats[key]/queries_logged

        return stats

    def monitor_process(self):
        while True:
            now = self.env.now
            if now >= self.warmup:
                self.sw[now] = self.social_welfare()
                self.demandstats[now] = self.demand_stats()
                X = self.cachesToMatrix()
                ecg = self.expected_caching_gain(X)
                rel = self.relaxation(X)
                tot = self.cost_without_caching()
                esw = tot-ecg
                self.funstats[now] = (ecg, rel, esw, tot)
                logging.info(pp([now, ':', 'DSW = %f, HSW = %f, WSW = %f' % self.sw[now],
                             ', DEMSTATS =', self.demandstats[now], 'FUNSTATS =', self.funstats[now]]))
                try:
                    # if True:
                    Y = self.statesToMatrix()
                    secg = self.expected_caching_gain(Y)
                    srel = self.relaxation(Y)
                    self.funstats[now] += (secg, srel)
                    logging.info(pp([now, ': SECG=', secg, 'SREL=', srel]))
                except AttributeError:
                    logging.debug(pp([now, ": No states in this class"]))
                # M=self.cachesToMatrix()
                #formatted = [  (i,j,'*')  if self.node[i]['cache'].isPermanent(j) else (i,j)  for i,j,v in  zip(M.I,M.J,M.V)   ]
                # logging.info(str(sorted(formatted)))
            yield self.env.timeout(random.expovariate(self.monitoring_rate))

    def minimizeRelaxation(self):
        n = len(self.nodes())
        m = max([d.item for d in self.demands])+1
        number_of_placement_variables = n*m

        def position(node, item):
            return node*m+item

        A = []
        b = []
        row = 0

        # Permanent set constraints
        logging.debug('Creating permanent set constaints...')
        row = 0
        for x in self.nodes():
            perm_set = self.node[x]['cache'].perm_set()
            if len(perm_set) > 0:
                for item in perm_set:
                    A += [(1.0, row, position(x, item))]
                    b += [1.0]
                    row += 1
        logging.debug('...done. Created %d constraints' % row)

        total_equality_constraints = row

        G = []
        h = []

        # Capacity constraints
        logging.debug('Creating capacity constaints...')
        G += [(1.0, x, position(x, item))
              for item in range(m) for x in self.nodes()]
        h += [self.node[x]['cache'].capacity() + len(self.node[x]['cache'].perm_set())
              for x in self.nodes()]
        #h += [ self.node[x]['cache'].capacity()    for x in self.nodes()]
        logging.debug('...done at %d rows' % len(h))

        row = n
        t = number_of_placement_variables

        # t's smaller than sums of y_vi's in path
        logging.debug('Creating t up constraints...')
        for d in self.demands:
            item = d.item
            path = d.path
            sofar = []
            for v in path:
                if len(sofar) > 0:
                    for u in sofar:
                        G += [(-1.0, row, position(u, item))]
                    G += [(1.0, row, t)]
                    h += [0.0]
                    row += 1
                    t += 1
                sofar += [v]

        logging.debug('...done at %d rows' % row)

        total_ts = t - number_of_placement_variables

        # t's smaller than 1.0
        logging.debug('Creating t ll one constraints...')
        t = number_of_placement_variables
        while t < (number_of_placement_variables+total_ts):
            G += [(1.0, row, t)]
            h += [1.0]
            row += 1
            t += 1

        logging.debug('...done at %d rows' % row)

        # y's less than 1
        logging.debug('Creating y ll one gg 0 constraints...')
        y = 0
        while y < number_of_placement_variables:
            G += [(1.0, row, y)]
            h += [1.0]
            row += 1
            G += [(-1.0, row, y)]
            h += [0.0]
            row += 1
            y += 1
        logging.debug('...done at %d rows' % row)

        total_inequality_constraints = row

        # objective
        logging.debug('Creating objective vector...')
        c = number_of_placement_variables*[0]
        for d in self.demands:
            rate = d.rate
            path = d.path

            x = d.query_source
            s = d.succ(x)
            while s is not None:
                c += [-rate*(self.edge[s][x]['weight'] +
                             self.edge[x][s]['weight'])]
                x = s
                s = d.succ(x)

        logging.debug('...done at %d terms' % len(c))
        val, I, J = zip(*A)
        J = list(map(int,J))
        A = spmatrix(val, I, J, size=(total_equality_constraints,
                     number_of_placement_variables+total_ts))
        b = matrix(b)

        val, I, J = zip(*G)
        J = list(map(int,J))
        G = spmatrix(val, I, J, size=(total_inequality_constraints,
                     number_of_placement_variables + total_ts))
        h = matrix(h)

        c = matrix(c)

        logging.debug('c has length %d ' % len(c))
        logging.debug('G has dims %d x %d and matrix_rank ' %
                      G.size + str(matrix_rank(G)))
        logging.debug('h has length %d ' % len(h))
        logging.debug('A has dims %d x %d and matrix_rank' %
                      A.size + str(matrix_rank(A)))
        logging.debug('b has length %d ' % len(b))

        # , primalstart={'x':matrix(np.zeros(c.size)*1.e-99),'s':matrix( total_inequality_constraints*[1.e-20])})
        res = lp(c, G, h, A, b, options={'show_progress': False})

        opt = res['x'][:number_of_placement_variables]
        return np.reshape(opt, (n, m), order='C'), res
            

class NetworkedCache(object):
    """An abstract networked cache.	
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, capacity, _id):
        pass

    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def perm_set(self):
        pass

    @abstractmethod
    def makePermanent(self, item):
        pass

    @abstractmethod
    def receive(self, message, edge, time):
        pass

    @abstractmethod
    def __contains__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def isPermanent(self, item):
        pass


class PriorityNetworkCache(NetworkedCache):
    """ A Priority Networked Cache. Supports LRU,LFU, and RR policies.

        Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the LRU principle.
    """

    def __init__(self, capacity, _id, principle):
        self.cache = PriorityCache(capacity, _id)
        self.permanent_set = set([])
        self._id = _id
        self._capacity = capacity
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.stats['history_power_consume'] = 0.0 # use to calculate the objective (power consumption)
        self.principle = principle

    def __str__(self):
        return str(self.cache)+'+'+str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def isPermanent(self, item):
        return item in self.permanent_set

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            item = d.item
            logging.debug(
                pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            inside_cache = item in self.cache
            inside_permanent_set = item in self.permanent_set
            if inside_cache or inside_permanent_set:
                logging.debug(pp([now, ': Item', item, 'is inside',
                              'permanent set' if inside_permanent_set else 'cache', 'of', self._id]))
                if inside_cache:  # i.e., not in permanent set
                    princ_map = {'LRU': now, 'LFU': self.cache.priority(
                        item)+1, 'RR': random.random(), 'FIFO': self.cache.priority(item)}
                    logging.debug(pp([now, ': Priority of', item, 'updated to',
                                  princ_map[self.principle], 'at cache', self._id]))
                    self.cache.add(item, princ_map[self.principle])
                self.stats['hits'] += 1
                if self._id == d.query_source:
                    logging.debug(pp([now, ': Response to query', query_id,
                                  'of', d, 'delivered to query source by cache', self._id]))
                    pred = d
                    # for the case that requesting node has item in its cache
                    self.stats['history_power_consume'] += 0.0
                else:
                    pred = d.pred(self._id)
                    logging.debug(
                        pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                        # for the case that  node has item in its cache, send respond msg
                    self.stats['history_power_consume'] += self.cache._powercap * self.cache._w[pred]
                e = (self._id, pred)
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)
                return [(rmsg, e)]
            else:
                logging.debug(
                    pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached',
                                  self._id, 'and has nowhere to go, will be dropped']))
                    return []

                # query continue propagates
                #self.stats['history_power_consume'] += self.cache._powercap * self.cache._w[succ]
                self.stats['history_power_consume'] += 0.0

                e = (self._id, succ)
                return [(msg, e)]

        if label == "RESPONSE":
            logging.debug(
                pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            princ_map = {'LRU': now, 'LFU': self.cache.priority(
                item)+1 if item in self.cache else 1, 'RR': random.random(), 'FIFO': self.cache.priority(item) if item in self.cache else now}
            logging.debug(pp([now, ': Priority of', item, 'updated to',
                          princ_map[self.principle], 'at', self._id]))
            # add the item to the cache/update priority
            self.cache.add(item, princ_map[self.principle])

            if d.query_source == self._id:
                logging.debug(pp([now, ': Response to query', query_id,
                              'of', d, ' finally delivered by cache', self._id]))
                pred = d
                # arrive at the query node
                self.stats['history_power_consume'] += 0.0

            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d,
                              'passes through cache', self._id, 'moving further down path']))
                pred = d.pred(self._id)
                # respond continue propagates
                self.stats['history_power_consume'] += self.cache._powercap * self.cache._w[pred]
            e = (self._id, pred)
            return [(msg, e)]


class ExploreMessage(Message):
    """
         An exploration message.
    """

    def __init__(self, d, query_id, initiator, stats=None):
        Message.__init__(self, header=("EXPLORE", d, query_id), payload=None,
                         length=CONFIG.EXPLORE_MESSAGE_LENGTH, stats=stats, u_bit=True)
        self. explore_source = initiator


class ExploreResponseMessage(Message):
    """
         An exploration response message.
    """

    def __init__(self, d, query_id, initiator, stats=None):
        Message.__init__(self, header=("EXPLORE_RESPONSE", d, query_id), payload=None,
                         length=CONFIG.EXPLORE_MESSAGE_LENGTH, stats=stats, u_bit=False)
        self. explore_source = initiator


class EWMAGradCache(NetworkedCache):
    """ An EWMA Gradient Networked Cache.

        Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

    def __init__(self, capacity, _id, beta=1.0):
        self.cache = EWMACache(capacity, _id, beta)
        self._id = _id
        self.permanent_set = set([])
        self._capacity = capacity
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.stats['explores'] = 0.0
        self.stats['explore_responses'] = 0.0

    def __str__(self):
        return 'Cache: '+str(self.cache)+'Permanent: '+str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def isPermanent(self, item):
        return item in self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            item = d.item
            logging.debug(
                pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            inside = item in self.cache or item in self.permanent_set
            if inside:
                logging.debug(
                    pp([now, ': Item', item, 'is inside ', self._id]))
                self.stats['hits'] += 1
                if self._id == d.query_source:
                    logging.debug(pp([now, ': Response to query', query_id,
                                  'of', d, ' finally delivered by cache', self._id]))
                    pred = d
                else:
                    pred = d.pred(self._id)
                    logging.debug(
                        pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)

                msglist = [(rmsg, e)]

                if item not in self.permanent_set:
                    succ = d.succ(self._id)
                    if succ != None:
                        logging.debug(
                            pp([now, ': Item', item, 'is inside cache of ', self._id, ' will forward exploration message']))
                        emsg = ExploreMessage(d, query_id, self._id)
                        e = (self._id, succ)
                        msglist += [(emsg, e)]

                return msglist
            else:
                logging.debug(
                    pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached',
                                  self._id, 'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                return [(msg, e)]

        if label == "RESPONSE":
            logging.debug(
                pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            logging.debug(pp([now, ': Node', self._id, 'updating derivative of item',
                          item, 'with measurement', msg.stats['downweight']]))
            # does not really add item, but updates its gradient estimate
            self.cache.add(item, msg.stats['downweight'], now)
            logging.debug(
                pp([now, ': Node', self._id, ' now stores', self.cache]))

            if d.query_source == self._id:
                logging.debug(pp([now, ': Response to query', query_id,
                              'of', d, ' finally delivered by cache', self._id]))
                pred = d
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d,
                              'passes through cache', self._id, 'moving further down path']))
                pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]

        if label == "EXPLORE":
            item = d.item
            logging.debug(
                pp([now, ': Explore message for item', item, 'received by cache', self._id]))
            self.stats['explores'] += 1.0

            inside = item in self.cache or item in self.permanent_set
            if inside:
                logging.debug(
                    pp([now, ': Item', item, 'is inside ', self._id]))
                pred = d.pred(self._id)
                logging.debug(pp([now, ': Explore Response to query',
                              query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                ermsg = ExploreResponseMessage(
                    d, query_id, msg.explore_source, stats=msg.stats)

                return [(ermsg, e)]

            else:
                logging.debug(
                    pp([now, ': Item', item, 'is not inside', self._id, 'continue exploring']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.debug(pp([now, ':Exploration for', query_id, 'of', d,
                                  'reached', self._id, 'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                return [(msg, e)]

        if label == "EXPLORE_RESPONSE":
            logging.debug(
                pp([now, ': Explore Response message for', d, 'received by cache', self._id]))
            self.stats['explore_responses'] += 1.0
            item = d.item

            if msg.explore_source == self._id:
                logging.debug(
                    pp([now, ': Node', self._id, 'received final exploration response for', query_id, 'of', d]))
                logging.debug(pp([now, ': Node', self._id, 'updating derivative of item',
                              item, 'with measurement', msg.stats['downweight']]))
                # does not really add item, but updates its gradient estimate
                self.cache.add(item, msg.stats['downweight'], now)
                logging.debug(
                    pp([now, ': Node', self._id, ' now stores', self.cache]))
                return []

            logging.debug(pp([now, ': Explore Response to query', query_id, 'of',
                          d, 'passes through cache', self._id, 'moving further down path']))
            pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]


class LMinCache(NetworkedCache):
    """ A Networked Cache minimizing the relaxation L.

        Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

    def __init__(self, capacity, _id, gamma=0.1, T=5, expon=0.5, interpolate=False):
        self.cache = LMinimalCache(capacity, _id, gamma, expon, interpolate)
        self._id = _id
        self.permanent_set = set([])
        self._capacity = capacity
        self.T = T
        self.stats = {}
        self.stats['queries'] = 0.0
        self.stats['hits'] = 0.0
        self.stats['responses'] = 0.0
        self.stats['explores'] = 0.0
        self.stats['explore_responses'] = 0.0
        self.stats['history_power_consume'] = 0.0 # use to calculate the objective (power consumption)

    def __str__(self):
        return 'Cache: '+str(self.cache)+'Permanent: '+str(self.permanent_set)

    def __contains__(self, item):
        return item in self.cache or item in self.permanent_set

    def __iter__(self):
        return itertools.chain(self.cache, self.permanent_set)

    def capacity(self):
        return self._capacity

    def perm_set(self):
        return self.permanent_set

    def isPermanent(self, item):
        return item in self.permanent_set

    def makePermanent(self, item):
        self.permanent_set.add(item)

    def receive(self, msg, e, now):
        label, d, query_id = msg.header

        if label == "QUERY":
            msglist = []
            item = d.item

            logging.debug(
                pp([now, ': Query message for item', item, 'received by cache', self._id]))
            self.stats['queries'] += 1.0

            if self._id == d.query_source:
                succ = d.succ(self._id)
                if succ != None:
                    logging.debug(pp([now, ': Item', item, 'is inside cache of ', self._id,
                                  ' which is the query source, will prepare an exploration message']))
                    emsg = ExploreMessage(d, query_id, self._id)
                    e = (self._id, succ)
                    sumsofar = self.cache.state(
                        item) + float(item in self.permanent_set)
                    emsg.payload = sumsofar
                    if sumsofar < 1.0:
                        msglist.append((emsg, e))

            inside_cache = item in self.cache
            inside_perm = item in self.permanent_set
            if inside_cache or inside_perm:
                logging.debug(pp([now, ': Item', item, 'is inside %s of %d' % (
                    'cache' if inside_cache else 'permanent set', self._id)]))
                self.stats['hits'] += 1
                if self._id == d.query_source:
                    logging.debug(pp([now, ': Response to query', query_id,
                                  'of', d, ' finally delivered by cache', self._id]))
                    pred = d
                    # for the case that requesting node has item in its cache
                    self.stats['history_power_consume'] += 0.0
                else:
                    pred = d.pred(self._id)
                    logging.debug(
                        pp([now, ': Response to query', query_id, 'of', d, ' generated by cache', self._id]))
                    # for the case that a hit cache for non-requesting node, consumes power
                    self.stats['history_power_consume'] += self.cache._powercap * self.cache._w[pred]
                    if self._id == 1:
                        #print("self.stats['history_power_consume'] +="+str(self.cache._powercap * self.cache._w[pred]))
                        pass

                e = (self._id, pred)
                rmsg = ResponseMessage(d, query_id, stats=msg.stats)

                msglist.append((rmsg, e))

            else:
                logging.debug(
                    pp([now, ': Item', item, 'is not inside', self._id, 'continue searching']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Query', query_id, 'of', d, 'reached',
                                  self._id, 'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                msglist.append((msg, e))

            return msglist

        if label == "RESPONSE":
            logging.debug(
                pp([now, ': Response message for', d, 'received by cache', self._id]))
            self.stats['responses'] += 1.0
            item = d.item
            #logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
            # self.cache.add(item,msg.stats['downweight'],now) #does not really add item, but updates its gradient estimate
            #logging.debug(pp([now,': Node',self._id,' now stores',self]))

            if d.query_source == self._id:
                logging.debug(pp([now, ': Response to query', query_id,
                              'of', d, ' finally delivered by cache', self._id]))
                pred = d

                # for case that a content is delivered, no power consumed
                self.stats['history_power_consume'] += 0.0
            else:
                logging.debug(pp([now, ': Response to query', query_id, 'of', d,
                              'passes through cache', self._id, 'moving further down path']))
                pred = d.pred(self._id)
                # for case that a content need further delivery, power consumed
                self.stats['history_power_consume'] += self.cache._powercap * self.cache._w[pred]
                if self._id == 1:
                    #print("self.stats['history_power_consume'] +="+str(self.cache._powercap * self.cache._w[pred]))
                    pass

            e = (self._id, pred)
            return [(msg, e)]

        if label == "EXPLORE":
            item = d.item
            logging.debug(pp([now, ': Explore message for item', item, 'received by cache', self._id, 'sumsofar is %f local state is %f in permanent is %f' % (
                msg.payload, self.cache.state(item), float(item in self.permanent_set))]))
            self.stats['explores'] += 1.0

            # step1: When p_k receives AGG, calculate 1 - w_{p_k p_k-1} + AGG, if <=1 set B = 1, else B = 0
            n_prev = e[0]
            n_curt = e[1]
            Temp = msg.payload + 1 - self.cache._w[n_prev]
            if Temp <= 1:
                B_temp = 1
            else:
                B_temp = 0
            self.cache.setB(d,B_temp) # msg.header[1] == d
			# step1 done

			# step2: add the value of B to n_{p_k p_k-1}
            self.cache.AddOnScore_n(n_prev,B_temp)
			# step2 done

			# step3:
            new_sum = msg.payload + self.cache.state(item) + float(item in self.permanent_set)
            if new_sum > 1.0 or item in self.permanent_set:
                logging.debug(pp([now, ': Item', item, 'has aggregate state value',
                              new_sum, 'on node', self._id, 'of demand', d]))
                pred = d.pred(self._id)
                logging.debug(pp([now, ': Explore Response to query',
                              query_id, 'of', d, ' generated by cache', self._id]))
                e = (self._id, pred)
                ermsg = ExploreResponseMessage(
                    d, query_id, msg.explore_source, stats=msg.stats)
                
                ermsg.payload = 0.0
                return [(ermsg, e)]
            else:
                logging.debug(pp([now, ': Item', item, 'is has agreggate',
                              new_sum, 'at node', self._id, 'continue exploring']))
                succ = d.succ(self._id)
                if succ == None:
                    logging.error(pp([now, ':Exploration for', query_id, 'of', d,
                                  'reached', self._id, 'and has nowhere to go, will be dropped']))
                    return []
                e = (self._id, succ)
                msg.payload = new_sum
                return [(msg, e)]

        if label == "EXPLORE_RESPONSE":
            logging.debug(
                pp([now, ': Explore Response message for', d, 'received by cache', self._id]))
            self.stats['explore_responses'] += 1.0
            item = d.item
            # setp1: for wireless case, use payload = sum of powercaps as the update of m
            logging.debug(
                pp([now, ': Node', self._id, 'received exploration response for', query_id, 'of', d]))
            # case of wireline, aggregate downweight: 	
            #logging.debug(pp([now, ': Node', self._id, 'updating derivative of item',
            #        item, 'with measurement', msg.stats['downweight']]))
            #self.cache.updateGradient(item, msg.stats['downweight'])

			# case of wireless, aggregate power caps:
            logging.debug(pp([now, ': Node', self._id, 'updating derivative of item',
                          item, 'with measurement', msg.payload]))
            self.cache.updateGradient(item, msg.payload)
            # step1 done

			# step2: if B = 1, add payload, otherwise dont
            if self.cache.getB(d) == 1:
                #n_curt = e[1]
                msg.payload += self.cache._powercap #PowerCap[n_curt]
            # step2 done

            if msg.explore_source == self._id:
                logging.debug(
                    pp([now, ': Node', self._id, 'is terminal node for exploration response for', query_id, 'of', d]))
                #logging.debug(pp([now,': Node',self._id,' now stores',self.cache]))
                return []

            logging.debug(pp([now, ': Explore Response to query', query_id, 'of',
                          d, 'passes through cache', self._id, 'moving further down path']))
            pred = d.pred(self._id)
            e = (self._id, pred)
            return [(msg, e)]

    def shuffleProcess(self):
        """A process handling messages sent to caches.

           It is effectively a wrapper for a receive call, made to a NetworkedCache object. 

        """
        while True:
            logging.debug(
                pp([self.env.now, ': New shuffling of cache', self._id]))
            # key,placements,probs,distr=self.cache.shuffle(self.env.now)
            #print("Cache "+str(self._id)+" shuffling at time "+str(self.env.now))
            self.cache.shuffle(self.env.now)
            logging.debug(
                pp([self.env.now, ': New cache at', self._id, ' is', self.cache]))
            #logging.debug('Setting cache at %d to %s, probability was:%f' %(self._id,str(self.cache),probs[key]) )
            yield self.env.timeout(self.T)

            # update power variables of cache
            #print("Current time = "+str(self.env.now))
            #self.push_power()

    def startShuffleProcess(self, env):
        self.env = env
        self.env.process(self.shuffleProcess())

    def state(self, item):
        return max(self.cache.state(item), float(item in self.permanent_set))

    def non_zero_state_items(self):
        return list(self.cache._state.keys())+list(self.permanent_set)


def main():
    #logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Simulate a Network of Caches', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('inputfile',help = 'Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.')
    parser.add_argument('outputfile', help='Output file')
    parser.add_argument('--max_capacity', default=2,
                        type=int, help='Maximum capacity per cache')
    parser.add_argument('--min_capacity', default=2,
                        type=int, help='Minimum capacity per cache')
    parser.add_argument('--max_weight', default=100.0,
                        type=float, help='Maximum edge weight')
    parser.add_argument('--min_weight', default=1.0,
                        type=float, help='Minimum edge weight')
    parser.add_argument('--max_rate', default=1.0,
                        type=float, help='Maximum demand rate')
    parser.add_argument('--min_rate', default=1.0,
                        type=float, help='Minimum demand rate')
    parser.add_argument('--time', default=1000.0, type=float,
                        help='Total simulation duration')
    parser.add_argument('--warmup', default=0.0, type=float,
                        help='Warmup time until measurements start')
    parser.add_argument('--catalog_size', default=100,
                        type=int, help='Catalog size')
#   parser.add_argument('--sources_per_item',default=1,type=int, help='Number of designated sources per catalog item')
    parser.add_argument('--demand_size', default=1000,
                        type=int, help='Demand size')
    parser.add_argument('--demand_change_rate', default=0.0,
                        type=float, help='Demand change rate')
    parser.add_argument('--demand_distribution', default="powerlaw", type=str,
                        help='Demand distribution', choices=['powerlaw', 'uniform'])
    parser.add_argument('--powerlaw_exp', default=1.2, type=float,
                        help='Power law exponent, used in demand distribution')
    parser.add_argument('--query_nodes', default=100, type=int,
                        help='Number of nodes generating queries')
    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type', choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                        'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz', 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom', 'servicenetwork', 'hetnet'])
    parser.add_argument('--graph_size', default=100,
                        type=int, help='Network size')
    
    parser.add_argument('--hetnet_params', nargs='+', default=[5.0,3.0,3.0],
                        type=float, help='Hetnet parameters as tuple (SC,R_cell,pathloss_exp). Default ex: 3 3 3')
    parser.add_argument('--wireless_consts', nargs='+', default=[10.0,0.1],
                        type=float, help='Constraints for wireless case as tuple (max_power,sinr). Ex: 10.0 0.1')
    parser.add_argument('--hetnet_load', type=str, help='Path to hetnet topology file')
    parser.add_argument('--hetnet_grid', help='Whether the hetnet topology should be a grid', default=False, action='store_true')
    
    parser.add_argument('--graph_degree', default=4, type=int,
                        help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
    parser.add_argument('--graph_p', default=0.10, type=int,
                        help='Probability, used in erdos_renyi, watts_strogatz')
    parser.add_argument('--random_seed', default=123456789,
                        type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str,
                        help='Debug Level', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--cache_type', default='LRU', type=str, help='Networked Cache type',
                        choices=['LRU', 'FIFO', 'LFU', 'RR', 'EWMAGRAD', 'LMIN'])
#   parser.add_argument('--cache_keyword_parameters',default='{}',type=str,help='Networked Cache additional constructor parameters')
    parser.add_argument('--query_message_length', default=0.0,
                        type=float, help='Query message length')
    parser.add_argument('--response_message_length', default=0.0,
                        type=float, help='Response message length')
    parser.add_argument('--monitoring_rate', default=1.0,
                        type=float, help='Monitoring rate')
    parser.add_argument('--interpolate', default=False,
                        type=bool, help='Interpolate past states, used by LMIN')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='beta used in EWMA')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma used in LMIN')
    parser.add_argument('--expon', default=0.5, type=float,
                        help='exponent used in LMIN')
    parser.add_argument('--T', default=5., type=float,
                        help='Suffling period used in LMIN')
    args = parser.parse_args()

    args.debug_level = eval("logging."+args.debug_level)

    def graphGenerator():
        if args.graph_type == "erdos_renyi":
            return networkx.erdos_renyi_graph(args.graph_size, args.graph_p)
        if args.graph_type == "balanced_tree":
            ndim = int(np.ceil(np.log(args.graph_size) /
                       np.log(args.graph_degree)))
            return networkx.balanced_tree(args.graph_degree, ndim)
        if args.graph_type == "cicular_ladder":
            ndim = int(np.ceil(args.graph_size*0.5))
            return networkx.circular_ladder_graph(ndim)
        if args.graph_type == "cycle":
            return networkx.cycle_graph(args.graph_size)
        if args.graph_type == 'grid_2d':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim, ndim)
        if args.graph_type == 'lollipop':
            ndim = int(np.ceil(args.graph_size*0.5))
            return networkx.lollipop_graph(ndim, ndim)
        if args.graph_type == 'expander':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.margulis_gabber_galil_graph(ndim)
        if args.graph_type == "hypercube":
            ndim = int(np.ceil(np.log(args.graph_size)/np.log(2.0)))
            return networkx.hypercube_graph(ndim)
        if args.graph_type == "star":
            ndim = args.graph_size-1
            return networkx.star_graph(ndim)
        if args.graph_type == 'barabasi_albert':
            return networkx.barabasi_albert_graph(args.graph_size, args.graph_degree)
        if args.graph_type == 'watts_strogatz':
            return networkx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
        if args.graph_type == 'regular':
            return networkx.random_regular_graph(args.graph_degree, args.graph_size)
        if args.graph_type == 'powerlaw_tree':
            return networkx.random_powerlaw_tree(args.graph_size)
        if args.graph_type == 'small_world':
            ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.navigable_small_world_graph(ndim)
        if args.graph_type == 'geant':
            return topologies.GEANT()
        if args.graph_type == 'dtelekom':
            return topologies.Dtelekom()
        if args.graph_type == 'abilene':
            return topologies.Abilene()
        if args.graph_type == 'servicenetwork':
            return topologies.ServiceNetwork()
        if args.graph_type == 'hetnet':
            print(args.hetnet_grid)
            if args.hetnet_load is not None:
                return topologies.LoadHetNet(args.hetnet_load)
            else:
                if args.hetnet_grid:
                    return topologies.HetNet(args.graph_size, int(args.hetnet_params[0]), args.hetnet_params[1], args.hetnet_params[2], 'grid')
                else:
                    return topologies.HetNet(args.graph_size, int(args.hetnet_params[0]), args.hetnet_params[1], args.hetnet_params[2], 'random')

    def cacheGenerator(capacity, _id):
        if args.cache_type == 'LRU':
            return PriorityNetworkCache(capacity, _id, 'LRU')
        if args.cache_type == 'LFU':
            return PriorityNetworkCache(capacity, _id, 'LFU')
        if args.cache_type == 'FIFO':
            return PriorityNetworkCache(capacity, _id, 'FIFO')
        if args.cache_type == 'RR':
            return PriorityNetworkCache(capacity, _id, 'RR')
        if args.cache_type == 'EWMAGRAD':
            return EWMAGradCache(capacity, _id, beta=args.beta)
        if args.cache_type == 'LMIN':
            return LMinCache(capacity, _id, gamma=args.beta, T=args.T, expon=args.expon, interpolate=args.interpolate)

    logging.basicConfig(level=args.debug_level)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed+2015)

    CONFIG.QUERY_MESSAGE_LENGTH = args.query_message_length
    CONFIG.RESPONSE_MESSAGE_LENGTH = args.response_message_length

    construct_stats = {}

    logging.info('Generating graph and weights...')
    temp_graph = graphGenerator()
    # networkx.draw(temp_graph)
    # plt.draw()
    logging.debug('nodes: '+str(temp_graph.nodes()))
    logging.debug('edges: '+str(temp_graph.edges()))
    G = DiGraph() # TODO: Need to fix how the edges in the directed graph works, because right now users can forward packets

    number_map = dict(zip(temp_graph.nodes(), range(len(temp_graph.nodes()))))
    G.add_nodes_from(number_map.values())
    weights = {}
    if args.graph_type == 'hetnet':
        gains = {}
        #costs = {}
    for (x, y) in temp_graph.edges():
        xx = number_map[x]
        yy = number_map[y]
        G.add_edges_from(((xx, yy), (yy, xx)))
        weights[(xx, yy)] = random.uniform(args.min_weight, args.max_weight)
        weights[(yy, xx)] = weights[(xx, yy)]
        if args.graph_type == 'hetnet':
            gains[(xx, yy)] = temp_graph.edges[x, y]['gain'] # Will be necessary for power calculations for HetNet case
            gains[(yy, xx)] = gains[(xx, yy)]
            #costs[(xx, yy)] = temp_graph.edges[x, y]['cost']
            #costs[(yy, xx)] = costs[(xx, yy)]
            G.edges[xx,yy]['cost'] = temp_graph.edges[x, y]['cost']
            if yy >= len(G.nodes()) - args.query_nodes: # If the edge is to a user, make cost infinite on the reverse path to discourage routing through users
                G.edges[yy,xx]['cost'] = 1e6 * temp_graph.edges[x, y]['cost']
            else:
                G.edges[yy,xx]['cost'] = temp_graph.edges[x, y]['cost']
    #if args.graph_type == 'hetnet':
        #for (xx,yy) in G.edges():
            #G.edges[xx,yy]['cost'] = costs[(xx,yy)]
    graph_size = G.number_of_nodes()
    edge_size = G.number_of_edges()
    logging.info('...done. Created graph with %d nodes and %d edges' %
                 (graph_size, edge_size))
    logging.debug('G is:'+str(G.nodes())+str(G.edges()))
    construct_stats['graph_size'] = graph_size
    construct_stats['edge_size'] = edge_size

    logging.info('Generating item sources...')
    if args.graph_type == 'hetnet': # For now assume MC is the source for all items in HetNet case
        item_sources = dict((item, [0]) for item in range(args.catalog_size))
    else:
        item_sources = dict((item, [list(G.nodes())[source]]) for item, source in zip(range(
            args.catalog_size), np.random.choice(range(graph_size), args.catalog_size)))
    logging.info('...done. Generated %d sources' % len(item_sources))
    logging.debug('Generated sources:')
    for item in item_sources:
        logging.debug(pp([item, ':', item_sources[item]]))

    construct_stats['sources'] = len(item_sources)

    logging.info('Generating query node list...')
    if args.graph_type == 'hetnet':
        query_node_list = [list(G.nodes())[i] for i in range(len(G.nodes()) - args.query_nodes, len(G.nodes()))] # User nodes are query nodes for HetNet
    else:
        query_node_list = [list(G.nodes())[i] for i in random.sample(
            range(graph_size), args.query_nodes)]
    logging.info('...done. Generated %d query nodes.' % len(query_node_list))

    construct_stats['query_nodes'] = len(query_node_list)

    logging.info('Generating demands...')
    if args.demand_distribution == 'powerlaw':
        def factor(i): return (1.0+i)**(-args.powerlaw_exp)
    else:
        def factor(i): return 1.0
    pmf = np.array([factor(i) for i in range(args.catalog_size)])
    pmf /= sum(pmf)
    distr = rv_discrete(values=(range(args.catalog_size), pmf))
    if args.catalog_size <= args.demand_size:
        items_requested = list(distr.rvs(
            size=(args.demand_size-args.catalog_size))) + list(range(args.catalog_size))
    else:
        items_requested = list(distr.rvs(size=args.demand_size))

    random.shuffle(items_requested)

    demands_per_query_node = args.demand_size // args.query_nodes
    remainder = args.demand_size % args.query_nodes
    demands = []
    for i, x in enumerate(query_node_list):
        dem = demands_per_query_node
        if i < remainder:
            dem = dem+1
        if args.graph_type == 'hetnet':            
            new_dems = [Demand(items_requested[pos], shortest_path(G, x, number_map["nodeMC"], weight='cost'), # For HetNet shortest path depends on link costs and is always routed to MC
                           random.uniform(args.min_rate, args.max_rate)) for pos in range(len(demands), len(demands)+dem)]
            #print(list(temp_graph.nodes())[x])
            #new_dems = [Demand(items_requested[pos], shortest_path(temp_graph, list(temp_graph.nodes())[x], "nodeMC", weight='cost'),
            #                random.uniform(args.min_rate, args.max_rate)) for pos in range(len(demands), len(demands)+dem)]
        else:
            new_dems = [Demand(items_requested[pos], shortest_path(G, x, item_sources[items_requested[pos]][0], weight='weight'),
                           random.uniform(args.min_rate, args.max_rate)) for pos in range(len(demands), len(demands)+dem)]
        logging.debug(pp(new_dems))
        demands = demands + new_dems

    logging.info('...done. Generated %d demands' % len(demands))
    #plt.hist([ d.item for d in demands], bins=np.arange(args.catalog_size)+0.5)
    # plt.show()

    construct_stats['demands'] = len(demands)

    logging.info('Generating capacities...')
    if args.graph_type == 'hetnet': # In HetNet case, SCs have capacity equal to passed max capacity, users have 0 and MC currently has capacity equal to catalog
        capacities = {0: args.catalog_size}
        capacities.update(dict((x,args.max_capacity) for x in range(1,len(G.nodes()) - args.query_nodes)))
        capacities.update(dict((x,0) for x in range(len(G.nodes()) - args.query_nodes, len(G.nodes))))
    else:
        capacities = dict((x, random.randint(args.min_capacity,
                      args.max_capacity)) for x in G.nodes())
    logging.info('...done. Generated %d caches' % len(capacities))
    logging.debug('Generated capacities:')
    for key in capacities:
        logging.debug(pp([key, ':', capacities[key]]))

    logging.info('Building CacheNetwork')
    cnx = CacheNetwork(G, cacheGenerator, demands, item_sources, capacities, weights, weights,
                       args.warmup, args.monitoring_rate, args.demand_change_rate, args.min_rate, args.max_rate) # This initializes a random process for link delays based on weights, it will be different for HetNet
    if args.graph_type == 'hetnet':
        cnx.initWireless(args.T, args.graph_size, args.hetnet_params[0], gains, args.cache_type, *args.wireless_consts) # For HetNet we have to initialize wireless parameters, currently only gains need to be passed, sinr_min/max, power_max and noise can also be set
    logging.info('...done')

    Y, res = cnx.minimizeRelaxation()

    logging.info('Optimal Relaxation is: '+str(cnx.relaxation(Y)))
    logging.info('Expected caching gain at relaxation point is: ' +
                 str(cnx.expected_caching_gain(Y)))

    optimal_stats = {}
    optimal_stats['res'] = res
    optimal_stats['Y'] = Y
    optimal_stats['L'] = cnx.relaxation(Y)
    optimal_stats['F'] = cnx.expected_caching_gain(Y)

    if args.cache_type == "LMIN":
        for x in cnx.nodes():
            cnx.node[x]['cache'].startShuffleProcess(cnx.env)

    cnx.run(args.time)

    if args.graph_type == 'hetnet':
        out = args.outputfile + "%s_%s" % (args.graph_type, args.cache_type)
        if args.hetnet_load is not None:
            out += args.hetnet_load.split("topfiles/top")[1]
        else:
            top_file = 'topfiles/top_V' + str(args.graph_size) + '_SC' + str(int(args.hetnet_params[0])) + '_R' + str(args.hetnet_params[1]) + '_exp' + str(args.hetnet_params[2])
            top_file = uniqueify(top_file, 'last')
            out += top_file.split("topfiles/top")[1]        
        out += "_%ditems_%ddemands_%ftime_%fpcap_%fsinr_%dcachecap" % (args.catalog_size, args.demand_size, args.time,
            args.wireless_consts[0], args.wireless_consts[1], args.max_capacity)

        with open(out, 'wb+') as f:
            pickle.dump([args, cnx.wirelessStats], f)
    else:
        demand_stats = {}
        node_stats = {}
        network_stats = {}

        for d in cnx.demands:
            demand_stats[str(d)] = cnx.demands[d]['stats']
            demand_stats[str(
                d)]['queries_spawned'] = cnx.demands[d]['queries_spawned']
            demand_stats[str(
                d)]['queries_satisfied'] = cnx.demands[d]['queries_satisfied']

        for x in cnx.nodes():
            node_stats[x] = cnx.node[x]['cache'].stats

        network_stats['demand'] = cnx.demandstats
        network_stats['fun'] = cnx.funstats
        network_stats['opt'] = cnx.optstats

        out = args.outputfile+"%s_%s_%ditems_%dnodes_%dquerynodes_%ddemands_%ftime_%fchange_%fgamma_%fexpon%fbeta" % (
            args.graph_type, args.cache_type, args.catalog_size, args.graph_size, args.query_nodes, args.demand_size, args.time, args.demand_change_rate, args.gamma, args.expon, args.beta)

        with open(out, 'wb') as f:
            pickle.dump([args, construct_stats, optimal_stats,
                        demand_stats, node_stats, network_stats], f)

#   for d in cnx.demands:
#	print d.item, d.rate, d.requests_tally.count()/time, len(d.path), d.hops_tally.mean(), d.weight_tally.mean(), d.time_tally.mean(), d.hit_source_tally.mean()

#   for x in cnx.nodes():
#	cache = cnx.node[x]['cache']
#	print x,cache.queries_tally.count(), cache.hits_tally.mean(), cache.downloads_tally.count()/time

# def plot_ecdf(y,x_label):
#     ecdf = ECDF(y)
#     x= sorted(list(set(y)))
#     plt.plot(x,ecdf(x))
#     plt.xlabel(x_label)
#     plt.ylabel('CDF')
#     plt.show()


if __name__ == "__main__":
    main()
