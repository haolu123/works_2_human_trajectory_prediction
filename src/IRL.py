import numpy as np
import copy 

class IRL:

    def __init__(self, FA) -> None:
        self.features_get = FA
        self.theta = np.random.uniform(0,1,self.features_get.Feature_dim)
        self.action_space = [(1,1),(0,1),(-1,1),(1,0),(0,0),(-1,0),(1,-1),(0,-1),(-1,-1)]
        self.pi = None
        self.Q = None
        self.V = None

    def get_mean_features(self,paths):
        sum_f = np.zeros(self.features_get.Feature_dim)
        count_f = 0
        for path in paths:
            for s in path:
                f = self.features_get.get_features(s)
                sum_f += f
                count_f += 1
        return sum_f/count_f

    def backward_pass(self, AreaInt, iter_num):

        V = -1000 * np.ones(self.features_get.fp.shape)
        Q = np.zeros(self.features_get.fp.shape, len(self.action_space))

        feature_map = self.features_get.get_feature_map()

        for i in range(iter_num):
            V[AreaInt] = 0
            Q = feature_map @ self.theta + self.get_shift_V(V)
            V = self.softmax(Q)
        pi = np.exp(Q-V) 
        self.pi = pi
        self.Q = Q
        self.V = V
        
    def get_shift_V(self, V):
        V_list = []
        V_pad = np.pad(V, ((1, 1), (1, 1)), 'constant', constant_values=(-1000, -1000))
        for a in self.action_space:
            V_shift = copy.deepcopy(V_pad)
            
            V_shift = np.roll(V_shift, a[0], axis=1)
            V_shift = np.roll(V_shift, a[1], axis=0)
                
