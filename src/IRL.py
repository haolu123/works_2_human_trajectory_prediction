import numpy as np
import copy 

class IRL:

    def __init__(self, FA) -> None:
        self.features_get = FA
        self.theta = np.random.uniform(-1,0,self.features_get.Feature_dims)
        self.action_space = [(1,1),(0,1),(-1,1),(1,0),(0,0),(-1,0),(1,-1),(0,-1),(-1,-1)]
        self.pi = None
        self.Q = None
        self.V = None
        self.feature_map = self.features_get.get_feature_map()
        self.fp_shape = self.features_get.fp.shape

    def get_mean_features(self,paths):
        sum_f = np.zeros(self.features_get.Feature_dims)
        count_f = 0
        for path in paths:
            for s in path:
                f = self.features_get.get_features(s)
                sum_f += f
                count_f += 1
        return sum_f/count_f

    def backward_pass(self, AreaInt, iter_num, hot_start=False):
        if not hot_start:
            V = -1000 * np.ones(self.features_get.fp.shape)
            Q = np.zeros(self.features_get.fp.shape+(len(self.action_space),))
        else:
            V = self.V
            Q = self.Q

        # feature_map = self.features_get.get_feature_map()

        for i in range(iter_num):
            V[tuple(AreaInt)] = 0
            R = self.feature_map @ self.theta
            Q = np.repeat(R[:,:,np.newaxis], len(self.action_space), axis=2) + 0.99 * self.get_shift_V(V)
            V = self.softmax(Q)
        pi = np.exp(Q-np.repeat(V[:,:,np.newaxis], len(self.action_space), axis=2)) 
        self.pi = pi
        self.modify_pi()
        self.Q = Q
        self.V = V

    def modify_pi(self):
        a_left = []
        a_right = []
        a_up = []
        a_down = []
        for i in range(len(self.action_space)):
            if self.action_space[i][0] == -1:
                a_left.append(i)
            if self.action_space[i][0] == 1:
                a_right.append(i)
            if self.action_space[i][1] == -1:
                a_up.append(i)
            if self.action_space[i][1] == 1:
                a_down.append(i)

        for i in range(self.fp_shape[0]):
            self.pi[i, 0, a_left] = 0
            self.pi[i,-1, a_right] = 0
        for j in range(self.fp_shape[1]):
            self.pi[0, j, a_up] = 0
            self.pi[-1,j, a_down] = 0
        self.pi = self.pi/(np.sum(self.pi,axis=2).reshape(self.pi.shape[0],self.pi.shape[1],1))

    def get_shift_V(self, V):
        V_list = []
        V_pad = np.pad(V, ((1, 1), (1, 1)), 'constant', constant_values=(-1000, -1000))
        for a in self.action_space:
            V_shift = copy.deepcopy(V_pad)
            
            V_shift = np.roll(V_shift, a[0], axis=1)
            V_shift = np.roll(V_shift, a[1], axis=0)
            
            V_unpad = V_shift[1:-1, 1:-1]
            V_list.append(V_unpad)
        V_shift_all = np.stack(V_list,axis=2)
        return V_shift_all
    
    def softmax(self, Q):
        V = np.log(np.sum(np.exp(Q)+0.00001, axis=2))
        return V

    def forward_pass(self, AreaInt, iter_num):
        # feature_map = self.features_get.get_feature_map()
        D_init = np.exp(self.feature_map @ self.theta)
        D_init = D_init/np.sum(D_init)
        D_sum = np.zeros(self.fp_shape)
        D = D_init
        for iter in range(iter_num):
            D[tuple(AreaInt)] = 0
            D = np.sum(self.get_shift_D(self.pi * D.reshape((D.shape[0],D.shape[1],1))), axis = 2)
            D_sum += D
        ##############################################
        feature_mean = D_sum @ self.feature_map
        return feature_mean

    def get_shift_D(self, D):
        D_pad = np.pad(D, ((1, 1), (1, 1), (0,0)), 'constant', constant_values=(0, 0))
        for a_i in range(len(self.action_space)):
            D_pad[:,:,a_i] = np.roll(D_pad[:,:,a_i], self.action_space[a_i][0], axis=1)
            D_pad[:,:,a_i] = np.roll(D_pad[:,:,a_i], self.action_space[a_i][1], axis=0)

            D_unpad = D_pad[1:-1, 1:-1,:]
        return D_unpad    

    def gradiant_theta(self, AreaInt, iter_num, paths, step_size, max_loop = 500, epsil=0.01):
        # initial
        feature_mean_ob = self.get_mean_features(paths)
        for i in range(max_loop):
            self.backward_pass(AreaInt, iter_num)
            feature_mean = self.forward_pass(AreaInt, iter_num)
            d_l = feature_mean_ob - feature_mean
            if np.max(d_l) <= epsil:
                print('converged')
                break
            self.theta = self.theta - step_size * d_l

    def pred_avg_heatmap(self, s_init, path_length):
        D_init = np.zeros(self.fp_shape)
        D_init[s_init] = 1
        D_sum = np.zeros(self.fp_shape)
        D = D_init
        for iter in range(path_length):
            D = np.sum(self.get_shift_D(self.pi * D.reshape((D.shape[0],D.shape[1],1))), axis = 2)
            D_sum += D
        return D_sum/np.sum(D_sum)

