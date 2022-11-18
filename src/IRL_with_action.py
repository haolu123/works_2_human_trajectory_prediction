import numpy as np
import copy 
from tqdm import tqdm
import numba     # import the types
from numba.experimental import jitclass

class IRL:

    def __init__(self, FA) -> None:
        self.features_get = FA
        self.theta = np.random.uniform(-1,0,self.features_get.Feature_dims)
        self.action_space = [(1,1),(0,1),(-1,1),(1,0),(0,0),(-1,0),(1,-1),(0,-1),(-1,-1)]
        self.feature_map = self.features_get.get_feature_map()
        self.fp_shape = self.features_get.fp.shape

        n_position = self.fp_shape[0]*self.fp_shape[1]
        n_action = len(self.action_space)
        self.pi = np.zeros((n_position, n_action, n_action))
        self.Q = np.zeros((n_position, n_action, n_action))
        self.V = -1000 * np.ones((n_position, n_action))


    def get_mean_features(self,paths):
        sum_f = np.zeros(self.features_get.Feature_dims)
        count_f = 0
        print("travel all paths")
        for path in tqdm(paths):
            for s in range(len(path)):
                if s == 0:
                    a_p = (0,0)
                else:
                    a_p = (path[s][0]-path[s-1][0], path[s][1]-path[s-1][1])
                if s >= len(path)-1:
                    a_c = (0,0)
                else:
                    a_c = (path[s+1][0]-path[s][0], path[s+1][1]-path[s][1])

                f = self.features_get.get_features(path[s], a_p, a_c)
                sum_f += f
                count_f += 1
        return sum_f/count_f

    def backward_pass(self, AreaInt, iter_num, hot_start=False):
        n_position = self.fp_shape[0]*self.fp_shape[1]
        n_action = len(self.action_space)
        if not hot_start:
            V = -1000 * np.ones((n_position, n_action))
            Q = np.zeros((n_position, n_action, n_action))
        else:
            V = self.V
            Q = self.Q

        # feature_map = self.features_get.get_feature_map()
        R = self.get_reward()
        for i in range(iter_num):
            V[AreaInt[0]*self.fp_shape[0] + AreaInt[1], :] = 0
            Q = R + 0.99 * self.get_shift_V(V)
            V = self.softmax(Q)
            if np.max(np.abs(self.V - V))<0.01:
                break
            self.Q = Q
            self.V = V
        pi = np.exp(Q-np.repeat(V[:,:,np.newaxis], len(self.action_space), axis=2)) 
        self.pi = pi
        self.modify_pi()
        

    def get_reward(self):
        n_position = self.fp_shape[0]*self.fp_shape[1]
        n_action = len(self.action_space)

        R = np.zeros((n_position, n_action, n_action))
        
        position_map = self.feature_map[0] @ self.theta[:4].reshape(-1)
        action_map = self.feature_map[1] @ self.theta[4:].reshape(-1)
        position_map = position_map.reshape((-1,1,1))
        action_map = action_map.reshape(1,n_action, n_action)       
        p_map_v = np.tile(position_map, (1,n_action,n_action))
        a_map_v = np.tile(action_map, (n_position,1,1))
        R = p_map_v + a_map_v
        return R

    def modify_pi(self):
        n_action = len(self.action_space)
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

        self.pi = self.pi.reshape((self.fp_shape[0], self.fp_shape[1], n_action, n_action))
        for i in range(self.fp_shape[0]):
            self.pi[i, 0, :, a_left] = 0
            self.pi[i,-1, :, a_right] = 0
        for j in range(self.fp_shape[1]):
            self.pi[0, j, :, a_up] = 0
            self.pi[-1,j, :, a_down] = 0
        self.pi = self.pi/(np.sum(self.pi,axis=3).reshape(self.pi.shape[0],self.pi.shape[1],self.pi.shape[2],1))
        self.pi = self.pi.reshape((self.fp_shape[0]*self.fp_shape[1], n_action, n_action))

    def get_shift_V(self, V):
        V_list = []
        """
        V[n_p,n_a] -> V[n_x,n_y,n_a]
        shift V[n_x,n_y,n_a]
        V[s,a] = V[x_invers_shift,y_inverse_shift,a]
        
        stack all V[s,a] along a.
        """
        V = V.reshape((self.fp_shape[0], self.fp_shape[1], len(self.action_space)))

        V_pad = np.pad(V, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(-1000, -1000))
        for a_i in range(len(self.action_space)):
            a = self.action_space[a_i]
            V_shift = copy.deepcopy(V_pad)
            
            V_shift = np.roll(V_shift, -a[0], axis=0)
            V_shift = np.roll(V_shift, -a[1], axis=1)
            
            V_unpad = V_shift[1:-1, 1:-1, :]
            V_s_a_vector = V_unpad[:,:,a_i].reshape(self.fp_shape[0] * self.fp_shape[1])
            V_s_a = np.tile(V_s_a_vector.reshape((-1,1)), (1,len(self.action_space)))
            V_list.append(V_s_a)
        V_shift_all = np.stack(V_list,axis=2)
        return V_shift_all
    
    def softmax(self, Q):
        V = np.log(np.sum(np.exp(Q)+0.00001, axis=2))
        return V

    def forward_pass(self, AreaInt, iter_num):
        # feature_map = self.features_get.get_feature_map()
        n_action = len(self.action_space)
        R = self.get_reward()
        D_init = np.sum(np.exp(R), axis=2)
        D_init = D_init/np.sum(D_init)
        D_init = D_init.reshape(self.fp_shape+(len(self.action_space),))
        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        feature_map_grid = self.get_feature_map_grid()
        feature_mean = 0
        for iter in range(iter_num):
            D[AreaInt[0], AreaInt[1], :] = 0
            D_s_a = self.pi.reshape((self.fp_shape[0],self.fp_shape[1],n_action,n_action)) * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = np.sum(D_with_act, axis=3)
            D_sum += D
            feature_mean += D_with_act.reshape(1,-1) @ feature_map_grid.reshape(-1,feature_map_grid.shape[2])
            if np.min(D) < 0.01:
                break
        return feature_mean

    def get_feature_map_grid(self):
        n_position = self.fp_shape[0] * self.fp_shape[1]
        n_action = len(self.action_space)
        feature_position_v = self.feature_map[0].reshape((-1,1,4))
        feature_action_v = self.feature_map[1].reshape((1,-1,2))
        feature_position = np.tile(feature_position_v, (1,n_action*n_action,1))
        feature_action = np.tile(feature_action_v, (n_position,1,1))
        feature_map_grid = np.concatenate((feature_position,feature_action), axis=2)
        return feature_map_grid

    def get_shift_D(self, D):
        D_pad = np.pad(D, ((1, 1), (1, 1), (0,0), (0,0)), 'constant', constant_values=(0, 0))
        for a_i in range(len(self.action_space)):
            D_pad[:,:,:,a_i] = np.roll(D_pad[:,:,:,a_i], self.action_space[a_i][0], axis=0)
            D_pad[:,:,:,a_i] = np.roll(D_pad[:,:,:,a_i], self.action_space[a_i][1], axis=1)
            D_unpad = D_pad[1:-1, 1:-1,:,a_i]
            D_next = np.zeros(D_unpad.shape+(len(self.action_space),))
            D_next[:,:,a_i,a_i] = np.sum(D_unpad, axis=2)
        return D_next

    def gradiant_theta(self, AreaInt, iter_num, paths, step_size, max_loop = 500, epsil=0.001):
        # initial
        feature_mean_ob = self.get_mean_features(paths)
        for i in tqdm(range(max_loop)):
            self.backward_pass(AreaInt, iter_num)
            feature_mean = self.forward_pass(AreaInt, iter_num)
            d_l = feature_mean_ob - feature_mean
            if np.max(d_l) <= epsil:
                print('converged')
                break
            self.theta = self.theta - step_size * d_l
            self.theta = self.theta.reshape(-1)
    def pred_avg_heatmap(self, s_init, path_length, AreaInt):
        # feature_map = self.features_get.get_feature_map()
        n_action = len(self.action_space)
        R = self.get_reward()
        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], 4] = 1
        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        for iter in range(path_length):
            D[AreaInt[0], AreaInt[1], :] = 0
            D_s_a = self.pi.reshape((self.fp_shape[0],self.fp_shape[1],n_action,n_action)) * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D = self.get_shift_D(D_s_a)
            D_sum += D
        ##############################################
        space_freq = np.sum(D_sum, axis = 2)
        return space_freq

