#%%
import numpy as np
import copy 
from tqdm import tqdm
import numba     # import the types
from numba.experimental import jitclass
from common.engine import get_fp, A_star_simulation
import os,pickle
import matplotlib.pyplot as plt

#%%
# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

list_type = numba.typeof([[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]])
tuple_type = numba.typeof((100,100))
spec = [
    ('Feature_dims', numba.int32),
    ('features', numba.float64[:]),
    ('fp', numba.int32[:,:]),
    ('action_space', list_type),
    # ('feature_map', numba.float32[:,:]),
    # ('theta', numba.float32[:]),
    # ('n_action', numba.int32),
    ('fp_shape', tuple_type),
    # ('pi', numba.float32[:,:]),
    # ('Q', numba.float32[:,:]),
    # ('V', numba.float32[:,:]),
    # ('R', numba.float32[:,:,:,:])
]

@jitclass(spec)
class IRL:

    def __init__(self, fp) -> None:
        self.Feature_dims = 6
        self.features = np.zeros(6)
        self.fp = fp.astype('int')
        # self.modify_fp()
        self.action_space = [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]
        # self.feature_map = self.get_feature_map()

        # self.theta = np.random.uniform(-1,0,self.Feature_dims)
        self.fp_shape = fp.shape

        # n_action = len(self.action_space)
        # self.pi = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        # self.Q = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        # self.V = -1000 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action))
        # self.R = self.get_reward()

    # def modify_fp(self):
    #     self.fp[self.fp==2] = 1
    #     self.fp[self.fp==3] = 2
    #     self.fp[self.fp>3] = 0

    # def get_features(self, s, a_p, a_c):
    #     '''
    #         feature map
    #         Args:
    #             s : (x,y) position
    #             a_p : (vx,vy) previous velocity
    #             a_c : (vx,vy) current velocity
    #         Returns:
    #             features : change self.features
    #     '''
    #     material = self.fp[s]
    #     b = np.zeros(self.fp.max()+1)
    #     b[int(material)] = 1
    #     self.features[:b.size] = b
    #     self.features[b.size+1] = self.min_distance_to_wall(s)
    #     self.features[-2] = np.dot(a_p,a_c)
    #     self.features[-1] = np.linalg.norm(a_c) - np.linalg.norm(a_p)
    #     return self.features

    # def get_feature_map(self):
    #     fp_x_size = self.fp.shape[0]
    #     fp_y_size = self.fp.shape[1]
    #     n_action = len(self.action_space)
        
    #     encoded_fp = np.zeros((fp_x_size,fp_y_size, self.fp.max()+1), dtype=int)
    #     encoded_fp[all_idx(self.fp, axis=2)] = 1 

    #     dist_map = self.distance_map()
    #     fp_feature_map = np.concatenate((encoded_fp,dist_map.reshape(dist_map.shape[0],dist_map.shape[1],1)), axis=2)

    #     # action_map : [previous action, current action]
    #     action_map = np.zeros((n_action, n_action,2))

    #     for i in range(n_action):
    #         for j in range(n_action):
    #             a_c = self.action_space[j]
    #             a_p = self.action_space[i]
    #             action_map[i,j,0] = np.dot(a_c, a_p)
    #             action_map[i,j,1] = np.linalg.norm(a_c) - np.linalg.norm(a_p)
        
    #     fp_feature_map_ls = np.tile(fp_feature_map.reshape((fp_x_size,fp_y_size,1,1,4)), (1,1,n_action,n_action,1))
    #     action_feature_map_ls = np.tile(action_map.reshape((1,1,n_action,n_action,2)), (fp_x_size, fp_y_size, 1, 1, 1))
    #     feature_map = np.concatenate((fp_feature_map_ls, action_feature_map_ls), axis=4)
    #     return feature_map

    # def distance_map(self):
    #     dist_map = np.zeros(self.fp.shape)
    #     for i in range(self.fp.shape[0]):
    #         for j in range(self.fp.shape[1]):
    #             dist_map[i,j] = self.min_distance_to_wall((i,j))
    #     return dist_map
        
    # def min_distance_to_wall(self,point):
    #     """
    #         get the minimum distance to the wall or obstacles
    #     """
    #     detect_dir = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    #     for i in range(10):
    #         for j in detect_dir:
    #             x = i*j[0] + point[0]
    #             y = i*j[1] + point[1]
    #             if x >= self.fp.shape[0] or y >=  self.fp.shape[1]:
    #                 break
    #             if self.fp[x,y] == 1 or self.fp[x,y] == 2:
    #                 return i/10
    #     return 10/10

    # def get_mean_features(self,paths):
    #     sum_f = np.zeros(self.Feature_dims)
    #     count_f = 0
    #     print("travel all paths")
    #     for path in tqdm(paths):
    #         for s in range(len(path)):
    #             if s == 0:
    #                 a_p = (0,0)
    #             else:
    #                 a_p = (path[s][0]-path[s-1][0], path[s][1]-path[s-1][1])
    #             if s >= len(path)-1:
    #                 a_c = (0,0)
    #             else:
    #                 a_c = (path[s+1][0]-path[s][0], path[s+1][1]-path[s][1])

    #             f = self.get_features(path[s], a_p, a_c)
    #             sum_f += f
    #             count_f += 1
    #     return sum_f/count_f

    # def backward_pass(self, AreaInt, iter_num, hot_start=False):
    #     n_position = self.fp_shape[0]*self.fp_shape[1]
    #     n_action = len(self.action_space)
    #     if not hot_start:
    #         V = -1000 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action))
    #         Q = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
    #     else:
    #         V = self.V
    #         Q = self.Q

    #     # feature_map = self.features_get.get_feature_map()
    #     for i in range(iter_num):
    #         V[AreaInt[0], AreaInt[1], :] = 0
    #         Q = self.R + 0.99 * self.get_shift_V(V)
    #         V = self.softmax(Q)
    #         if np.max(np.abs(self.V - V))<0.01:
    #             break
    #         self.Q = Q
    #         self.V = V
    #     pi = np.exp(Q-np.repeat(V[:,:,:,np.newaxis], len(self.action_space), axis=3)) 
    #     self.pi = pi
    #     self.modify_pi()
        

    # def get_reward(self):
    #     n_action = len(self.action_space)

    #     R = np.zeros((self.fp_shape[0], self.fp_shape[1], n_action, n_action))
    #     R = self.feature_map @ self.theta.reshape(-1)

    #     return R

    # def modify_pi(self):
    #     n_action = len(self.action_space)
    #     a_left = []
    #     a_right = []
    #     a_up = []
    #     a_down = []
    #     for i in range(len(self.action_space)):
    #         if self.action_space[i][0] == -1:
    #             a_left.append(i)
    #         if self.action_space[i][0] == 1:
    #             a_right.append(i)
    #         if self.action_space[i][1] == -1:
    #             a_up.append(i)
    #         if self.action_space[i][1] == 1:
    #             a_down.append(i)

    #     for i in range(self.fp_shape[0]):
    #         self.pi[i, 0, :, a_left] = 0
    #         self.pi[i,-1, :, a_right] = 0
    #     for j in range(self.fp_shape[1]):
    #         self.pi[0, j, :, a_up] = 0
    #         self.pi[-1,j, :, a_down] = 0
    #     self.pi = self.pi/(np.sum(self.pi,axis=3).reshape(self.pi.shape[0],self.pi.shape[1],self.pi.shape[2],1))
        

    # def get_shift_V(self, V):
    #     n_action = len(self.action_space)
    #     V_paa = np.zeros((self.fp_shape[0], self.fp_shape[1], n_action, n_action))

    #     V_pad = np.pad(V, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(-1000, -1000))

    #     for a_i in range(len(self.action_space)):
    #         a = self.action_space[a_i]

    #         V_paa[:,:,:,a_i] = V_pad[1-a[0]:self.fp_shape[0]+1-a[0], 1-a[1]:self.fp_shape[1]+1-a[1], a_i].reshape((self.fp_shape[0],self.fp_shape[1],1))
            
    #     return V_paa
    
    # def softmax(self, Q):
    #     V = np.log(np.sum(np.exp(Q)+0.00001, axis=3))
    #     return V

    # def forward_pass(self, AreaInt, iter_num):
    #     # feature_map = self.features_get.get_feature_map()
    #     n_action = len(self.action_space)

    #     D_init = np.sum(np.exp(self.R), axis=3)
    #     D_init = D_init/np.sum(D_init)

    #     D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
    #     D = D_init
    #     feature_mean = 0
    #     for iter in range(iter_num):
    #         D[AreaInt[0], AreaInt[1], :] = 0
    #         D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
    #         D = self.get_shift_D(D_s_a)
    #         D_sum += D
    #         feature_mean += D_s_a.reshape(1,-1) @ self.feature_map.reshape(-1,self.Feature_dims)
    #         if np.min(D) < 0.01:
    #             break
    #     return feature_mean


    # def get_shift_D(self, D):
    #     D_pad = np.pad(D, ((1, 1), (1, 1), (0,0), (0,0)), 'constant', constant_values=(0, 0))
    #     D_next = np.zeros((self.fp_shape[0], self.fp_shape[1], len(self.action_space)))
    #     for a_i in range(len(self.action_space)):
    #         a = self.action_space[a_i]
    #         D_next[:,:,a_i] = np.sum(D_pad[1+a[0]:self.fp_shape[0]+1+a[0], 1+a[1]:self.fp_shape[1]+1+a[1], :, a_i], axis=2)
    #     return D_next

    # def gradiant_theta(self, AreaInt, iter_num, paths, step_size, max_loop = 500, epsil=0.001):
    #     # initial
    #     feature_mean_ob = self.get_mean_features(paths)
    #     for i in tqdm(range(max_loop)):
    #         self.backward_pass(AreaInt, iter_num)
    #         feature_mean = self.forward_pass(AreaInt, iter_num)
    #         d_l = feature_mean_ob - feature_mean
    #         if np.max(d_l) <= epsil:
    #             print('converged')
    #             break
    #         self.theta = self.theta - step_size * d_l
    #         self.theta = self.theta.reshape(-1)


    # def pred_avg_heatmap(self, s_init, path_length, AreaInt):

    #      # feature_map = self.features_get.get_feature_map()
    #     n_action = len(self.action_space)

    #     D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
    #     D_init[s_init[0], s_init[1], 4] = 1

    #     D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
    #     D = D_init

    #     for iter in range(path_length):
    #         D[AreaInt[0], AreaInt[1], :] = 0
    #         D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
    #         D_with_act = self.get_shift_D(D_s_a)
    #         D = np.sum(D_with_act, axis=3)
    #         D_sum += D

    #         if np.min(D) < 0.01:
    #             break
    #     space_freq = np.sum(D_sum, axis = 2)
       
    #     return space_freq
#%%
grid_width = 10
baseColor_fn = 'baseColor.png'
fp_zone_fn = 'fp2_zone.png'
fp_code_fn = 'fp2_code.png'

tempdir  = './temp_result/'
with open(tempdir + 'fp_grid.pickle', 'rb') as f:
    fp = pickle.load(f)

with open(tempdir + 'path_all.pickle', 'rb') as f:
    path_train = pickle.load(f)

s_goal_list = [i[-1] for i in path_train]
s_goal_set = set()
for s in s_goal_list:
    if s not in s_goal_set:
        s_goal_set.add(s)

#%%
irl = IRL(fp)
# %%
