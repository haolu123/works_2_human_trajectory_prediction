import numpy as np
import copy, pickle
from tqdm import tqdm
import numba     # import the types
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import math
import heapq

# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[0]-p2[0])**2)
# list_type = numba.types.ListType(numba.types.int32)
# tuple_type = numba.typeof((100,100))
# spec = [
#     ('Feature_dims', numba.int32)
#     ('features', numba.float32[:]),
#     ('fp', numba.int32[:,:]),
#     ('action_space', numba.types.ListType(list_type)),
#     ('feature_map', numba.float32[:,:]),
#     ('theta', numba.float32[:]),
#     ('n_action', numba.int32),
#     ('fp_shape', tuple_type),
#     ('pi', numba.float32[:,:]),
#     ('Q', numba.float32[:,:]),
#     ('V', numba.float32[:,:]),
#     ('R', numba.float32[:,:,:,:])
# ]

# @jitclass(spec)
class IRL:

    def __init__(self, fp) -> None:
        self.Feature_dims = 2+1+2
        self.features = np.zeros(self.Feature_dims)
        self.fp = fp.astype('int')
        self.modify_fp()
        self.action_space = [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]
        self.feature_map = self.get_feature_map()

        self.theta = np.random.uniform(-0.2,0,self.Feature_dims)
        self.fp_shape = fp.shape

        n_action = len(self.action_space)
        self.pi = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        self.Q = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        self.V = -1000 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action))
        self.R = self.get_reward()
        self.best_policy = []

    def modify_fp(self):
        self.fp[self.fp==2] = 1
        # self.fp[self.fp==4] = 2
        self.fp[self.fp==3] = 0
        self.fp[self.fp>3] = 0

    def get_features(self, s, a_p, a_c):
        '''
            feature map
            Args:
                s : (x,y) position
                a_p : (vx,vy) previous velocity
                a_c : (vx,vy) current velocity
            Returns:
                features : change self.features
        '''
        material = self.fp[s]
        b = np.zeros(self.fp.max()+1)
        b[int(material)] = 1
        self.features[:b.size] = b
        self.features[b.size] = self.min_distance_to_wall(s)
        self.features[b.size+1] = abs(a_p[0]-a_c[0])
        self.features[b.size+2] = abs(a_p[1]-a_c[1])
        # self.features[-1] = np.linalg.norm(a_c) - np.linalg.norm(a_p)
        return self.features

    def get_feature_map(self):
        fp_x_size = self.fp.shape[0]
        fp_y_size = self.fp.shape[1]
        n_action = len(self.action_space)
        
        encoded_fp = np.zeros((fp_x_size,fp_y_size, self.fp.max()+1), dtype=int)
        encoded_fp[all_idx(self.fp, axis=2)] = 1 

        dist_map = self.distance_map()
        fp_feature_map = np.concatenate((encoded_fp,dist_map.reshape(dist_map.shape[0],dist_map.shape[1],1)), axis=2)
        # fp_feature_map = encoded_fp
        # action_map : [previous action, current action]
        action_map = np.zeros((n_action, n_action, 2))

        for i in range(n_action):
            for j in range(n_action):
                a_c = self.action_space[j]
                a_p = self.action_space[i]
                action_map[i,j,0] = abs(a_p[0]- a_c[0])
                action_map[i,j,1] = abs(a_p[1]- a_c[1])
        fp_feature_map_ls = np.tile(fp_feature_map.reshape((fp_x_size,fp_y_size,1,1,-1)), (1,1,n_action,n_action,1))
        action_feature_map_ls = np.tile(action_map.reshape((1,1,n_action,n_action,-1)), (fp_x_size, fp_y_size, 1, 1, 1))
        feature_map = np.concatenate((fp_feature_map_ls, action_feature_map_ls), axis=4)
        return feature_map

    def distance_map(self):
        dist_map = np.zeros(self.fp.shape)
        for i in range(self.fp.shape[0]):
            for j in range(self.fp.shape[1]):
                dist_map[i,j] = self.min_distance_to_wall((i,j))
        return dist_map
        
    def min_distance_to_wall(self,point):
        """
            get the minimum distance to the wall or obstacles
        """
        detect_dir = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for i in range(6):
            for j in detect_dir:
                x = i*j[0] + point[0]
                y = i*j[1] + point[1]
                if x >= self.fp.shape[0] or y >=  self.fp.shape[1]:
                    return 0
                if self.fp[x,y] == 1 or self.fp[x,y] == 2:
                    return i/6
        return 1

    def get_mean_features(self,paths):
        sum_f = np.zeros(self.Feature_dims)
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

                f = self.get_features(path[s], a_p, a_c)
                sum_f += f
                count_f += 1
        return sum_f/count_f

    def backward_pass(self, AreaInt, iter_num, hot_start=False):
        n_position = self.fp_shape[0]*self.fp_shape[1]
        n_action = len(self.action_space)
        if not hot_start:
            V = -1000 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action))
            Q = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        else:
            V = self.V
            Q = self.Q
        self.R = self.get_reward()
        # feature_map = self.features_get.get_feature_map()
        for i in range(iter_num):
            V[AreaInt[0], AreaInt[1], :] = 0
            Q = self.R + 0.99 * self.get_shift_V(V)
            V = self.softmax(Q)
            # if np.max(np.abs(self.V - V))<0.01:
            #     break
            self.Q = Q
            self.V = V
            # plt.imshow(np.sum(V, axis=2))
            # plt.show()
        pi = np.exp(Q-np.repeat(V[:,:,:,np.newaxis], len(self.action_space), axis=3)) 
        self.pi = pi
        self.modify_pi()
        

    def get_reward(self):
        n_action = len(self.action_space)

        R = np.zeros((self.fp_shape[0], self.fp_shape[1], n_action, n_action))
        R = self.feature_map @ self.theta.reshape(-1)

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

        for i in range(self.fp_shape[0]):
            self.pi[i, 0, :, a_left] = 0
            self.pi[i,-1, :, a_right] = 0
        for j in range(self.fp_shape[1]):
            self.pi[0, j, :, a_up] = 0
            self.pi[-1,j, :, a_down] = 0
        self.pi = self.pi/(np.sum(self.pi,axis=3).reshape(self.pi.shape[0],self.pi.shape[1],self.pi.shape[2],1) + 0.000001)
        

    def get_shift_V(self, V):
        n_action = len(self.action_space)
        V_paa = np.zeros((self.fp_shape[0], self.fp_shape[1], n_action, n_action))

        V_pad = np.pad(V, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(-1000, -1000))

        for a_i in range(len(self.action_space)):
            a = self.action_space[a_i]

            V_paa[:,:,:,a_i] = V_pad[1+a[0]:self.fp_shape[0]+1+a[0], 1+a[1]:self.fp_shape[1]+1+a[1], a_i].reshape((self.fp_shape[0],self.fp_shape[1],1))
            
        return V_paa
    
    def softmax(self, Q):
        # b = np.exp(Q)+0.00001
        # V = np.log(np.sum(b, axis=3))
        V = np.max(Q, axis=3)
        return V

    def forward_pass(self, AreaInt, iter_num):
        # feature_map = self.features_get.get_feature_map()
        # [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]

        n_action = len(self.action_space)

        D_init = np.sum(np.exp(self.R), axis=3)
        D_init = D_init/np.sum(D_init)

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        feature_mean = 0
        for iter in range(iter_num):
            D[AreaInt[0], AreaInt[1], :] = 0
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D = self.get_shift_D(D_s_a)
            D_sum += D
            feature_mean += (D_s_a.reshape(1,-1)/(np.sum(D_s_a)+0.000001)) @ self.feature_map.reshape(-1,self.Feature_dims)
            if np.max(D) < 0.0001:
                break
        return feature_mean/iter_num


    def get_shift_D(self, D):
        D_pad = np.pad(D, ((1, 1), (1, 1), (0,0), (0,0)), 'constant', constant_values=(0, 0))
        D_next = np.zeros((self.fp_shape[0], self.fp_shape[1], len(self.action_space)))
        for a_i in range(len(self.action_space)):
            a = self.action_space[a_i]
            D_next[:,:,a_i] = np.sum(D_pad[1-a[0]:self.fp_shape[0]+1-a[0], 1-a[1]:self.fp_shape[1]+1-a[1], :, a_i], axis=2)
        return D_next

    def gradiant_theta(self, AreaInt, iter_num, paths, step_size, max_loop = 500, epsil=0.0001):
        # initial
        feature_mean_ob = self.get_mean_features(paths)
        feature_mean_list = []
        d_l_list = []
        for i in tqdm(range(max_loop)):
            self.backward_pass(AreaInt, iter_num)
            feature_mean = self.forward_pass(AreaInt, iter_num)
            d_l = feature_mean_ob - feature_mean
            # if np.max(d_l) <= epsil:
            #     print('converged')
            #     break
            ################################
            self.theta = self.theta + step_size * d_l
            self.theta = self.theta.reshape(-1)
            feature_mean_list.append(feature_mean)
            d_l_list.append(d_l)
            print(self.theta)
            print(feature_mean)
            print(d_l)
            print(np.max(np.abs(self.R-self.get_reward())))
        with open('./result/feature_list.pickle','wb') as f:
            pickle.dump(feature_mean_list, f)
        with open('./result/gradient_list.pickle', 'wb') as f:
            pickle.dump(d_l_list,f)


    def pred_avg_heatmap(self, s_init, a_init, path_length, AreaInt):

        # feature_map = self.features_get.get_feature_map()
        # [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]
        n_action = len(self.action_space)
        i_action_init = self.action_space.index(a_init)

        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], i_action_init] = 1

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init

        space_freq = np.zeros(self.fp_shape)
        for iter in range(path_length):
            D[AreaInt[0], AreaInt[1], :] = 0
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = D_with_act
            D_sum += D
            if np.max(D) < 0.0001:
                break
        space_freq = np.sum(D_sum, axis = 2)
       
        return space_freq

    def distance_map_2_point(self, p):
        grid = np.ogrid[0:self.fp_shape[0], 0:self.fp_shape[1]]
        x_dist = np.tile(np.square(grid[0]-p[0]).reshape(-1,1), (1,self.fp_shape[1]))
        y_dist = np.tile(np.square(grid[1]-p[1]).reshape(1,-1), (self.fp_shape[0],1))
        dist_map = np.sqrt(x_dist + y_dist)
        return dist_map
    
    """
        3 metrics:
            1. expectation of average distance
            2. average sampled distance
            3. likelihood of the true path
    """
    def mean_distance(self, s_init, a_init, step_num, following_path):
        n_action = len(self.action_space)
        i_action_init = self.action_space.index(a_init)

        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], i_action_init] = 1

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        ave_dist_expct = 0
        for iter in range(1, step_num):
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = D_with_act
            space_freq = np.sum(D, axis = 2)
            dist_map = self.distance_map_2_point(following_path[iter])
            dist_expct = np.sum(space_freq*dist_map)
            ave_dist_expct += dist_expct
        return ave_dist_expct/step_num
    
    
    def sampe_distance(self, s_init, a_init, step_num, following_path):
        n_action = len(self.action_space)
        i_action_init = self.action_space.index(a_init)

        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], i_action_init] = 1

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        ave_dist = 0
        for iter in range(1, step_num):
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = D_with_act
            space_freq = np.sum(D, axis = 2)
            p_e = np.unravel_index(np.argmax(space_freq), space_freq.shape)

            dist = distance(p_e, following_path[iter])
            ave_dist += dist
        return ave_dist/step_num

    def likelihood(self, s_init, a_init, step_num, following_path):
        n_action = len(self.action_space)
        i_action_init = self.action_space.index(a_init)

        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], i_action_init] = 1

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        ave_dist = 0
        for iter in range(1, step_num):
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = D_with_act
        space_freq = np.sum(D, axis = 2)
        p = space_freq[following_path[step_num]]
        return p
    
    def get_metric(self, paths, max_step):
        metric_dist = []
        for step_num in range(1, max_step):
            dist_all = 0

            for count in tqdm(range(1000)):
                while True:
                    i = np.random.choice(len(paths))
                    path = paths[i]
                    if len(path) > 30:
                        break
                
                while True:
                    j = np.random.choice(len(path))
                    if len(path)-j > step_num+1:
                        break

                s_init = path[j]
                if j == 0:
                    a_init = [0,0]
                else:
                    a_init = [path[j][0]-path[j-1][0], path[j][1]-path[j-1][1]]

                following_path = path[j:]
                dist = self.sampe_distance(s_init, a_init, step_num, following_path)
                dist_all += dist
            metric_dist.append(dist_all/1000)
            # for i in range(len(paths)):
            #     path = paths[i]

            #     for j in range(len(path)):
            #         if len(path)-j <= step_num+1:
            #             break
        return metric_dist

    def best_policy_given_destination(self, AreaInt, discount_factor=1, epsilon=0.1):
        """
            get a best policy for give destination,

            Args:
                AreaInt: a list of destinations
            Returns:
                P_l: a list of policy
        """
        n_action = len(self.action_space)
        Env_nS = (self.fp_shape[0],self.fp_shape[1], n_action)
        V_old = np.zeros(Env_nS)
        policy_init = 1/9 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
        best_policy = []
        # V = []
        for i_dest in range(len(AreaInt[0])):
        # for i_dest in range(1):
            policy = policy_init
            delta = 0
            V_old = np.zeros(Env_nS)
            policy_init = 1/9 * np.ones((self.fp_shape[0],self.fp_shape[1], n_action, n_action))
            loop1_max = 100
            flag = True
            while loop1_max>0:
                loop1_max -= 1
                loop2_max = 500
                while loop2_max > 0:
                    loop2_max -= 1
                    V_old[AreaInt[0][i_dest], AreaInt[1][i_dest], :] = 20
                    V_old[0,:,:] = -1
                    V_old[:,0,:] = -1
                    V_old[self.fp_shape[0]-1,:,:] = -1
                    V_old[:,self.fp_shape[1]-1,:] = -1
                    Q = self.R + discount_factor * self.get_shift_V(V_old)
                    # plt.imshow(np.sum(V_old,axis=2))
                    """
                        not finished, 
                        V_new = sum_a (policy(a|s) * Q(s,a))
                        delta = max(delta, |V_new-V_old|)                   
                    """
                    V_new = np.sum(policy * Q, axis=3)
                    delta = max(delta, np.max(np.abs(V_new-V_old)))
                    if delta < epsilon:
                        break
                    V_old = V_new
                """
                    update policy:
                    policy_new = argmax_a (Q(s,a))
                """
                policy_best = np.argmax(Q, axis=3)
                policy_new = np.zeros((self.fp_shape[0],self.fp_shape[1], n_action, n_action), dtype=int)
                policy_new[all_idx(policy_best, axis=3)] = 1 
                
                if not np.array_equal(policy_new, policy):
                    policy = policy_new
                else:
                    best_policy.append(policy_new)
                    # V.append(V_new)
                    flag = False
                    break
            if flag:
                best_policy.append(policy_new)
        self.best_policy = best_policy
        return 

    def follow_best_policy(self, start_point, start_action, AreaInt, AreaInt_idx):
        policy = self.best_policy[AreaInt_idx]
        goal = [AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx]]
        a_ini = self.action_space.index(start_action)
        s_ini = start_point

        path_best = [(s_ini,a_ini,a_ini)]
        a = a_ini
        s = s_ini
        over_count = 500
        while True:
            a_next = np.argmax(policy[s[0],s[1],a,:])
            action = self.action_space[a_next]
            x = max(0,min(self.fp_shape[0]-2, s[0]+action[0]))
            y = max(0,min(self.fp_shape[1]-2, s[1]+action[1]))
            s_next = [x,y]
            path_best.append((s_next, a, a_next))
            
            if s_next == goal:
                return path_best
            
            a = a_next
            s = s_next

            over_count -= 1
            if over_count<=0:
                break
        return path_best

    def get_best_path(self, s_init):
        
        # define necessary varbles and basic functions
        path_parent = np.empty((self.fp_shape[0], self.fp_shape[1]),dtype=object)
        cost = -np.sum(self.R, axis=(2,3))
        min_cost = np.ones(self.fp_shape) * np.inf
        edge_grid = []
        u_set = [[1,1],[0,1],[-1,1],[1,0],[-1,0],[1,-1],[0,-1],[-1,-1]]

        def get_neighbor(s, u_set, x_range, y_range):
            """
            find neighbors of state s that not in obstacles.
            :param s: state
            :return: neighbors
            """
            output = []
            for u in u_set:
                x = s[0] + u[0]
                y = s[1] + u[1]
                if 0 <= x < x_range and 0 <= y < y_range:
                    output.append((x,y))
            return output

        # initialization
        for i in s_init:
            path_parent[i] = i
            min_cost[i] = cost[i]
            heapq.heappush(edge_grid, (min_cost[i], i))


        while edge_grid:
            _, s = heapq.heappop(edge_grid)

            for s_n in get_neighbor(s, u_set, self.fp_shape[0], self.fp_shape[1]):
                new_cost = min_cost[s] + cost[s_n]
                
                if new_cost < min_cost[s_n]:
                    min_cost[s_n] = new_cost
                    path_parent[s_n] = s
                    heapq.heappush(edge_grid, (new_cost, s_n))
        return path_parent

    def get_prob_from_path_map(self, path_parent, point, s_init):
        s = point
        cur_action = [0,0]
        pre_action = [0,0]
        prob = 1
        while s not in s_init:
            cur_action = [path_parent[s][0] - s[0], path_parent[s][1] - s[1]]
            a_cur = self.action_space.index(cur_action)
            a_pre = self.action_space.index(pre_action)
            prob = prob * np.exp(self.R[s[0],s[1],a_pre,a_cur])
            s = path_parent[s]
            pre_action = cur_action
        return prob
            
    def best_path_probability_for_all_goals_all_startings(self, AreaInt):
        prob_map = np.zeros((self.fp_shape[0], self.fp_shape[1], len(AreaInt[0])))
        for x_i in range(self.fp_shape[0]):
            for y_i in range(self.fp_shape[1]):
                start_point = (x_i,y_i)
                start_action = [0,0]
                for AreaInt_indx in range(len(AreaInt[0])):
                    best_path = self.follow_best_policy(start_point, start_action, AreaInt, AreaInt_indx)
                    sum_reward_start = 0
                    for i in range(len(best_path)):
                        s0 = best_path[i][0][0]
                        s1 = best_path[i][0][1]
                        a_pre = best_path[i][1]
                        a_cur = best_path[i][2]
                        r = self.R[s0,s1,a_pre,a_cur]
                        sum_reward_start += r
                    prob_map[x_i,y_i,AreaInt_indx] = np.exp(sum_reward_start)
        return prob_map

    def prob_goal(self, start_point, current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, p_goal):
        best_path_start = self.follow_best_policy(start_point, [0,0], AreaInt, AreaInt_idx)
        best_path_cur = self.follow_best_policy(current_piont, cur_action, AreaInt, AreaInt_idx)
        sum_reward_start = 0
        for i in range(len(best_path_start)):
            s0 = best_path_start[i][0][0]
            s1 = best_path_start[i][0][1]
            a_pre = best_path_start[i][1]
            a_cur = best_path_start[i][2]
            r = self.R[s0,s1,a_pre,a_cur]
            sum_reward_start += r

        sum_reward_cur = 0
        for i in range(len(best_path_cur)):
            s0 = best_path_cur[i][0][0]
            s1 = best_path_cur[i][0][1]
            a_pre = best_path_cur[i][1]
            a_cur = best_path_cur[i][2]
            r = self.R[s0,s1,a_pre,a_cur]
            sum_reward_cur += r
        
        sum_reward_walked = 0
        for i in range(len(walked_path)):
            s0 = walked_path[i][0][0]
            s1 = walked_path[i][0][1]
            a_pre = self.action_space.index(walked_path[i][1])
            a_cur = self.action_space.index(walked_path[i][2])
            r = self.R[s0,s1,a_pre,a_cur]
            sum_reward_walked += r

        p_cond = np.exp(sum_reward_cur)*np.exp(sum_reward_walked)/np.exp(sum_reward_start)
                
        p_goal = p_cond*p_goal
        return p_goal

    def sampled_forward_predict(self, s_init, a_init, path_length, AreaInt):
        # feature_map = self.features_get.get_feature_map()
        # [[1,1],[0,1],[-1,1],[1,0],[0,0],[-1,0],[1,-1],[0,-1],[-1,-1]]
        n_action = len(self.action_space)
        i_action_init = self.action_space.index(a_init)

        D_init =  np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D_init[s_init[0], s_init[1], i_action_init] = 1

        D_sum = np.zeros((self.fp_shape[0],self.fp_shape[1],len(self.action_space)))
        D = D_init
        path_sampled = []
        for iter in range(path_length):
            D_s_a = self.pi * D.reshape((D.shape[0], D.shape[1], D.shape[2], 1))
            D_with_act = self.get_shift_D(D_s_a)
            D = D_with_act
            space_freq = np.sum(D, axis = 2)
            p_s = np.unravel_index(np.argmax(space_freq), space_freq.shape)
            path_sampled.append(p_s)
        return path_sampled

    def get_G(self, c, rad):
        fp_grid_cover = copy.deepcopy(self.fp)
        fp_grid_cover[fp_grid_cover > 2] = 0

        x_range = fp_grid_cover.shape[0]
        y_range = fp_grid_cover.shape[1]
        
        x_i = c[0]
        x_j = c[1]
        G = -np.ones((x_range,y_range)).astype(np.int8)
        if fp_grid_cover[x_i,x_j] != 0:
            return G
        for quad in range(4):
            if quad == 0:
                sign_i = 1
                sign_j = 1
            elif quad == 1:
                sign_i = -1
                sign_j = 1
            elif quad == 2:
                sign_i = -1
                sign_j = -1
            else:
                sign_i = 1
                sign_j = -1
            
        # sign_i = -1
        # sign_j = 1
            obs_slop = []
            for i in range(rad):
                for j in range(rad):
                    # if i == 4 and j == 2:
                    #     aaa = 0
                        
                    sc_grid_i = x_i + i*sign_i
                    sc_grid_j = x_j + j*sign_j
                    
                    if sc_grid_i<0 or sc_grid_i>=x_range:
                        continue
                    if sc_grid_j<0 or sc_grid_j>=y_range:
                        continue
                    
                    if fp_grid_cover[sc_grid_i,sc_grid_j] != 0: 
                        # if this grid is obstacle
                        # set G
                        G[sc_grid_i,sc_grid_j] = 0
                        
                        # update shaddow
                        obs_r = np.sqrt(i**2 + j**2)
                        if i != 0:
                            obs_a = np.arctan(j/i)
                        else:
                            obs_a = np.pi/2
                        obs_slop.append((obs_r,obs_a))
                    else: 
                        # if this gird is not obstacle
                        
                        # compute the polar coordinate of this grid
                        g_r = np.sqrt(i**2 + j**2)
                        if i!=0:
                            g_a = np.arctan(j/i)
                        else:
                            g_a = np.pi/2
                        # set G[]=1 (can be detected) first then kick off the shaddows.
                        G[sc_grid_i,sc_grid_j] = 1
                        # whether this grid in in shaddow
                        for obs_i in obs_slop:
                            if g_r > obs_i[0] and obs_i[1]-0.9/obs_i[0] <= g_a <= obs_i[1] + 0.9/obs_i[0]:
                                # if it is in shaddow, set G[] = 0
                                G[sc_grid_i,sc_grid_j] = 0
                                break
                    
       
        G[G == -1] = 0
        return G

    def get_area_perimeter_idx(self, c, rad):
        G = self.get_G(c, rad)

        perimeter = []
        a = np.where(G==1)
        area = [(a[0][i], a[1][i]) for i in range(len(a[0]))]
        for i in area:
            if (i[0]-1, i[1]-1) in area and \
               (i[0]-1, i[1]+1) in area and \
               (i[0]+1, i[1]-1) in area and \
               (i[0]+1, i[1]+1) in area:
                continue
            else:
                perimeter.append(i)
        return area, perimeter
        
    def sum_area(self, A, c, rad):
        output = 0
        area,_ = self.get_area_perimeter_idx(c, rad)
        for i in area:
            output += A[i]
        # b_l = max(0, c[1]-rad)
        # b_r = min(self.fp_shape[1], c[1]+rad)
        # b_t = max(0, c[0]-rad)
        # b_b = min(self.fp_shape[0], c[0]+rad)
        
        # for i in range(b_t,b_b):
        #     for j in range(b_l,b_r):
        #         output += A[i,j]
        return output
    
    def sum_perimeter_index(self, c, rad):
        idx_list = []
        for j in range(c[1]-rad,c[1]+rad):
            idx_x = c[0]-rad
            idx_y = j
            if idx_x < 0 or idx_x >= self.fp_shape[0] or idx_y <0 or idx_y >= self.fp_shape[1]:
                continue
            idx_list.append((idx_x, idx_y))

        for i in range(c[0]-rad+1, c[0]+rad):
            idx_x = i
            idx_y = c[1]+rad-1
            if idx_x < 0 or idx_x >= self.fp_shape[0] or idx_y <0 or idx_y >= self.fp_shape[1]:
                continue
            idx_list.append((idx_x, idx_y))            
            # idx_list.append((i, c[1]+rad-1))
            
        for j in range(c[1]+rad-1, c[1]-rad-1, -1):
            idx_x = c[0]+rad-1
            idx_y = j
            if idx_x < 0 or idx_x >= self.fp_shape[0] or idx_y <0 or idx_y >= self.fp_shape[1]:
                continue
            idx_list.append((idx_x, idx_y))
            # idx_list.append((c[0]+rad-1, j))

        for i in range(c[0]+rad-1, c[0]-rad, -1):
            idx_x = i
            idx_y = c[1]-rad
            if idx_x < 0 or idx_x >= self.fp_shape[0] or idx_y <0 or idx_y >= self.fp_shape[1]:
                continue
            idx_list.append((idx_x, idx_y))
        return idx_list
# try new backward with updated destination probability

