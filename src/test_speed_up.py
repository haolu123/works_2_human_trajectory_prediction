#%%
from IRL_with_motion_speed_up import IRL
from feature_abstraction import f_abs_with_direction
from common.config import argparser
from common.engine import get_fp, A_star_simulation
import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns; sns.set()
from tqdm import tqdm
#%% test fp, path_train and s_goal
grid_width = 10
baseColor_fn = 'baseColor.png'
fp_zone_fn = 'fp2_zone.png'
fp_code_fn = 'fp2_code.png'

tempdir  = './temp_result/'
with open(tempdir + 'fp_grid.pickle', 'rb') as f:
    fp = pickle.load(f)

with open(tempdir + 'path_all.pickle', 'rb') as f:
    path_train = pickle.load(f)
for i in range(len(path_train)-1,-1,-1):
    if len(path_train[i]) == 0:
        path_train.pop(i)
s_goal_list = [i[-1] for i in path_train]
s_goal_set = set()
for s in s_goal_list:
    if s not in s_goal_set:
        s_goal_set.add(s)

#%% test IRL.get_mean_features

irl = IRL(fp)

# %%
print('test_bp')
AreaInt = [[],[]]
for s in s_goal_set:
    AreaInt[0].append(s[0])
    AreaInt[1].append(s[1])
# irl.backward_pass(AreaInt, 500)

# # %%
# print('test_fp')
# feature_mean = irl.forward_pass(AreaInt,500)

# %%
print('test_gradiant')
irl.gradiant_theta(AreaInt, 500, path_train, [0.005,1000,0.005,0.1,0.1], max_loop=50, epsil=0.0001)
a = np.sum(irl.R,axis=(2,3))
a = (a-np.min(a))/(np.max(a)-np.min(a))
plt.imshow(-np.log(a))


# # %% 
# import seaborn as sns; sns.set()
# from tqdm import tqdm
# """
# 1. plot floor plan
# 2. plot path on the top of floor plan
# 3. plot current position on the top of floor plan
# 4. plot the heatmap on the floor plan
# 5. make it to video
# """
# # floor plan plot
# color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
# fp_shape = fp.shape
# fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
# for i in range(fp_shape[0]):
#     for j in range(fp_shape[1]):
#         fp_show[i,j] = color_map[fp[i,j]]


# # add paths and position
# path_color = [128,0,0]
# fp_show_copy = copy.deepcopy(fp_show)

# #path = path_train[path_num]
# count = 0
# for path in tqdm(path_train[::50]):
#     fp_show_copy = copy.deepcopy(fp_show)
#     for i in path:
#         fp_show_copy[i] = path_color

#     fp_path_copy = copy.deepcopy(fp_show_copy)
#     # plt.imshow(fp_path_copy)
#     # plt.grid(False)
#     # plt.axis('off')
#     # plt.show()

#     for i in range(len(path)):
#         s_init = path[i]
#         if i == 0:
#             a_init = [0,0]
#         else:
#             a_init = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
#         heatmap_data = irl.pred_avg_heatmap(s_init,a_init,50,AreaInt)
#         heatmap_data = heatmap_data/np.sum(heatmap_data)
#         heatmap_data = np.arctan(heatmap_data)*2/np.pi
#         heatmap_data = heatmap_data*255
#         heatmap_data = heatmap_data.astype('uint8')
#         fp_path_copy[s_init] = [0,255,255]


#     # plot heatmap on the floor plan

#         hmax = sns.heatmap( heatmap_data,
#                                 cmap  = 'Greens',
#                                 alpha = 0.7,
#                                 cbar = False,
#                                 )
#         hmax.imshow(fp_path_copy,
#                 aspect = hmax.get_aspect(),
#                 extent = hmax.get_xlim() + hmax.get_ylim(),
#         ) 
#         plt.axis('off')
#         count += 1
#         plt.savefig('./result/predict_pictures2/heatmap%05d.png'%count)
#         plt.close()
#         #put the map under the heatmap

# #%%
# with open('./result/theta.pickle', 'wb') as f:
#     pickle.dump(irl.theta, f)
# # %%
# with open('./result/theta.pickle', 'rb') as f:
#     theta = pickle.load(f)

# irl = IRL(fp)
# irl.theta = theta
# AreaInt = [[],[]]
# for s in s_goal_set:
#     AreaInt[0].append(s[0])
#     AreaInt[1].append(s[1])
# irl.backward_pass(AreaInt, 500)

# max_step = 20
# matric_dist = irl.get_metric(path_train, max_step)
# plt.plot(matric_dist)
# # %%
# path = path_train[4]
# color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
# fp_shape = fp.shape
# fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
# for i in range(fp_shape[0]):
#     for j in range(fp_shape[1]):
#         fp_show[i,j] = color_map[fp[i,j]]
# path_color = [128,0,0]
# # for i in path:
# #     fp_show[i] = path_color

# # plt.imshow(fp_show)
# # plt.show()

# for i in range(len(path)):
#     s_init = path[i]
#     if i == 0:
#         a_init = [0,0]
#     else:
#         a_init = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
#     path_sampled = irl.sampled_forward_predict(s_init,a_init,10,AreaInt)
#     heatmap_data = np.zeros(fp_shape)
#     fp_show_path = copy.deepcopy(fp_show)
#     for i_sample in range(len(path_sampled)):
#         heatmap_data[path_sampled[i_sample]] = 1
#         fp_show_path[path[min(i+i_sample, len(path)-1)]] = path_color
#     heatmap_data = heatmap_data*255
#     heatmap_data = heatmap_data.astype('uint8')
    
    
#     hmax = sns.heatmap( heatmap_data,
#                             cmap  = 'Greens',
#                             alpha = 0.7,
#                             cbar = False,
#                             )
#     hmax.imshow(fp_show_path,
#             aspect = hmax.get_aspect(),
#             extent = hmax.get_xlim() + hmax.get_ylim(),
#     ) 
#     plt.axis('off')
#     plt.show()
#     # count += 1
#     # plt.savefig('./result/predict_pictures2/heatmap%05d.png'%count)
#     # plt.close()
# %% test best policy
with open('./result/theta.pickle', 'rb') as f:
    theta = pickle.load(f)

irl = IRL(fp)
irl.theta = theta
AreaInt = [[],[]]
for i,s in enumerate(s_goal_set):
    AreaInt[0].append(s[0])
    AreaInt[1].append(s[1])
irl.backward_pass(AreaInt, 50)
print('finish bp')
a = np.sum(irl.V,axis=2)
a = (a-np.min(a))/(np.max(a)-np.min(a))
plt.imshow(a)
#%%
# irl.best_policy_given_destination(AreaInt, discount_factor=1, epsilon=0.1)
# with open('./result/best_policy.pickle', 'wb') as f:
#     pickle.dump(irl.best_policy, f)
#%%
with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)
prob_map = irl.best_path_probability_for_all_goals_all_startings(AreaInt)
with open('./result/prob_map.pickle', 'wb') as f:
    pickle.dump(prob_map, f)
# #%%
# color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
# fp_shape = fp.shape
# fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
# for i in range(fp_shape[0]):
#     for j in range(fp_shape[1]):
#         fp_show[i,j] = color_map[fp[i,j]]

# for i in range(fp_shape[0]):
#     for j in range(fp_shape[1]):
#         fp_show_copy = copy.deepcopy(fp_show)
#         fp_show_copy[i,j] = [255,0,0]
#         goal_idx = np.argmax(prob_map[i,j,:])
#         goal = [AreaInt[0][goal_idx], AreaInt[1][goal_idx]]
#         fp_show_copy[goal[0], goal[1]] = [0,125,0]
#         plt.imshow(fp_show_copy)
#         plt.show()
#         # input()


# #%%
# path = path_train[4]
# for i in range(len(AreaInt[0])):
#     path_best = irl.follow_best_policy(path[4], [0, 0], AreaInt, i)
#     color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
#     fp_shape = fp.shape
#     fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
#     for i in range(fp_shape[0]):
#         for j in range(fp_shape[1]):
#             fp_show[i,j] = color_map[fp[i,j]]
#     path_color = [128,0,0]
#     for i in path_best:
#         fp_show[i[0][0],i[0][1]] = path_color

#     plt.imshow(fp_show)
#     plt.show()
# # %%

# # color map of the floor plan
# color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
# fp_shape = fp.shape
# fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
# for i in range(fp_shape[0]):
#     for j in range(fp_shape[1]):
#         fp_show[i,j] = color_map[fp[i,j]]

# path_color = [128,0,0]

# fp_show_save = copy.deepcopy(fp_show)
# probs = []
# count = 0
# for i in range(len(path_train)):
#     fp_show = copy.deepcopy(fp_show_save)
#     path = path_train[i]
#     start_point = path[0]
#     probabilities = np.zeros((len(AreaInt[0]), len(path)))
#     walked_path = []
#     a_pre = [0,0]
#     for i in range(len(path)):
        
#         current_piont = path[i]
#         if i == 0:
#             cur_action = [0,0]
#         else:
#             cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
#         walked_path.append((current_piont, a_pre, cur_action))
#         for AreaInt_idx in range(len(AreaInt[0])):
#             prob = irl.prob_goal(start_point, current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
#             probabilities[AreaInt_idx, i] = prob
#         a_pre = cur_action

#         #plot heatmap
#         fp_show[path[i][0],path[i][1]] = path_color
#         p = probabilities[:, i]/np.sum(probabilities[:, i])
#         heatmap_data = np.zeros(fp.shape)
#         for AreaInt_idx in range(len(AreaInt[0])):
#             heatmap_data[AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx]] = p[AreaInt_idx]
        
#         hmax = sns.heatmap( heatmap_data,
#                                 cmap  = 'Greens',
#                                 alpha = 0.7,
#                                 cbar = False,
#                                 )
#         hmax.imshow(fp_show,
#                 aspect = hmax.get_aspect(),
#                 extent = hmax.get_xlim() + hmax.get_ylim(),
#         ) 
#         plt.axis('off')
#         count += 1
#         plt.savefig('./result/predict_pictures3/heatmap%05d.png'%count)
#         plt.close()
#         # plt.show()

#     probabilities = probabilities/np.sum(probabilities, axis=0)
#     for i in range(len(AreaInt[0])):
#         plt.plot(probabilities[i,:])
#     plt.show()

#     probs.append(probabilities)
# %%
with open('./result/prob_map.pickle', 'rb') as f:
    prob_map = pickle.load(f)
I_map = np.zeros(fp.shape)
rad = 3
for c1 in range(fp.shape[0]):
    for c2 in range(fp.shape[1]):
        if fp[c1, c2] == 1 or fp[c1,c2] == 2:
            continue
        c = [c1,c2]
        R = np.exp(np.sum(irl.R, axis=(2,3))/81 )
        A = -R*np.log(R)
        term1 = irl.sum_area(A, c, rad)
        area, perimeter_idx = irl.get_area_perimeter_idx(c, rad)
        term2_1 = 0
        term3_1 = 0
        term4_1 = 0

        for i in perimeter_idx:
            for j in perimeter_idx:       
                if  fp[i] == 1 or fp[i] == 2 or  fp[j] == 1 or fp[j] == 2:
                    continue
                term2_1 += np.sum( (A[i] + A[j])*prob_map[j]/prob_map[i])
                term3_1 += np.sum( (R[i] + R[j])*prob_map[j]/prob_map[i]*np.log(prob_map[j]))
                term4_1 += np.sum( (R[i] + R[j])*prob_map[j]/prob_map[i]*np.log(prob_map[i]))
            
        term2_2 = irl.sum_area(A, c, rad-1)
        term3_2 = irl.sum_area(R, c, rad-1)
        term4_2 = irl.sum_area(R, c, rad-1)

        I = term1 + 1/len(AreaInt[0])*(term2_1*term2_2 + term3_1*term3_2 - term4_1*term4_2)
        I_map[c1,c2] = I
#%%
I_map_cut = copy.deepcopy(I_map[:51,:54])
I_map_cut[:10,:20] = 0
sensor_num  =19
sensor_placement = []
count = 0
while sensor_num>0:
    sensor_num -= 1
    sp = np.unravel_index(np.argmax(I_map_cut, axis=None), I_map_cut.shape)
    sensor_placement.append(sp)
    I_map_cut[max(0,sp[0]-(2*rad)):min(fp.shape[0], sp[0]+(2*rad)), max(0,sp[1]-(2*rad)):min(fp.shape[1], sp[1]+(2*rad))] = -1

    color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
    fp_shape = fp.shape
    fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
    for i in range(fp_shape[0]):
        for j in range(fp_shape[1]):
            fp_show[i,j] = color_map[fp[i,j]]
    path_color = [128,0,0]
    for i in sensor_placement:
        fp_show[i[0],i[1]] = path_color
    count += 1
    plt.imshow(fp_show)
    plt.savefig('./result/predict_pictures4/sp%02d.png'%count)
with open('./result/sp_original.pickle', 'wb') as f:
    pickle.dump(sensor_placement, f)
# %% cover rate
# covered_path = []
cr = []
for s_num in range(20):
    cp_num = 0
    for idx, path_i in enumerate(path_train):
        flag=False
        for i in path_i:
            for s_i in sensor_placement[:s_num]:
                if abs(i[0]-s_i[0])<rad and abs(i[1]-s_i[1])<rad:
                    # covered_path.append(path_i)
                    cp_num += 1
                    flag = True
                    break
            if flag:
                break
    cover_rate = cp_num/len(path_train)
    print(cover_rate)
    cr.append(cover_rate)
plt.plot(cr)

#%%
path_parent_list = []
for sensor_idx in range(len(sensor_placement)):
    path_parent = irl.get_best_path([sensor_placement[sensor_idx]])
    path_parent_list.append(path_parent)
# %% 
# may be possible matrix: 
#   1. arriving time
#   2. detecting destination

# 1. 先把sensor_placement 格式改成 AreaInt
# 2. 找到被检测到的路径，做动画
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

Oran = cm.get_cmap('Oranges_r', 128)
Blue = cm.get_cmap('Blues', 128)

color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
fp_shape = fp.shape
fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
for i in range(fp_shape[0]):
    for j in range(fp_shape[1]):
        fp_show[i,j] = color_map[fp[i,j]]
path_color = [128/255,0,0]
path_detected_color = [0,0,128/255]

sp_formed = [[],[]]
for i,s in enumerate(sensor_placement):
    sp_formed[0].append(s[0])
    sp_formed[1].append(s[1])

walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
prob_sensor = np.zeros(len(sensor_placement))
count = 0
for path in path_train:
    fp_show_save = copy.deepcopy(fp_show/255)
    for i in range(len(path)):
        fp_show_save[path[i][0],path[i][1]] = path_color
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                fp_show_save[path[i][0],path[i][1]] = path_detected_color

                walked_path.append((current_piont, a_pre, cur_action))
                for AreaInt_idx in range(len(AreaInt[0])):
                    prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
                    prob_areaint[AreaInt_idx] = prob
                prob_areaint = prob_areaint/np.sum(prob_areaint)

                for sensor_idx in range(len(sensor_placement)):
                    path_parent = path_parent_list[sensor_idx]
                    # print(sensor_idx)
                    prob_b = irl.get_prob_from_path_map(path_parent, current_piont, sensor_placement[sensor_idx])
                    prob_c = irl.get_prob_from_path_map(path_parent, walked_path[0][0], sensor_placement[sensor_idx])
                    prob_a = 1
                    for si in walked_path:
                        cur_a_n = irl.action_space.index(cur_action) 
                        pre_a_n = irl.action_space.index(a_pre)
                        prob_a = prob_a * irl.R[si[0][0],si[0][1],pre_a_n, cur_a_n]


                    prob_sensor[sensor_idx] = prob_a*prob_b/prob_c
                prob_sensor = prob_sensor/np.sum(prob_sensor)

                for AreaInt_idx in range(len(AreaInt[0])):
                    color = Oran(int(1/prob_areaint[AreaInt_idx]))
                    fp_show_save[AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx]] = np.array(color[:3])
                for sensor_idx in range(len(sensor_placement)):
                    color = Oran(int(1/prob_sensor[sensor_idx]*5))
                    fp_show_save[sensor_placement[sensor_idx][0], sensor_placement[sensor_idx][1]] = np.array(color[:3])
        if not flag_seen:
            walked_path = []
        a_pre = cur_action
        count += 1
        plt.imshow(fp_show_save)
        plt.savefig('./result/predict_pictures5/heatmap%05d.png'%count)
        # plt.imshow(fp_show_save)
        # plt.show()
        # plot results.
        # 1. plot fp
        # 2. plot path
        # 3. plot heatmaps
# %% 
# may be possible matrix: 
#   1. arriving time
#   2. detecting destination

# 1. 先把sensor_placement 格式改成 AreaInt
# 2. 找到被检测到的路径，做动画
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2

with open('./result/sp_original.pickle', 'rb') as f:
    sensor_placement = pickle.load(f)

def plot_circles(image, center, radius, color, alpha):
    overlay = image.copy()
    overlay = cv2.circle(overlay, center, radius, color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

Oran = cm.get_cmap('Oranges_r', 128)
Blue = cm.get_cmap('Blues', 128)
rad = 3
color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
fp_shape = fp.shape
fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
for i in range(fp_shape[0]):
    for j in range(fp_shape[1]):
        fp_show[i,j] = color_map[fp[i,j]]
path_color = [128/255,0,0]
path_detected_color = [0,0,128/255]

sp_formed = [[],[]]
for i,s in enumerate(sensor_placement):
    sp_formed[0].append(s[0])
    sp_formed[1].append(s[1])

walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0])) + 0.01
prob_sensor = np.zeros(len(sensor_placement))
count = 0
for path in path_train:
    fp_show_save = copy.deepcopy(fp_show/255)
    prob_areaint = np.zeros(len(AreaInt[0])) + 0.01
    prob_sensor = np.zeros(len(sensor_placement))
    walked_path = []
    for i in range(len(path)):
        fp_show_save[path[i][0],path[i][1]] = path_color
        # fp_resize = cv2.resize(fp_show_save, (fp.shape[0]*20,fp.shape[1]*20), interpolation = cv2.INTER_NEAREST)
        # for sensor_idx in range(len(sensor_placement)):
        #     color = Oran(1)
        #     # fp_show_save[sensor_placement[sensor_idx][0], sensor_placement[sensor_idx][1]] = np.array(color[:3])
        #     fp_resize = plot_circles(fp_resize, (sensor_placement[sensor_idx][1]*20,sensor_placement[sensor_idx][0]*20+40), int(200*prob_sensor[sensor_idx]), color, 0.5)
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                fp_show_save[path[i][0],path[i][1]] = path_detected_color

                walked_path.append((current_piont, a_pre, cur_action))
                for AreaInt_idx in range(len(AreaInt[0])):
                    prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
                    prob_areaint[AreaInt_idx] = prob
                prob_areaint = prob_areaint/np.sum(prob_areaint)

                for sensor_idx in range(len(sensor_placement)):
                    path_parent = path_parent_list[sensor_idx]
                    # print(sensor_idx)
                    prob_b = irl.get_prob_from_path_map(path_parent, current_piont, [sensor_placement[sensor_idx]])
                    prob_c = irl.get_prob_from_path_map(path_parent, walked_path[0][0], [sensor_placement[sensor_idx]])
                    prob_a = 1
                    for si in walked_path:
                        cur_a_n = irl.action_space.index(cur_action) 
                        pre_a_n = irl.action_space.index(a_pre)
                        prob_a = prob_a * irl.R[si[0][0],si[0][1],pre_a_n, cur_a_n]


                    prob_sensor[sensor_idx] = prob_a*prob_b/prob_c
                prob_sensor = prob_sensor/np.sum(prob_sensor)

        fp_resize = cv2.resize(fp_show_save, (fp.shape[0]*20,fp.shape[1]*20), interpolation = cv2.INTER_NEAREST)
        for sensor_idx in range(len(sensor_placement)):
            color = Oran(1)
            # fp_show_save[sensor_placement[sensor_idx][0], sensor_placement[sensor_idx][1]] = np.array(color[:3])
            fp_resize = plot_circles(fp_resize, (sensor_placement[sensor_idx][1]*20,sensor_placement[sensor_idx][0]*20+40), int(200*prob_sensor[sensor_idx]), color, 0.5)
        # if not flag_seen:
        #     walked_path = []
        a_pre = cur_action
        
        #plot heatmap
        p = prob_areaint
        heatmap_data = np.zeros(fp.shape)
        for AreaInt_idx in range(len(AreaInt[0])):
            heatmap_data[AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx]] = p[AreaInt_idx]
        heatmap_data = cv2.resize(heatmap_data, (fp.shape[0]*20,fp.shape[1]*20), interpolation = cv2.INTER_NEAREST)
        hmax = sns.heatmap( heatmap_data,
                                cmap  = 'Greens',
                                alpha = 0.5,
                                cbar = False,
                                )
        hmax.imshow(fp_resize,
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
        ) 
        plt.axis('off')
        count += 1
        plt.savefig('./result/predict_pictures7/heatmap%05d.png'%count)
        plt.close()
        # plt.show()

        # plt.imshow(fp_show_save)
        # plt.show()
        # plot results.
        # 1. plot fp
        # 2. plot path
        # 3. plot heatmaps
# %%
import math
import heapq

# define necessary varbles and basic functions
s_init = (10,10)
path_parent = np.empty((fp.shape[0],fp.shape[1]),dtype=object)
cost = -np.sum(irl.R, axis=(2,3))
min_cost = np.ones(fp.shape) * np.inf
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
path_parent[s_init] = s_init
min_cost[s_init] = cost[s_init]
heapq.heappush(edge_grid, (min_cost[s_init], s_init))

while edge_grid:
    _, s = heapq.heappop(edge_grid)

    for s_n in get_neighbor(s, u_set, fp.shape[0], fp.shape[1]):
        new_cost = min_cost[s] + cost[s_n]
        
        if new_cost < min_cost[s_n]:
            min_cost[s_n] = new_cost
            path_parent[s_n] = s
            heapq.heappush(edge_grid, (new_cost, s_n))


# %%
s_goal = (30,30)
path = [s_goal]
s = s_goal
while s!= s_init:
    s = path_parent[s]
    path.append(s)

color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
fp_shape = fp.shape
fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
for i in range(fp_shape[0]):
    for j in range(fp_shape[1]):
        fp_show[i,j] = color_map[fp[i,j]]
path_color = [128,0,0]
path_detected_color = [0,0,128]
for i in path:
    fp_show[i] = path_color
plt.imshow(fp_show)
