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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

with open('./result/theta.pickle', 'rb') as f:
    theta = pickle.load(f)

irl = IRL(fp)
irl.theta = theta
AreaInt = [[],[]]
for i,s in enumerate(s_goal_set):
    AreaInt[0].append(s[0])
    AreaInt[1].append(s[1])
irl.backward_pass(AreaInt, 500)
print('finish bp')

#%%
def get_path_from_path_map(path_parent, point, s_init):
    s = copy.copy(point)
    path = []
    while s not in s_init:
        path.append(s)
        s = path_parent[s]
    return path
with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)

Oran = cm.get_cmap('Oranges_r', 128)
Blue = cm.get_cmap('Blues', 128)


color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204], -1:[155,155,155]}
fp_shape = fp.shape
fp_show = np.zeros((fp_shape[0], fp_shape[1], 3))
for i in range(fp_shape[0]):
    for j in range(fp_shape[1]):
        fp_show[i,j] = color_map[fp[i,j]]
path_color = [0,128/255,0]
path_detected_color = [0,0,128/255]


walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
count = 0
path_parent_list = []
for aint_idx in range(len(AreaInt[0])):
    path_parent = irl.get_best_path([(AreaInt[0][aint_idx], AreaInt[1][aint_idx])])
    path_parent_list.append(path_parent)

for path in path_train:
    walked_path = []
    for i in range(len(path)):
        fp_show_save = copy.deepcopy(fp_show/255)
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        
        walked_path.append((current_piont, a_pre, cur_action))
        for AreaInt_idx in range(len(AreaInt[0])):
            prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
            prob_areaint[AreaInt_idx] = prob
        prob_areaint = prob_areaint/np.sum(prob_areaint)

        for AreaInt_idx in range(len(AreaInt[0])):
            path_parent = path_parent_list[AreaInt_idx]
            path_future = get_path_from_path_map(path_parent, path[i], [(AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx])])

            color = Oran(int(1/prob_areaint[AreaInt_idx]))
            for i_future in path_future:
                fp_show_save[i_future] = np.array(color[:3])
        for i_pre in range(i):
            fp_show_save[path[i_pre][0],path[i_pre][1]] = path_color
        for i_next in range(i+1,len(path)):
            fp_show_save[path[i_next][0],path[i_next][1]] = path_detected_color
        
        a_pre = cur_action
        count += 1
        plt.imshow(fp_show_save)
        plt.savefig('./result/predict_pictures6/heatmap%05d.png'%count)
        # plt.imshow(fp_show_save)
        # plt.show()
        # plot results.
        # 1. plot fp
        # 2. plot path
        # 3. plot heatmaps
# %% full ob hausdorff distance
from scipy.spatial.distance import directed_hausdorff
def get_path_from_path_map(path_parent, point, s_init):
    s = copy.copy(point)
    path = []
    while s not in s_init:
        path.append(s)
        s = path_parent[s]
    return path
with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)


walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
count = 0
path_parent_list = []
for aint_idx in range(len(AreaInt[0])):
    path_parent = irl.get_best_path([(AreaInt[0][aint_idx], AreaInt[1][aint_idx])])
    path_parent_list.append(path_parent)

result = []
for path in tqdm(path_train):
    walked_path = []
    for i in range(len(path)):
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        gt_path_future = []
        for i_next in range(i+1,len(path)):
            gt_path_future.append(path[i_next])
        B = np.array(gt_path_future)

        walked_path.append((current_piont, a_pre, cur_action))
        for AreaInt_idx in range(len(AreaInt[0])):
            prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
            prob_areaint[AreaInt_idx] = prob
        prob_areaint = prob_areaint/np.sum(prob_areaint)
        dist = 0
        for AreaInt_idx in range(len(AreaInt[0])):
            path_parent = path_parent_list[AreaInt_idx]
            path_future = get_path_from_path_map(path_parent, path[i], [(AreaInt[0][AreaInt_idx], AreaInt[1][AreaInt_idx])])
            A = np.array(path_future)
            if A.shape[0] != 0 and B.shape[0]!=0:
                dist += prob_areaint[AreaInt_idx] * max(directed_hausdorff(A,B)[0],directed_hausdorff(B,A)[0])
        result.append([i,dist])
        a_pre = cur_action
        count += 1
print(result)
with open('./result/result_hausdorff_distance_Mean.pickle', 'wb') as f:
    pickle.dump(result, f)
#%%
from scipy.spatial.distance import directed_hausdorff
def get_path_from_path_map(path_parent, point, s_init):
    s = copy.copy(point)
    path = []
    while s not in s_init:
        path.append(s)
        s = path_parent[s]
    return path
with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)


walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
count = 0
path_parent_list = []
for aint_idx in range(len(AreaInt[0])):
    path_parent = irl.get_best_path([(AreaInt[0][aint_idx], AreaInt[1][aint_idx])])
    path_parent_list.append(path_parent)

result = []
for path in tqdm(path_train):
    walked_path = []
    for i in range(len(path)):
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        gt_path_future = []
        for i_next in range(i+1,len(path)):
            gt_path_future.append(path[i_next])
        B = np.array(gt_path_future)

        walked_path.append((current_piont, a_pre, cur_action))
        for AreaInt_idx in range(len(AreaInt[0])):
            prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
            prob_areaint[AreaInt_idx] = prob
        prob_areaint = prob_areaint/np.sum(prob_areaint)
        
        max_areaint_idx = np.argmax(prob_areaint)
        
        path_parent = path_parent_list[max_areaint_idx]
        path_future = get_path_from_path_map(path_parent, path[i], [(AreaInt[0][max_areaint_idx], AreaInt[1][max_areaint_idx])])
        A = np.array(path_future)
        if A.shape[0] != 0 and B.shape[0]!=0:
            dist = max(directed_hausdorff(A,B)[0],directed_hausdorff(B,A)[0])
        result.append([i,len(path)-i,dist])
        a_pre = cur_action
        count += 1
print(len(result))
with open('./result/result_hausdorff_distance_ML.pickle', 'wb') as f:
    pickle.dump(result, f)
# %%
with open('./result/result_hausdorff_distance_ML.pickle', 'rb') as f:
    result = pickle.load(f)
max_dis = 0
for i in result:
    if i[0]>max_dis:
        max_dis = i[0]
md = np.zeros(max_dis)
count = np.ones(max_dis)
for i in range(1,max_dis-1):
    for j in result:
        if j[1] == i:
            md[i] += j[2]
            count[i] += 1
md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('EMHD')
plt.axis('on')
# %%

with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)


walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
count = 0
path_parent_list = []
for aint_idx in range(len(AreaInt[0])):
    path_parent = irl.get_best_path([(AreaInt[0][aint_idx], AreaInt[1][aint_idx])])
    path_parent_list.append(path_parent)

result = []
prob_areaint_list = []
for path in tqdm(path_train):
    walked_path = []
    for i in range(len(path)):
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]


        walked_path.append((current_piont, a_pre, cur_action))
        for AreaInt_idx in range(len(AreaInt[0])):
            prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
            prob_areaint[AreaInt_idx] = prob
        prob_areaint = prob_areaint/np.sum(prob_areaint)
        prob_areaint_list.append([i,len(path)-i,copy.copy(prob_areaint)])
        if AreaInt[0][np.argmax(prob_areaint)] == path[-1][0] and  AreaInt[1][np.argmax(prob_areaint)] == path[-1][1]:
            result.append([i,len(path)-i,1])
        else:
            result.append([i,len(path)-i,0])
        a_pre = cur_action
        count += 1
with open('./result/result_prob_areaint.pickle', 'wb') as f:
    pickle.dump(prob_areaint_list, f)
#%%
    
max_dis = 0
for i in result:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis)
count = np.ones(max_dis)
for i in range(1,max_dis-1):
    for j in result:
        if j[0] == i:
            md[i] += j[2]
            count[i] += 1

md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('top-1 accuracy')
plt.axis('on')
# %%
with open('./result/result_prob_areaint.pickle', 'rb') as f:
    prob_areaint_list = pickle.load(f)

max_dis = 0
for i in prob_areaint_list:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        length = prob_areaint_list[idx][0]
        p = copy.copy(prob_areaint_list[idx][2])
        for top_k in range(5):
            if AreaInt[0][np.argmax(p)] == path[-1][0] and AreaInt[1][np.argmax(p)] == path[-1][1]:
                md[length] += 1
                break
            else:
                p[np.argmax(p)] = -1
        count[length] += 1
        idx += 1
md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('top-5 accuracy')
plt.axis('on')

# %%
with open('./result/result_prob_areaint.pickle', 'rb') as f:
    prob_areaint_list = pickle.load(f)

max_dis = 0
for i in prob_areaint_list:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        length = prob_areaint_list[idx][0]
        p = copy.copy(prob_areaint_list[idx][2])
        x = np.array(p) @ np.array(AreaInt[0])
        y = np.array(p) @ np.array(AreaInt[1]) 
        dist_sq = (x-path[-1][0])**2 + (y-path[-1][1])**2
        dist = np.sqrt(dist_sq)
        md[length] += dist
        count[length] += 1
        idx += 1
md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('mean distance')
plt.axis('on')

# %%
with open('./result/result_prob_areaint.pickle', 'rb') as f:
    prob_areaint_list = pickle.load(f)

max_dis = 0
for i in prob_areaint_list:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        length = prob_areaint_list[idx][0]
        p = copy.copy(prob_areaint_list[idx][2])
        x = AreaInt[0][np.argmax(p)]
        y = AreaInt[1][np.argmax(p)]
        dist_sq = (x-path[-1][0])**2 + (y-path[-1][1])**2
        dist = np.sqrt(dist_sq)
        md[length] += dist
        count[length] += 1
        idx += 1
md = md/len(path_train)
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('ML distance')
plt.axis('on')
#%%


from scipy.spatial.distance import directed_hausdorff
def get_path_from_path_map(path_parent, point, s_init):
    s = copy.copy(point)
    path = []
    while s not in s_init:
        path.append(s)
        s = path_parent[s]
    return path
with open('./result/best_policy.pickle', 'rb') as f:
    irl.best_policy = pickle.load(f)
with open('./result/sp_original.pickle', 'rb') as f:
    sensor_placement = pickle.load(f)

walked_path = []
a_pre = [0,0]
prob_areaint = np.zeros(len(AreaInt[0]))
count = 0
path_parent_list = []
for aint_idx in range(len(AreaInt[0])):
    path_parent = irl.get_best_path([(AreaInt[0][aint_idx], AreaInt[1][aint_idx])])
    path_parent_list.append(path_parent)

result = []
prob_areaint_list = []
rad = 3
for path in tqdm(path_train):
    walked_path = []
    seen_len = 0
    for i in range(len(path)):
        current_piont = path[i]
        flag_seen = False
        if i == 0:
            cur_action = [0,0]
        else:
            cur_action = [path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]]
        gt_path_future = []
        for i_next in range(i+1,len(path)):
            gt_path_future.append(path[i_next])
        B = np.array(gt_path_future)
        
        
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                seen_len += 1
                walked_path.append((current_piont, a_pre, cur_action))
                for AreaInt_idx in range(len(AreaInt[0])):
                    prob = irl.prob_goal(walked_path[0][0], current_piont, cur_action, walked_path, AreaInt, AreaInt_idx, 1)
                    prob_areaint[AreaInt_idx] = prob
                prob_areaint = prob_areaint/np.sum(prob_areaint)
                prob_areaint_list.append([seen_len,len(path)-i,copy.copy(prob_areaint)])
                max_areaint_idx = np.argmax(prob_areaint)
        
                path_parent = path_parent_list[max_areaint_idx]
                path_future = get_path_from_path_map(path_parent, path[i], [(AreaInt[0][max_areaint_idx], AreaInt[1][max_areaint_idx])])
                A = np.array(path_future)
                if A.shape[0] != 0 and B.shape[0]!=0:
                    dist = max(directed_hausdorff(A,B)[0],directed_hausdorff(B,A)[0])
                result.append([seen_len,len(path)-i,dist])
        if not flag_seen:
            walked_path = []
        a_pre = cur_action
        count += 1
print(len(result))
with open('./result/result_partial_ob_hausdorff_distance_ML.pickle', 'wb') as f:
    pickle.dump(result, f)

with open('./result/result_partial_ob_prob_areaint.pickle', 'wb') as f:
    pickle.dump(prob_areaint_list, f)
# %%
md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                length = prob_areaint_list[idx][0]
                p = copy.copy(prob_areaint_list[idx][2])
                for top_k in range(1):
                    if AreaInt[0][np.argmax(p)] == path[-1][0] and AreaInt[1][np.argmax(p)] == path[-1][1]:
                        md[length] += 1
                        break
                    else:
                        p[np.argmax(p)] = -1
                count[length] += 1
                idx += 1
md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('top-5 accuracy')
plt.axis('on')
# %%
with open('./result/result_partial_ob_prob_areaint.pickle', 'rb') as f:
    prob_areaint_list = pickle.load(f)


max_dis = 0
for i in prob_areaint_list:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                length = prob_areaint_list[idx][0]
                p = copy.copy(prob_areaint_list[idx][2])
                x = np.array(p) @ np.array(AreaInt[0])
                y = np.array(p) @ np.array(AreaInt[1]) 
                dist_sq = (x-path[-1][0])**2 + (y-path[-1][1])**2
                dist = np.sqrt(dist_sq)
                md[length] += dist
                count[length] += 1
                idx += 1
md = md/count
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('mean distance')
plt.axis('on')
# %%
max_dis = 0
for i in prob_areaint_list:
    if i[0]>max_dis:
        max_dis = i[0]

md = np.zeros(max_dis+1)
count = np.ones(max_dis+1)
idx = 0
for path in tqdm(path_train):
    for i in range(len(path)):
        for s_i in sensor_placement[:20]:
            if abs(path[i][0]-s_i[0])<rad and abs(path[i][1]-s_i[1])<rad:
                flag_seen = True
                length = prob_areaint_list[idx][0]
                p = copy.copy(prob_areaint_list[idx][2])
                x = AreaInt[0][np.argmax(p)]
                y = AreaInt[1][np.argmax(p)]
                dist_sq = (x-path[-1][0])**2 + (y-path[-1][1])**2
                dist = np.sqrt(dist_sq)
                md[length] += dist
                count[length] += 1
                idx += 1
md = md/len(path_train)
plt.style.use('default')
plt.plot(np.arange(max_dis+1)[1:-1], md[1:-1])
plt.xlabel('observed path length')
plt.ylabel('ML distance')
plt.axis('on')
# %%

