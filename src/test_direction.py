#%%
from IRL_with_action import IRL
from feature_abstraction import f_abs_with_direction
from common.config import argparser
from common.engine import get_fp, A_star_simulation
import os,pickle
import numpy as np
import matplotlib.pyplot as plt
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

s_goal_list = [i[-1] for i in path_train]
s_goal_set = set()
for s in s_goal_list:
    if s not in s_goal_set:
        s_goal_set.add(s)

# %% test IRL.get_mean_features
FA = f_abs_with_direction(fp)
irl = IRL(FA)

# D = irl.get_mean_features(path_train)

# %%
print('test_bp')
AreaInt = [[],[]]
for s in s_goal_set:
    AreaInt[0].append(s[0])
    AreaInt[1].append(s[1])
irl.backward_pass(AreaInt, 100)

# %%
print('test_fp')
feature_mean = irl.forward_pass(AreaInt,100)
# %%
print('test_gradiant')
irl.gradiant_theta(AreaInt, 500, path_train, 0.01)
# %%
s_init = path_train[0][20]
heat_map = irl.pred_avg_heatmap(s_init,100)

# %%
import copy

# heat_map = heat_map/np.sum(heat_map)
# log_hm = np.log(heat_map+0.00001)
# min_lhm = np.min(log_hm)
# max_lhm = np.max(log_hm)
# log_hm_norm = (log_hm-min_lhm)/(max_lhm-min_lhm)

def get_fp_path(fp, path):
    fp_copy = copy.deepcopy(fp)
    for i in path:
        fp[i] = 10
    return fp_copy
# fp_path = get_fp_path(fp, path_train[0])
# plt.imshow(10*log_hm_norm+fp_path)

def plot_hm_fp_path(heat_map, path):
    heat_map = heat_map/np.sum(heat_map)
    log_hm = np.log(heat_map+0.00001)
    min_lhm = np.min(log_hm)
    max_lhm = np.max(log_hm)
    log_hm_norm = (log_hm-min_lhm)/(max_lhm-min_lhm)
    fp_path = get_fp_path(fp, path_train[0])
    return 10*log_hm_norm+fp_path
for i in range(1,len(path_train[0]),20):
    s_init = path_train[0][i]
    heat_map = irl.pred_avg_heatmap(s_init,250)
    hm_fp_path = plot_hm_fp_path(heat_map, path_train[0])
    plt.imshow(hm_fp_path)
    plt.show()
# %%
