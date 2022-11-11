#%%
from IRL import IRL
from feature_abstraction import f_abs
from common.config import argparser
from common.engine import get_fp, A_star_simulation
import os,pickle

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
FA = f_abs(fp)
irl = IRL(FA)

# D = irl.get_mean_features(path_train)

# %%
AreaInt = [[],[]]
for s in s_goal_set:
    AreaInt[0].append(s[0])
    AreaInt[1].append(s[1])
irl.backward_pass(AreaInt, 100)
# %%
feature_mean = irl.forward_pass(AreaInt,100)
# %%
irl.gradiant_theta(AreaInt, 500, path_train, 0.01)