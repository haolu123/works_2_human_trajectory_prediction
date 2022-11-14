#%%
import numpy as np
from common.config import argparser
from common.engine import Get_BaseColor, Label_gridFloorPlan, get_fp
import os,pickle
import matplotlib.pyplot as plt
#%%

# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

class f_abs:

    
    def __init__(self, fp) -> None:
        self.Feature_dims = 7
        self.features = np.zeros(self.Feature_dims)
        self.fp = fp.astype('int')
        
            
    def get_features(self, s):
        '''
            feature map
            Args:
                s : (x,y) position
            Returns:
                features : change self.features
        '''
        material = self.fp[s]
        b = np.zeros(6)
        b[int(material)] = 1
        self.features[:6] = b
        self.features[6] = self.min_distance_to_wall(s)
        return self.features

    def get_feature_map(self):
        fp_x_size = self.fp.shape[0]
        fp_y_size = self.fp.shape[1]
        
        encoded_fp = np.zeros((fp_x_size,fp_y_size, self.fp.max()+1), dtype=int)
        encoded_fp[all_idx(self.fp, axis=2)] = 1 

        dist_map = self.distance_map()


        return np.concatenate((encoded_fp,dist_map.reshape(dist_map.shape[0],dist_map.shape[1],1)), axis=2)

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
        for i in range(10):
            for j in detect_dir:
                x = i*j[0] + point[0]
                y = i*j[1] + point[1]
                if x >= self.fp.shape[0] or y >=  self.fp.shape[1]:
                    break
                if self.fp[x,y] == 1 or self.fp[x,y] == 2:
                    return i/10
        return 10/10

#%%
class f_abs_with_direction:
    
    def __init__(self, fp) -> None:
        self.Feature_dims = 6
        self.features = np.zeros(self.Feature_dims)
        self.fp = fp.astype('int')
        self.modify_fp()
        self.action_space = [(1,1),(0,1),(-1,1),(1,0),(0,0),(-1,0),(1,-1),(0,-1),(-1,-1)]

    def modify_fp(self):
        self.fp[self.fp==2] = 1
        self.fp[self.fp==3] = 2
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
        self.features[b.size+1] = self.min_distance_to_wall(s)
        self.features[-2] = np.dot(a_p,a_c)
        self.features[-1] = np.linalg.norm(a_c) - np.linalg.norm(a_p)
        return self.features

    def get_feature_map(self):
        fp_x_size = self.fp.shape[0]
        fp_y_size = self.fp.shape[1]
        
        encoded_fp = np.zeros((fp_x_size,fp_y_size, self.fp.max()+1), dtype=int)
        encoded_fp[all_idx(self.fp, axis=2)] = 1 

        dist_map = self.distance_map()
        fp_feature_map = np.concatenate((encoded_fp,dist_map.reshape(dist_map.shape[0],dist_map.shape[1],1)), axis=2)

        # action_map : [previous action, current action]
        action_map = np.zeros(len(self.action_space), len(self.action_space),2)

        for i in range(len(self.action_space)):
            for j in range(len(self.action_space)):
                a_c = self.action_space[j]
                a_p = self.action_space[i]
                action_map[i,j,0] = np.dot(a_c, a_p)
                action_map[i,j,1] = np.linalg.norm(a_c) - np.linalg.norm(a_p)
        return (fp_feature_map, action_map)

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
        for i in range(10):
            for j in detect_dir:
                x = i*j[0] + point[0]
                y = i*j[1] + point[1]
                if x >= self.fp.shape[0] or y >=  self.fp.shape[1]:
                    break
                if self.fp[x,y] == 1 or self.fp[x,y] == 2:
                    return i/10
        return 10/10

#%%
def run():
    grid_width = 10
    baseColor_fn = 'baseColor.png'
    fp_zone_fn = 'fp2_zone.png'
    fp_code_fn = 'fp2_code.png'
    #%%
    fp = get_fp(baseColor_fn, fp_zone_fn, fp_code_fn, grid_width)
    print(fp.shape)
    # plt.imshow(fp)
    # plt.show()

    feat = f_abs(fp)
    feat.get_features((87,99))
    print(feat.features)
    
if __name__ == "__main__":
    path = os.getcwd()
    print("current working directory: {0}".format(path))
    if path.split('/')[-1] != 'src':
        path = os.path.join(path,'src')
        os.chdir(path)
    run()



