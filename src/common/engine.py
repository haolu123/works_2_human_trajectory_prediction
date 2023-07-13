import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
from common.config import argparser
from common.A_star import Env, AStar
import tqdm
import cvxpy as cp

def Get_BaseColor(img_name, pickle_name, baseColor_num):
    """This function is used to get the base color for color board
        Args: 
            img_name: the file name of the base color image
            pickle_name: the file name of the output file, which save the base colors.
            baseColor_num : the number of base colors.
            
        Returns: 
            baseColor: the list of the base color (bgr)
    """
    baseColor_img = cv2.imread(img_name)
    bgr_get = []
    def getposBgr(event, x, y, flags, param):
        nonlocal bgr_get
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr_get = baseColor_img[y, x]
    
    baseColor = []
    for i in range(baseColor_num):
        cv2.imshow("baseColor",baseColor_img)
        cv2.setMouseCallback("baseColor",getposBgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        baseColor.append(bgr_get)
    
    with open(pickle_name,'wb') as f:
        pickle.dump(baseColor, f)
    return baseColor

def get_grid_width(width, length):
    return int(min(40, math.sqrt(width*length/10000)))


def Label_gridFloorPlan(img_file, img_zone_file, width, length, grid_width, temp_result_folder, baseColor):
    """
    This function is used to label the colored floor plan

    Parameters
    ----------
    img_file : strings
        file name of the floor plan image
        
    width : int or float
        the width of the floor plan (cm)
    length : int or float
        the length of the floor plan (cm)
    grid_width : int or float
        the edge size of a grid (cm)
    temp_result_folder: str
        the direction of the temp_reulst folder
    baseColor: list (bgr color)
        the list of base color

    Returns
    -------
    fp_grid : np.array
        the grided floor plan 
        for each element ( 0: free spaces, 1: walls, 2: obstacles, 3: doorways, 4: area of interets, 5: boundarys, 6; others)

    """
    floorPlan_img = cv2.imread(img_file)
    fp_zone_img = cv2.imread(img_zone_file)
    fp_grid_w_len = int(floorPlan_img.shape[0]/(width/grid_width))
    fp_grid_l_len = int(floorPlan_img.shape[1]/(length/grid_width))
    
    grid_area = fp_grid_w_len*fp_grid_l_len
    
    grid_w = int(floorPlan_img.shape[0]/fp_grid_w_len)
    grid_l = int(floorPlan_img.shape[1]/fp_grid_l_len)
    
    
    
    fp_grid = np.zeros((grid_w, grid_l))
    fp_grid_zone = np.zeros((grid_w, grid_l))
    
    for i in range(grid_w):
        for j in range(grid_l):
            grid_patch = floorPlan_img[i*fp_grid_w_len:(i+1)*fp_grid_w_len, j*fp_grid_l_len:(j+1)*fp_grid_l_len]
            num_p_max = 0
            for i_color in range(len(baseColor)):
                num_pixel = len(np.where((grid_patch == baseColor[i_color]).all(2))[0])
                if num_pixel > num_p_max:
                    fp_grid[i,j] = i_color + 1
                    num_p_max = num_pixel
                    grid_patch = np.tile(baseColor[i_color],(grid_width,grid_width,1))
    
    for i in range(grid_w):
        for j in range(grid_l):
            grid_patch = fp_zone_img[i*fp_grid_w_len:(i+1)*fp_grid_w_len, j*fp_grid_l_len:(j+1)*fp_grid_l_len]
            num_p_max = 0
            for i_color in range(len(baseColor)):
                num_pixel = len(np.where((grid_patch == baseColor[i_color]).all(2))[0])
                if num_pixel > num_p_max:
                    fp_grid_zone[i,j] = i_color + 1
                    num_p_max = num_pixel
                    grid_patch = np.tile(baseColor[i_color],(grid_width,grid_width,1))
                    
    with open(temp_result_folder + 'fp_grid_groundtruth.pickle','wb') as f:
        pickle.dump(fp_grid_zone, f)
    
    fp_grid_gt = copy.deepcopy(fp_grid_zone)
    fp_grid[fp_grid>=6] = 0
    with open(temp_result_folder + 'fp_grid.pickle','wb') as f:
        pickle.dump(fp_grid, f)
    return fp_grid, fp_grid_gt

def get_fp(baseColor_fn, fp_zone_fn, fp_code_fn, grid_width):
        args = argparser.parse_args()
        base_color_fig_name = args.data_dir + baseColor_fn
        floor_plan_zone_fig_name = args.data_dir + fp_zone_fn
        floor_plan_code_fig_name = args.data_dir + fp_code_fn

        room_width = args.room_width
        room_length = args.room_length

        baseColor_temp_dir = args.temp_result_dir+'baseColor.pickle'
        flag = os.path.exists(baseColor_temp_dir)
        if args.use_saved_base_color and flag:
            with open(args.temp_result_dir+'baseColor.pickle', 'rb') as f:
                baseColor = pickle.load(f)
        else:
            baseColor = Get_BaseColor(base_color_fig_name, args.temp_result_dir + 'baseColor.pickle', 16)

        flag = os.path.exists(args.temp_result_dir + 'fp_grid.pickle')
        if args.use_save_temp_result and flag:
            with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
                fp_grid = pickle.load(f)
            with open(args.temp_result_dir + 'fp_grid_groundtruth.pickle','rb') as f:
                fp_grid_gt = pickle.load(f)
        else:
            fp_grid, fp_grid_gt =  Label_gridFloorPlan(floor_plan_code_fig_name, floor_plan_zone_fig_name, room_width, room_length, grid_width, args.temp_result_dir, baseColor)
        
        
        return fp_grid

def A_star_simulation(args, fp_grid):
    """
        get training dataset (using A* algroithm to mimic human behavior)
        Args: 
            args: pre-defined parameters,
            fp_grid: grided floor plan (array)
            grid_width: grid width/cm (float)
            range_b: boundary range (may not use here)

        Returns:
            path_all : list of paths
            bound_slices, b_mid: may not used.
    """
    walk_freq_map = np.zeros(fp_grid.shape)
    # build enviroment for A* algorithm
    fp_env = Env(fp_grid)

    # begin simulation
    path_all = []
    for loop_num in tqdm.tqdm(range(args.path_num)):
        # random select two points as start and end point
        (s_start, s_goal) = fp_env.get_start_end_point()

        # add random small obstacles
        fp_env.add_random_obs(args.random_obstacle_dense)

        # get the shortest path between start point and end point
        astar = AStar(s_start, s_goal, "euclidean", fp_env)
        path, visited = astar.searching(args.doorway_penalty, args.wall_penalty)

        # save shortest paths
        path_all.append(path)

        # mark in the walk frequence map
        if args.save_hotmap:
            for p in path:
                walk_freq_map[p[0],p[1]] += 1

    if args.save_hotmap:
        # plot walk_freq_map
        plt.imshow(walk_freq_map, cmap='hot', interpolation='nearest')
        plt.title("obstacle dense:%.3f, doorway penalty:%d , wall_penalty:%d" % (args.random_obstacle_dense, args.doorway_penalty, args.wall_penalty))
        img_name = "obstacle_dense_" + str(args.random_obstacle_dense) + "doorway_penalty" + str(args.doorway_penalty) + "wall_penalty" + str(args.wall_penalty) +'.png'
        folder_name = args.temp_result_dir+'hotmap/'
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(folder_name + img_name)
        # plt.show()

        with open(args.temp_result_dir+"walk_freq_map.pickle", 'wb') as f:
            pickle.dump(walk_freq_map,f)

    if args.save_temp_result:
        # save simulated paths
        with open(args.temp_result_dir+'path_all.pickle', 'wb') as f:
            pickle.dump(path_all, f)
    
    if args.save_unity3d_result:
        folder_name_unity ='unity3d_result/'
        os.makedirs(folder_name_unity, exist_ok=True)

        # save simulated information for unity 3d simulation
        fp_grid_unity = copy.deepcopy(fp_grid)
        fp_grid_unity[fp_grid_unity==2] = 1
        
        # save obstacles
        obstacle_where = np.where(fp_grid_unity==1)
        obstacle_list = np.zeros((obstacle_where[0].shape[0],2))
        for i in range(obstacle_where[0].shape[0]):
            obstacle_list[i,:] = [obstacle_where[0][i],obstacle_where[1][i]]
            
        obstacle_list = obstacle_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_obstacle.txt", 'w+') as f:
            f.write("numWall,%d" % obstacle_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_obstacle.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, obstacle_list, fmt='%d',delimiter=',')
        
        # save area of interestings
        interest_where = np.where(fp_grid_unity==4)
        
        area_interest_list = np.zeros((interest_where[0].shape[0],2))
        for i in range(interest_where[0].shape[0]):
            area_interest_list[i,:] = [interest_where[0][i],interest_where[1][i]]
        area_interest_list = area_interest_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_interest.txt", 'w+') as f:
            f.write("numInterest,%d" % interest_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_interest.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, area_interest_list, fmt='%d',delimiter=',')
        

    return path_all

    
def get_G_matrix(fp_grid, grid_width):
    fp_grid_cover = copy.deepcopy(fp_grid)
    fp_grid_cover[fp_grid_cover > 2] = 0
    
    x_range = fp_grid_cover.shape[0]
    y_range = fp_grid_cover.shape[1]
    
    G = -np.ones((x_range,y_range,x_range,y_range)).astype(np.int8)
    
    for x_i in range(x_range):
        for x_j in range(y_range):
            # print((x_i,x_j))
            if fp_grid_cover[x_i,x_j] != 0:
                continue
            
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
                for i in range(100//grid_width):
                    for j in range(100//grid_width):
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
                            G[x_i,x_j,sc_grid_i,sc_grid_j] = 0
                            
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
                            G[x_i,x_j,sc_grid_i,sc_grid_j] = 1
                            # whether this grid in in shaddow
                            for obs_i in obs_slop:
                                if g_r > obs_i[0] and obs_i[1]-0.9/obs_i[0] <= g_a <= obs_i[1] + 0.9/obs_i[0]:
                                    # if it is in shaddow, set G[] = 0
                                    G[x_i,x_j,sc_grid_i,sc_grid_j] = 0
                                    break
                    
    G = G.reshape(x_range*y_range, x_range*y_range).astype(np.int8)
    G[G == -1] = 0
    G = G.T
    return G


def ILP_solver(n, G, R):
    sensor_placement = []
    # x_init = np.concatenate(np.zeros(n,1), np.zeros(n,1), axis=0)

    x = cp.Variable(2*n, boolean = True)
    gamma = cp.Parameter(nonneg=True)
    gamma_vals = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    Obj = np.concatenate((np.zeros((n,1)), R.reshape((n,1))), axis=0)
    
    temp_line_1 = np.concatenate((np.ones((1,n)), np.zeros((1,n))),axis = 1)
    temp_line_2 = np.concatenate((-G, np.eye(n)),axis = 1)


    objective = 0.01* np.concatenate((np.ones((1,n)),np.zeros((1,n))), axis = 1) @ x - Obj.T @ x

    constrains = []
    for i in range(n):
        constrains.append(temp_line_2[i] @ x <= 0)
    
    constrains.append(temp_line_1 @ x <= gamma)

    problem = cp.Problem(cp.Minimize(objective), constrains)

    for val in gamma_vals:
        gamma.value = val
        problem.solve(solver = 'CBC',verbose=True, warm_start = False, maximumSeconds=4000)
        sensor_placement.append(x.value[:n])
    
    return sensor_placement