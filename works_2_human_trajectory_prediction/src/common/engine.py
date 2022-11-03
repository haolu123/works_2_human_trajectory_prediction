import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os

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
