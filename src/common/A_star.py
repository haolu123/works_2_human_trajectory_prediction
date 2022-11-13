import copy
import random
import cv2
import numpy as np
import math
import heapq


class Env:
    def __init__(self, grid_fp):
        self.grid_fp = grid_fp
        self.x_range = grid_fp.shape[0]  # size of background
        self.y_range = grid_fp.shape[1]
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        self.obs_fp = self.obs_map()
        self.obs = copy.deepcopy(self.obs_fp)
        
        self.arIntrst = self.arIntrst_map()
        self.doorway = self.doorway_map() 
        
    def add_random_obs(self, prob):
        obs = set()
        for i in range(self.x_range):
            for j in range(self.y_range):
                if self.grid_fp[i,j] == 0:
                    float_num = random.uniform(0, 1)
                    if float_num < prob:
                        obs.add((i,j))
        self.obs = self.obs_fp.union(obs)
                    
                
    def doorway_map(self):
        """
        Get the doorway positions based on the grided map

        Returns
        -------
        doorway : set {(x1,y1),(x2,y2),...}
            points belongs to doorway.
        """
        
        doorway = set()
        for i in range(self.x_range):
            for j in range(self.y_range):
                if self.grid_fp[i,j] == 3:
                    doorway.add((i,j))
        return doorway
    
    def obs_map(self):
        """
        Get the obstacles based on the grided floor plan
        
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            for j in range(y):
                if self.grid_fp[i,j] == 1 or self.grid_fp[i,j] == 2:
                    obs.add((i,j))
        return obs
    
    def arIntrst_map(self):
        """
        Get the list of area of interest points

        Returns
        -------
        arIntrst: list of area of interest points
                arIntrst = {1:[point1, point2,...], 2:[point1, point2,...],...}
                point = (x,y)
        """
        
        grid_ft_arIntrst = copy.deepcopy(self.grid_fp)
        grid_ft_arIntrst[grid_ft_arIntrst!=4] = 0
        grid_ft_arIntrst[grid_ft_arIntrst==4] = 1
        arIntrst = dict()
        
        num_area, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_ft_arIntrst.astype(np.uint8), connectivity=8)
        
        for i in range(1, num_area):
            arIntrst[i-1] = []
            pixels = np.where(labels == i)
            for j in range(len(pixels[0])):
                arIntrst[i-1].append((pixels[0][j],pixels[1][j])) 
             
        return arIntrst
          
    def get_boundary_areas(self, grid_width, range_b):
        """
        Get the area near the boundaries

        Returns
        -------
        bound_slices: list with 4 numbers of each element
            [x_low, y_low, x_high, y_high]

        """         
        # range_b = 200
        
        grid_ft_bound = copy.deepcopy(self.grid_fp)
        grid_ft_bound[grid_ft_bound!=5] = 0
        grid_ft_bound[grid_ft_bound==5] = 1
        
        
        num_area, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_ft_bound.astype(np.uint8), connectivity=8)

        bound_slices = [[]]*(num_area-1)
        x_lim = grid_ft_bound.shape[1]
        y_lim = grid_ft_bound.shape[0]
        
        for i in range(num_area-1):
            x_l = max(stats[i+1][0]-(range_b//grid_width), 0)
            x_u = min(stats[i+1][0]+stats[i+1][2]+(range_b//grid_width),x_lim)
            y_l = max(stats[i+1][1]-(range_b//grid_width), 0)
            y_u = min(stats[i+1][1]+stats[i+1][3]+(range_b//grid_width),y_lim)
            bound_slices[i] = [x_l, y_l, x_u, y_u]
        
        b_mid = [[]]*(num_area-1)
        for i in range(num_area-1):
            x_m = stats[i+1][0] + stats[i+1][2]/2
            y_m = stats[i+1][1] + stats[i+1][3]/2
            b_mid[i] = [x_m, y_m]
        return bound_slices, b_mid
        
    def get_half_neighbor(self, s):
        """
        find up and left neighbors of point s.
        :param s: point(x ,y)
        :return: half_neighbors
        """
        half_motions = [u for u in self.motions if u[0]<=0 or u[1]<=0]
        return [(min(max(s[0] + u[0],0),self.grid_fp.shape[0]-1), min(max(s[1] + u[1],0),self.grid_fp.shape[1]-1)) for u in half_motions]
    
    def get_start_end_point(self):
        '''
        From the arIntrst dictionary random choose two points to be the start and end point

        Returns
        -------
        str_point : (x,y)
            start point
        end_point : (x,y)
            end point
        '''
        
        area_num = len(self.arIntrst)
        
        area_idx = np.random.choice(np.arange(area_num),2, replace = False)
        
        point_idx_str_len = len(self.arIntrst[area_idx[0]])
        point_idx_end_len = len(self.arIntrst[area_idx[1]])
        
        point_idx_str = random.randint(0, point_idx_str_len-1)
        point_idx_end = random.randint(0, point_idx_end_len-1)
        
        str_point = self.arIntrst[area_idx[0]][point_idx_str]
        end_point = self.arIntrst[area_idx[1]][point_idx_end]
        
        return (str_point, end_point)


'''
In Hao's program, we need to initialize the:
    1. start point of the A* alg
    2. end point of A*
    3. the 'heuristic_type', (usually it should be the Eucildean distan, since human can walk through all directions)
    4. the floor plan (map for the room) (class Env, defined above)

'''

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type, Env):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env  # class Env
        self.x_range = self.Env.x_range
        self.y_range = self.Env.y_range
        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.doorway = self.Env.doorway # position of doorways
        
        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self, doorway_penalty, wall_penalty):
        """
        A_star Searching.
        :return: path, visited order
        """
        try:
            self.PARENT[self.s_start] = self.s_start
        except:
            print(self.s_start)
            print(self.PARENT)
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n, doorway_penalty, wall_penalty)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))
        if s == self.s_goal:
            return self.extract_path(self.PARENT), self.CLOSED
        else:
            return [],[]

    

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(min(max(s[0] + u[0],0),self.x_range-1), min(max(s[1] + u[1],0),self.y_range-1)) for u in self.u_set]

    def cost(self, s_start, s_goal, doorway_penalty,wall_penalty):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        cost = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        
        if self.is_collision(s_start, s_goal):
            return math.inf
        
        if s_goal in self.doorway:
            cost += doorway_penalty
        
        wall_distance = self.min_distance_to_wall(s_goal)
        cost += 1/(wall_distance**2) * wall_penalty

        # s_goal_neigh = self.get_neighbor(s_start)
        # for i in s_goal_neigh:
        #     if i in self.obs:
        #         cost += wall_penalty
        
        
        return cost
    
    def min_distance_to_wall(self,point):
        """
        """
        detect_dir = [(0,1),(1,0),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for i in range(10):
            for j in detect_dir:
                x = i*j[0] + point[0]
                y = i*j[1] + point[1]
                if (x,y) in self.obs:
                    return i
        return 10

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_goal]
        s = self.s_goal

        while True:

            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

