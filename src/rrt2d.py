import math
import random

class RRT(object): 
    def __init__(self, q_init, env, extend_len=5.0, goal_bias_ratio=0.05):
        self._root = q_init
        self._rrt = {q_init: q_init} 
        self._graph = {q_init: []}
        self._env = env
        self._extend_len = extend_len
        self._num_vertices = 1
        self.goal_bias_ratio = goal_bias_ratio
        self.x_lims = env.x_lims
        self.y_lims = env.y_lims
        self.x_range = self.x_lims[1] - self.x_lims[0]
        self.y_range = self.y_lims[1] - self.y_lims[0]

    def search_nearest_vertex(self, p):
        rrt_vertices = self._rrt.keys()
        min_dist = self.x_range ** 2 + self.y_range ** 2
        min_q = tuple()
        for q in rrt_vertices:
            distance = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
            if min_dist > distance:
                min_dist = distance
                min_q = q
        return min_q

    def is_contain(self, q):
        return q in self._rrt

    def add(self, q_new, q_near):
        self._rrt[q_new] = q_near 
        self._graph[q_new] = [] 
        self._graph[q_near].append(q_new)


    def get_rrt(self):
        return self._rrt

    def get_parent(self, q):
        return self._rrt[q]
    
    def extend(self, q_rand, add_node=True):
        q_near = self.search_nearest_vertex(q_rand)
        q_new = self._calc_new_point(q_near, q_rand, delta_q=self._extend_len)
        if self.is_collision(q_new) or self.is_contain(q_new) or (not self._env.collision_free_edge(q_near, q_new)):
            return None
        if add_node:
            self.add(q_new, q_near)
            self._num_vertices += 1
        return q_new

    def _calc_new_point(self, q_near, q_rand, delta_q=1.0):
        if self.distance(q_near, q_rand) < delta_q:
            return q_rand
        angle = math.atan2(q_rand[1] - q_near[1], q_rand[0] - q_near[0])
        q_new = (q_near[0] + delta_q * math.cos(angle), q_near[1] + delta_q * math.sin(angle))
        return q_new

    def is_collision(self, p):
        return not self._env.collision_free(p)

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_goal_reached(self, q_new, end, goal_region_radius=3):
        if self.distance(q_new, end) <= goal_region_radius:
            return True
        else:
            return False

    def reconstruct_path(self, end):
        path = []
        q = end
        while not q == self._root:
            path.append([q, self.get_parent(q)])
            q = self.get_parent(q)
        return path
    
    def get_solution_node(self, end):
        path = []
        q = end
        while not q == self._root:
            path.append(self.get_parent(q))
            q = self.get_parent(q)
        return path[:-1]

    def solve(self, goal, max_iter, goal_region_radius=3):
        iter = 0
        while iter <= max_iter:
            iter += 1
            if random.uniform(0, 1) > self.goal_bias_ratio:
                random_sample = (random.randint(self._env.x_lims[0], self._env.x_lims[1] - 1), random.randint(self._env.y_lims[0], self._env.y_lims[1] - 1))
                if self.is_collision(random_sample) or self.is_contain(random_sample):
                    continue
            else: # goal bias
                random_sample = goal
            q_new = self.extend(random_sample)
            if not q_new:
                continue

            if self.is_goal_reached(q_new, goal, goal_region_radius):
                # solution_path = self.reconstruct_path(q_new)
                return True
        return False



# # My previous code reference:
# class nonhn_RRT:
#     def __init__(self,
#                  start_config,
#                  goal,
#                  v_init,
#                  w_init,
#                  n_obs,
#                  n_circ,
#                  n_rect,
#                  c_center_x,
#                  c_center_y,
#                  c_radius,
#                  rect_x ,
#                  rect_y,
#                  xMax,
#                  yMax):
#         (x0,y0,theta0) = start_config
#         self.xg = goal[0]
#         self.yg = goal[1]
#         self.x = []
#         self.y = []
#         self.theta = []
#         self.x.append(x0)
#         self.y.append(y0)
#         self.theta.append(theta0)
#         self.vc = [] #platform centre v list
#         self.wc = [] #platform centre w list
#         self.vc.append(v_init)
#         self.wc.append(w_init)
#         self.init_link = []
#         self.init_link.append(0)
#         #For Obstacles
#         self.n_obs = n_obs
#         self.n_circ = n_circ
#         self.n_rect = n_rect
#         self.c_center_x = c_center_x
#         self.c_center_y = c_center_y
#         self.c_radius = c_radius
#         self.rect_x = rect_x
#         self.rect_y = rect_y
#         self.xMax = xMax
#         self.yMax = yMax
#         self.way = []
        
        
#     # implement non-holonomic RRT to find platform centre velocities
#     def clearFiles(self,folder):
#         mypath = folder
#         for root, dirs, files in os.walk(mypath):
#             for file in files:
#                 os.remove(os.path.join(root, file))
        
#     def findDist(self,param1,param2):
#         return (((self.x[param1]-self.x[param2])**2 + (self.y[param1]-self.y[param2])**2)**0.5) 
        
        
#     #Expand_nh
#     def grow(self):
#         (x,y,theta) = rd.uniform (0,100),rd.uniform (0,100),rd.uniform(-math.pi, math.pi)
#         node = len(self.x) 
#         self.x.insert(node, x)
#         self.y.insert(node, y)
#         self.theta.insert(node,theta)
#         self.vc.insert(node, 0)
#         self.wc.insert(node, 0)
#         init_dist = self.findDist(node,0)
#         close_node = 0
#         for i in range(0,node):
#             if self.findDist(i,node) < init_dist:
#                 close_node = i
#                 init_dist =self.findDist(i,node)
#         final_pos,final_con = self.RRT_Implement(close_node,node)
#         if 0<=final_pos[0]<=100 and 0<=final_pos[1]<=100:
#             self.x.pop(node)
#             self.y.pop(node)
#             self.theta.pop(node)
#             self.vc.pop(node)
#             self.wc.pop(node)
#             self.x.insert(node, final_pos[0])
#             self.y.insert(node, final_pos[1])
#             self.theta.insert(node,final_pos[2])
#             self.vc.insert(node, final_con[0])
#             self.wc.insert(node, final_con[1])
#             if self.collisionNode()!=False:
#                 #Go to the next node
#                 self.nextNode(close_node,node,final_pos[2],final_con)
#                 if self.collisionEdge(self.x[close_node],self.y[close_node],self.x[node],self.y[node]) == False:
#                     self.x.pop(node)
#                     self.y.pop(node)
#                     self.theta.pop(node)
#                     self.vc.pop(node)
#                     self.wc.pop(node)
#                 else:
#                     self.init_link.insert(node,close_node)
#         else:
#             self.x.pop(node)
#             self.y.pop(node)
#             self.theta.pop(node)
#             self.vc.pop(node)
#             self.wc.pop(node)
    
    
    
    
#     def nextNode(self,close_node,node,theta,arr_con):
#         init = 10
#         if self.findDist(close_node,node)>init:
#             newT = self.theta[close_node] + arr_con[1]*dt
#             newX = self.x[close_node] + arr_con[0]*dt*math.cos(newT)
#             newY = self.y[close_node] + arr_con[0]*dt*math.sin(newT)
#             #Delete old node
#             self.x.pop(node)
#             self.y.pop(node)
#             self.theta.pop(node)
#             self.vc.pop(node)
#             self.wc.pop(node)
#             #Insert modified node
#             self.x.insert(node, newX)
#             self.y.insert(node, newY)
#             self.theta.insert(node,theta)
#             self.vc.insert(node, arr_con[0])
#             self.wc.insert(node, arr_con[1])
    
    
#     def collisionNode(self):
#         #Check for rectangle)
#         maxIter = len(self.x)-1
#         x = self.x[maxIter]
#         y = self.y[maxIter]
#         for i in range(1,self.n_rect+1):
#             x_l = self.rect_x[4*(i-1)]
#             x_r = self.rect_x[4*(i-1)+2]
#             y_b = self.rect_y[4*(i-1)]
#             y_t = self.rect_y[4*(i-1)+2]
#             if x_l<=x<=x_r and y_b<=y<=y_t: 
#                 self.x.pop(maxIter)
#                 self.y.pop(maxIter)
#                 self.theta.pop(maxIter)
#                 self.vc.pop(maxIter)
#                 self.wc.pop(maxIter)
#                 return False
#         #Check for circle
#         for i in range(self.n_circ):
#             x_c = self.c_center_x[i]
#             y_c = self.c_center_y[i]
#             rad = self.c_radius[i]
#             dist_c = ((x-x_c)**2 + (y-y_c)**2)**0.5
#             if dist_c<=rad: 
#                 self.x.pop(maxIter)
#                 self.y.pop(maxIter)
#                 self.theta.pop(maxIter)
#                 self.vc.pop(maxIter)
#                 self.wc.pop(maxIter)
#                 return False         
#         return True
    
#     def RRT_Implement(self,close_node,node):
#         initial_pos = np.array([self.x[close_node],self.y[close_node],self.theta[close_node]])
#         final_pos = np.array([self.x[node],self.y[node],self.theta[node]])
#         move = []
#         #High Velocity Movement
#         move.append((10,-math.pi/9))
#         move.append((10,math.pi/9))
#         move.append((10,0))
#          #Medium Velocity movement
#         move.append((5,-math.pi/9))
#         move.append((5,math.pi/9))
#         move.append((5,0))
#         #Low velocity movement -20,0,20 degree movement
#         move.append((1,-math.pi/9))
#         move.append((1,math.pi/9))
#         move.append((1,0))
        
#         dMax = 1e7+7
#         actual_pos = [0,0,0]
#         actual_con = [0,0]
#         #Iterate through possible controls
#         for i in range(len(move)):
#             xnew = initial_pos[0] + move[i][0]*dt*math.cos(initial_pos[2] + move[i][1]*dt)
#             ynew = initial_pos[1] + move[i][0]*dt*math.sin(initial_pos[2] + move[i][1]*dt)
#             d_c_g = np.linalg.norm(np.array([final_pos[0],final_pos[1]])-np.array([xnew,ynew])) 
#             if d_c_g < dMax:
#                 actual_pos[0] = xnew
#                 actual_pos[1] = ynew
#                 actual_pos[2] = initial_pos[2] + move[i][1]*dt
#                 actual_con[0] = move[i][0]
#                 actual_con[1] = move[i][1]
#                 dMax = d_c_g
#         return actual_pos,actual_con
    
    
#     def collisionEdge(self,x1,y1,x2,y2):
#         #Check for rectangle
#         n = len(self.x)-1
#         subDiv = 100
#         for i in range(1,self.n_rect+1):
#             x_l = self.rect_x[4*(i-1)]
#             x_r = self.rect_x[4*(i-1)+2]
#             y_b = self.rect_y[4*(i-1)]
#             y_t = self.rect_y[4*(i-1)+2]
#             for j in range(0,subDiv+1):
#                 sDiv= j/subDiv
#                 valX = x1*sDiv + x2*(1-sDiv)
#                 valY = y1*sDiv + y2*(1-sDiv)
#                 if x_l<=valX<=x_r and y_b<=valY<=y_t:
#                     return False
#         #Check for circle
#         for i in range(self.n_circ):
#             x_c = self.c_center_x[i]
#             y_c = self.c_center_y[i]
#             rad = self.c_radius[i]
#             for j in range(0,subDiv+1):
#                 sDiv= j/subDiv
#                 valX = x1*sDiv + x2*(1-sDiv)
#                 valY = y1*sDiv + y2*(1-sDiv)
#                 dist_c = ((valX-x_c)**2 + (valY-y_c)**2)**0.5
#                 if dist_c<=rad: 
#                     return False   
#         return True
    
        
#     def reached(self):
#         end_pos  = np.array([self.x[len(self.x)-1],self.y[len(self.y)-1]])
#         goal_pos = np.array([self.xg,self.yg])
#         diff_pos = end_pos - goal_pos
#         dist_pos_goal = np.linalg.norm(diff_pos**2)
#         if dist_pos_goal<4:
#             return True
#         return False
 
#     def wayGoal(self):
#     #find goal state
#         way_goal = []
#         actual = 0
#         #Iterate past nodes:
#         for i in range(0,len(self.x)):
#             curr_pos = np.array([self.x[i],self.y[i]])
#             goal_pos = np.array([self.xg,self.yg])
#             diff_pos = curr_pos - goal_pos
#             dist_curr_goal = np.linalg.norm(diff_pos**2)
#             if dist_curr_goal<4:
#                 break
#         #Now find path to goal_link
#         way_goal.append(i)
#         pre_goal_link = self.init_link[i]
#         while(pre_goal_link!=0):
#             way_goal.append(pre_goal_link)
#             pre_goal_link = self.init_link[pre_goal_link]
#         way_goal.append(0)
#         self.way = way_goal



#     def showpath(self):
#         """
#         Create a new directory (say nonholonomic_path) and save snapshots of the robot moving along the
#         trajectory found by RRT at every step with the step index.
#         e.g: snap1.png, snap2.png ...
#         """
#         isExist = os.path.exists("nonholonomic_path")
#         if not isExist:

#           # Create a new directory because it does not exist 
#             os.makedirs("nonholonomic_path")
#             print("The new directory is created!")
#         self.clearFiles("nonholonomic_path")

#         n = len(self.way)-1
#         pp = 0
#         for i in range (0,len(self.x)):
#             par=self.init_link[i]
#             plt.plot([self.x[i],self.x[par]],[self.y[i],self.y[par]],color='black')
#         for i in range (n):
#             start = self.way[-i-1]
#             end = self.way[-i-2]
#             centre_v.append((self.vc[self.way[n-i-1]]))
#             w_list.append((self.wc[self.way[n-i-1]]))
            
            
#             obst_vertex_x = [30, 30,50, 50, 
#                          70, 70, 80, 80,  
#                          40, 40, 60, 60,
#                          40, 40, 60, 60, 
#                          10, 10, 20, 20, 
#                          60, 60, 80, 80] 
#             obst_vertex_y = [72,  90,  90, 72, 
#                              40,  60,  60, 40,
#                              50,  60,  60, 50,
#                               0,  28,  28,  0,
#                              60,  75,  75, 60,
#                              95, 100, 100, 95]
#             plt.plot(self.xg,self.yg,'g*',markersize=20)
#             plt.plot(self.x[0],self.y[0],'b*',markersize=20)
#             for j in range(1,self.n_rect+1): 
#                 x = obst_vertex_x[4*(j-1)]
#                 y = obst_vertex_y[4*(j-1)]
#                 width  = obst_vertex_x[4*(j-1)+2] - obst_vertex_x[4*(j-1)]
#                 height = obst_vertex_y[4*(j-1)+2] - obst_vertex_y[4*(j-1)]
#                 rect = plt.Rectangle((x,y),width,height,linewidth=1,color='r')
#                 plt.gca().add_patch(rect)

#             circle_x = self.c_center_x
#             circle_y = self.c_center_y
#             radii = self.c_radius-np.full((self.n_circ), length)

#             for j in range(self.n_circ): #plotting  circles
#                 circle = plt.Circle((circle_x[j],circle_y[j]),radii[j], color='r')
#                 plt.gca().add_patch(circle)
            
#             way_x = np.array([self.x[start],self.x[end]])
#             way_y = np.array([self.y[start],self.y[end]])
#             plt.plot(way_x,way_y)

#             for j in range (i):
#                 start = self.way[-j-1]
#                 end = self.way[-j-2]
#                 way_x = np.array([self.x[start],self.x[end]])
#                 way_y = np.array([self.y[start],self.y[end]])
#                 plt.plot(way_x,way_y)
#             plt.gca().set_aspect('equal', adjustable='box')
#             name = f'nonholonomic_path/snap{str(pp)}.png'
#             plt.savefig(name,dpi=250)
#             pp+=1
#         plt.show()
            
        

        
#     def showtree(self):
#         """
#         Create a new directory (say nonholonomic_tree) and save snapshots of the evolution of the RRT tree
#         at every step with the step index.
#         e.g: snap1.png, snap2.png ...
#         """
#         isExist = os.path.exists("nonholonomic_tree")
#         if not isExist:

#           # Create a new directory because it does not exist 
#             os.makedirs("nonholonomic_tree")
#             print("The new directory is created!")
#         self.clearFiles("nonholonomic_tree")
        
        
#         node_num = 0
#         obst_vertex_x = [30, 30,50, 50, 
#                          70, 70, 80, 80,  
#                          40, 40, 60, 60,
#                          40, 40, 60, 60, 
#                          10, 10, 20, 20, 
#                          60, 60, 80, 80] 
#         obst_vertex_y = [72,  90,  90, 72, 
#                          40,  60,  60, 40,
#                          50,  60,  60, 50,
#                           0,  28,  28,  0,
#                          60,  75,  75, 60,
#                          95, 100, 100, 95]
#         plt.plot(self.xg,self.yg,'g*',markersize=20)
#         plt.plot(self.x[0],self.y[0],'b*',markersize=20)
#         for j in range(1,self.n_rect+1): 
#             x = obst_vertex_x[4*(j-1)]
#             y = obst_vertex_y[4*(j-1)]
#             width  = obst_vertex_x[4*(j-1)+2] - obst_vertex_x[4*(j-1)]
#             height = obst_vertex_y[4*(j-1)+2] - obst_vertex_y[4*(j-1)]
#             rect = plt.Rectangle((x,y),width,height,linewidth=1,color='r')
#             plt.gca().add_patch(rect)
            
#         circle_x = self.c_center_x
#         circle_y = self.c_center_y
#         radii = self.c_radius-np.full((self.n_circ), length)
#         # print("Circle")
#         for j in range(self.n_circ): 
#             circle = plt.Circle((circle_x[j],circle_y[j]),radii[j], color='r')
#             plt.gca().add_patch(circle)
#         # print("FINAL")
#         pp = 0
#         for j in range (0,len(self.x)):
#             par=self.init_link[j]
#             plt.plot([self.x[j],self.x[par]],[self.y[j],self.y[par]],color='black')
#             plt.gca().set_aspect('equal', adjustable='box')
#             if j%50==0:
#                 name = f'nonholonomic_tree/snap{str(pp)}.png'
#                 plt.savefig(name,dpi=250)
#                 pp+=1
#         name = f'nonholonomic_tree/snap{str(pp)}.png'
#         plt.savefig(name,dpi=250)
#         plt.show()
    
        
#     def test_env(self):
#         """
#         Function to generate test environment
#         """
#         # the vertices for each obstacle are given in the clockwise order starting from lower left
#         obst_vertex_x = [30, 30,50, 50, 
#                          70, 70, 80, 80,  
#                          40, 40, 60, 60,
#                          40, 40, 60, 60, 
#                          10, 10, 20, 20, 
#                          60, 60, 80, 80] 
#         obst_vertex_y = [72,  90,  90, 72, 
#                          40,  60,  60, 40,
#                          50,  60,  60, 50,
#                           0,  28,  28,  0,
#                          60,  75,  75, 60,
#                          95, 100, 100, 95]
#         plt.plot(self.xg,self.yg,'g*',markersize=20)
#         plt.plot(self.x[0],self.y[0],'b*',markersize=20)
#         for i in range(1,self.n_rect+1): 
#             x = obst_vertex_x[4*(i-1)]
#             y = obst_vertex_y[4*(i-1)]
#             width = obst_vertex_x[4*(i-1)+2] - obst_vertex_x[4*(i-1)]
#             height = obst_vertex_y[4*(i-1)+2] - obst_vertex_y[4*(i-1)]
#             rect = plt.Rectangle((x,y),width,height,linewidth=1,color='r')
#             plt.gca().add_patch(rect)
           
#         #centre positions of circular obstacles
#         circle_x = self.c_center_x
#         circle_y = self.c_center_y
#         radii = self.c_radius-np.full((self.n_circ), length)

#         for i in range(self.n_circ): #plotting  circles
#             circle = plt.Circle((circle_x[i],circle_y[i]),radii[i], color='r')
#             plt.gca().add_patch(circle)

#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.show()
