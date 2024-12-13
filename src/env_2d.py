#!/usr/bin/env python
""" @package environment_interface
Loads an environment file from a database and returns a 2D
occupancy grid.

Inputs : file_name, x y resolution (meters to pixel conversion)
Outputs:  - 2d occupancy grid of the environment
          - ability to check states in collision
"""
# Reference: https://github.com/mohakbhardwaj/planning_python/tree/1a41e75bb59d308e7eae2d49797291a13d0a4851

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage


class Env2D():
  def __init__(self):
    self.plot_initialized = False
    self.image = None
    

  def initialize(self, envfile, params= {'x_lims': (0,100), 'y_lims': (0,100)}):
    """
      Initialize environment from file with given params
      @param envfile - full path of the environment file
      @param params  - dict containing relevant parameters
                           {x_lims: [lb, ub] in x coordinate (meters),
                            y_lims: [lb, ub] in y coordinate (meters)}
      The world origin will always be assumed to be at (0,0) with z-axis pointing outwards
      towards right
    """
    try:
      self.image = plt.imread(envfile)
      self.image2 = plt.imread(envfile)# for visualizing alpha image
      # Resize the image to 100x100
      self.image2 = ndimage.zoom(self.image2, (100/self.image2.shape[0], 100/self.image2.shape[1]), order=0)
      self.image2[self.image2 != 0] = 1

      if len(self.image.shape) > 2:
        self.image = np.mean(self.image, axis=2)
        self.image[self.image!=0.0] = 1.0 
    except IOError:
      print("File doesn't exist. Please use correct naming convention for database eg. 0.png, 1.png .. and so on. You gave, %s"%(envfile))
    self.x_lims = params['x_lims']
    self.y_lims = params['y_lims']

    # resolutions
    self.x_res  = (self.x_lims[1] - self.x_lims[0])/((self.image.shape[1]-1)*1.)
    self.y_res  = (self.y_lims[1] - self.y_lims[0])/((self.image.shape[0]-1)*1.)

    orig_pix_x = math.floor(0 - self.x_lims[0]/self.x_res) #x coordinate of origin in pixel space
    orig_pix_y = math.floor(0 - self.y_lims[0]/self.y_res) #y coordinate of origin in pixel space
    self.orig_pix = (orig_pix_x, orig_pix_y)
    
  

  def get_env_image(self):
    return self.image

  def get_random_start_and_goal(self):
    random_start = tuple()
    random_goal = tuple()
    while True:
      random_start = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), random.uniform(self.y_lims[0], self.y_lims[1] - 1))
      if self.collision_free(random_start):
        break
    while True:
      random_goal = (random.uniform(self.x_lims[0], self.x_lims[1] - 1), random.uniform(self.y_lims[0], self.y_lims[1] - 1))
      if self.collision_free(random_goal):
        break
    return (random_start, random_goal)

  def collision_free(self, state):
    """ Check if a state (continuous values) is in collision or not.

      @param state - tuple of (x,y) or (x,y,th) values in world frame
      @return 1 - free
              0 - collision
    """
    try:
      pix_x, pix_y = self.to_image_coordinates(state)
      return round(self.image[pix_y][pix_x])
    except IndexError:
      # print("Out of bounds, ", state, pix_x, pix_y)
      return 0


  def es_points_along_line(self, start, end, r):
      """
      Equally-spaced points along a line defined by start, end, with resolution r
      :param start: starting point
      :param end: ending point
      :param r: maximum distance between points
      :return: yields points along line from start to end, separated by distance r
      """
      d = np.linalg.norm(np.array(end) - np.array(start))
      n_points = int(np.ceil(d / r))
      if n_points > 1:
          step = d / (n_points - 1)
          for i in range(n_points):
              next_point = self.steer(start, end, i * step)
              yield next_point
              
  def free_space_volume(self):
    free_space_volume = 0
    for i in range(len(self.image)):
      for j in range(len(self.image[i])):
        free_space_volume += round(self.image[i][j])
    return free_space_volume

  def steer(self, start, goal, d):
      """
      Return a point in the direction of the goal, that is distance away from start
      :param start: start location
      :param goal: goal location
      :param d: distance away from start
      :return: point in the direction of the goal, distance away from start
      """
      start, end = np.array(start), np.array(goal)
      v = end - start
      u = v / (np.sqrt(np.sum(v ** 2))) # (cos, sin)
      steered_point = start + u * d
      return tuple(steered_point)

  def collision_free_edge(self, state1, state2, r=1):
    """ Check if a state (continuous values) is in collision or not.

      @param state - tuple of (x,y) or (x,y,th) values in world frame
      @return 1 - free
              0 - collision
    """
    if (state1==state2):
      return True
    points = self.es_points_along_line(state1, state2, r)
    coll_free = all(map(self.collision_free, points))
    return coll_free

  

  def to_image_coordinates(self, state):
    """Helper function that returns pixel coordinates for a state in
    continuous coordinates

    @param  - state in continuous world coordinates
    @return - state in pixel coordinates """
    pix_x = int(self.orig_pix[0] + math.floor(state[0]/self.x_res))
    pix_y = int(self.image.shape[1]-1 - (self.orig_pix[1] + math.floor(state[1]/self.y_res)))
    return (pix_x,pix_y)



  def initialize_plot(self, start, goal, grid_res=None, plot_grid=False):
    # if not self.plot_initialized:
    self.figure, self.axes = plt.subplots()
    self.axes.set_xlim(self.x_lims)
    self.axes.set_ylim(self.y_lims)
    if plot_grid and grid_res:
      self.axes.set_xticks(np.arange(self.x_lims[0], self.x_lims[1], grid_res[0]))
      self.axes.set_yticks(np.arange(self.y_lims[0], self.y_lims[1], grid_res[1]))
      self.axes.grid(which='both')
    self.visualize_environment()
    self.line, = self.axes.plot([],[])
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_state(start, color='red',marker='*')
    self.plot_state(goal, color='blue', marker='*')
    self.figure.canvas.draw()
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 
    self.plot_initialized = True


  def visualize_environment(self):
    # Check if image is grayscale (2D) or RGB (3D)
    if self.image2.ndim == 2:  # Grayscale image
        # Convert grayscale to pseudo-RGB by stacking the same channel
        self.image2 = np.stack([self.image2] * 3, axis=-1)
    alpha = (~np.all(self.image2 == 1.0, axis=2)).astype(np.uint8) * 255  # Create alpha channel
    rgba = np.dstack((self.image2, alpha)).astype(np.uint8)  # Add alpha channel to the image
    
    # Display the image
    self.axes.imshow(rgba, 
                     extent=(self.x_lims[0], self.x_lims[1], self.y_lims[0], self.y_lims[1]), 
                     zorder=1)


  def plot_edge(self, edge, linestyle='solid', color='black', linewidth=1):
    x_list = []
    y_list = []
    for s in edge:
      x_list.append(s[0])
      y_list.append(s[1])
    self.figure.canvas.restore_region(self.background)
    self.line.set_xdata(x_list)
    self.line.set_ydata(y_list)
    self.line.set_linestyle(linestyle)
    self.line.set_linewidth(linewidth)
    self.line.set_color(color)
    self.axes.draw_artist(self.line)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox) 

  def plot_edges(self, edges,linestyle='solid', color='black', linewidth=1):
    """Helper function that simply calls plot_edge for each edge"""
    for edge in edges:
      self.plot_edge(edge, linestyle, color, linewidth)

  def plot_state(self, state, color='red', marker='o'):
    """Plot a single state on the environment"""
    # self.figure.canvas.restore_region(self.background)
    self.axes.plot(state[0], state[1], marker=marker,color = color)
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)
    
  def plot_path(self, path, linestyle='solid', color='red', linewidth=4):
    flat_path = [item for sublist in path for item in sublist]
    self.plot_edge(flat_path, linestyle, color, linewidth)
  
  # Added functions
  def plot_kde(self, tree, rx, ry, rz, random_sample):
    self.plot_tree(tree)
    self.axes.scatter(rx, ry, c=rz, s=50, alpha=0.20, edgecolors='none')
    self.plot_state(random_sample, color='k')
    self.figure.canvas.blit(self.axes.bbox)
    self.background = self.figure.canvas.copy_from_bbox(self.axes.bbox)

  # Added functions
  def plot_pcolor(self, X, Y, Z, alpha=1.0, cmap='plasma'):
    self.axes.pcolor(X, Y, Z, alpha=alpha, cmap=cmap, shading='auto')

  def plot_title(self, name):
    self.axes.set_title(name)


  def plot_tree(self, tree, color='black', linewidth=1):
    for child, parent in tree.items(): 
      self.axes.plot((child[0], parent[0]),
                     (child[1], parent[1]), 
                     color=color, 
                     linewidth=linewidth)
      
  def plot_save(self, name):
    plt.savefig(name +'.png')
    plt.close(self.figure) 
    
  def plot_states(self, states, color='red'):
    for state in states:
      self.plot_state(state, color=color)    