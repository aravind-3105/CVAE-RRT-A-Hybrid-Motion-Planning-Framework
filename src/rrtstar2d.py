import math
from rrt2d import RRT as RRTBase

class RRTstar(RRTBase):
    def __init__(self, q_init, env, extend_len=1.0, dimension=2):
        super(RRTstar, self).__init__(q_init, env, extend_len=extend_len)
        self._dimension = dimension
        self._gamma = 2.0 * math.pow(1.0 + 1.0 / self._dimension, 1.0 / self._dimension) \
                         * math.pow(self._env.free_space_volume() / math.pi, 1.0 / self._dimension)
        self._q_goal_set = []
        self._q_best = None
        self._best_cost = math.inf

    def cost(self, q):
        q_now = q
        c = 0
        while(q_now != self._root):
            c += self.distance(q_now, self.get_parent(q_now))
            q_now = self.get_parent(q_now)
        return c

    def rewire(self, q_new):
        r = min(self._extend_len, self._gamma * math.pow(math.log(self._num_vertices) / self._num_vertices, 1.0 / self._dimension))
        q_near = [q for q in self._rrt.keys() if self.distance(q_new, q) <= r] 
        for q in q_near: #ChooseParent
            if not self._env.collision_free_edge(q, q_new):
                continue
            if self.cost(q) + self.distance(q, q_new) < self.cost(q_new):
                self._rrt[q_new] = q
        
        for q in q_near: 
            if q == self.get_parent(q_new):
                continue
            if not self._env.collision_free_edge(q, q_new):
                continue
            if self.cost(q_new) + self.distance(q_new, q) < self.cost(q):
                self._rrt[q] = q_new
    
    def update_best(self, goal):
        for q in self._q_goal_set:
            new_cost = self.cost(q) + self.distance(q, goal)
            if new_cost < self._best_cost:
                self._q_best = q
                self._best_cost = new_cost




# class RRTstar(RRTBase):
#     def __init__(self, q_init, env, extend_len=1.0, dimension=2):
#         super().__init__(q_init, env, extend_len=extend_len)
#         self._dimension = dimension
#         # self._gamma = 2.0 * (1.0 + 1.0 / self._dimension) ** (1.0 / self._dimension) \
#         #               * (env.free_space_volume() / math.pi) ** (1.0 / self._dimension)
#         self._gamma= 5.0
#         self._q_goal_set = []
#         self._q_best = None
#         self._best_cost = math.inf

#     def cost(self, q):
#         """Calculate the cost to reach node `q` from the root."""
#         total_cost = 0
#         while q != self._root:
#             parent = self._rrt[q]
#             total_cost += self.distance(q, parent)
#             q = parent
#         return total_cost

#     def rewire(self, q_new):
#         """Rewire the tree to ensure optimal connections."""
#         # r = min(
#         #     self._extend_len,
#         #     self._gamma * (math.log(self._num_vertices) / self._num_vertices) ** (1.0 / self._dimension)
#         # )
#         r = self._extend_len

#         q_near = [q for q in self._rrt.keys() if self.distance(q, q_new) <= r]

#         # Choose Parent
#         for q in q_near:
#             if not self._env.collision_free_edge(q, q_new):
#                 continue
#             if self.cost(q) + self.distance(q, q_new) < self.cost(q_new):
#                 self._rrt[q_new] = q  # Update parent of q_new

#         # Rewire Nearby Nodes
#         for q in q_near:
#             if q == self._rrt[q_new]:  # Skip the current parent
#                 continue
#             if not self._env.collision_free_edge(q, q_new):
#                 continue
#             if self.cost(q_new) + self.distance(q_new, q) < self.cost(q):
#                 self._rrt[q] = q_new  # Rewire q to q_new

#     def update_best(self, goal):
#         """Update the best goal node and its cost."""
#         for q in self._q_goal_set:
#             new_cost = self.cost(q) + self.distance(q, goal)
#             if new_cost < self._best_cost:
#                 self._q_best = q
#                 self._best_cost = new_cost
