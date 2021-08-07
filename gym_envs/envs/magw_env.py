from gym_multigrid.multigrid import *
from marl import IqAgent
from random import randrange
from numpy.lib.shape_base import split

from numpy.random.mtrand import rand

class MAGW_Env(MultiGridEnv):
    def __init__(self, size, width, height, key_loc, goal_locs, view_size=7):
        self.world = World
        colors = [2,0]
        agents = []
        self.goal_locs = goal_locs
        self.key_loc = key_loc
        for i in range(2):
            agents.append(IqAgent(self.world, index=colors[i], view_size=view_size))
        
        self.key_loc = key_loc
        self.goal_locs = goal_locs

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=10000,
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )
    

    def _gen_grid(self, width, height):
        colors = [2,0]
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)
        split_idx = self._rand_int(2, width-2)
        self.grid.vert_wall(self.world, split_idx, 0)
        for i in range(len(self.goal_locs)):
            self.put_obj(Goal(self.world, colors[i]), self.goal_locs[i][0], self.goal_locs[i][1])

        
        door_idx = self._rand_int(1, height-2)

        self.put_obj(Door(self.world, color='blue', is_locked=True), split_idx, door_idx)
        
        self.place_obj(Key(self.world, color='blue'), top=(0, 0),
            size=(split_idx, height))
        
        self.place_agent(self.agents[0], top=[0,0], size=(split_idx, height))
        self.place_agent(self.agents[1], top=[split_idx+1, 1], size=[width-1-split_idx, height])

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
    

class MAGW_Env10x15(MAGW_Env):
    def __init__(self):
        super().__init__(size=None, height=10, width=15, key_loc=[4,5], goal_locs=[[1,8], [13,8]])