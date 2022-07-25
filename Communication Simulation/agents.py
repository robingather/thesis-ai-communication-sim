import numpy as np
import torch
import uuid
from helper import Point, distance_between
from model import MLP
import constants as C

class Agent:

    def __init__(self, env):
        self._id = uuid.uuid1()
        self.model = MLP(C.N_INPUTS.prey, C.N_HIDDEN, C.N_ACTIONS)
        self.pos = env.random_point() if env is not None else Point(0,0)
        self.com = 0
        self.dead = False
        self.moving = True
        self.scores = {'killed':0, 'eaten':0, 'survived':0}
        self.target_id = None

    def get_state(self, env):
        # get agent current input state vector based on environment
        if self._type == 'prey' and not self.moving:
            return None

        point_l = Point(self.pos.x - 1, self.pos.y)
        point_r = Point(self.pos.x + 1, self.pos.y)
        point_u = Point(self.pos.x, self.pos.y - 1)
        point_d = Point(self.pos.x, self.pos.y + 1)

        target_pop = env.preds if self._type == 'prey' else env.preys
        target, _ = target_pop.get_closest_agent(self.pos)
        self.target_id = target._id
        
        state = [
        
            # target relative distance
            np.round(distance_between(self.pos, target.pos, normalize=False, justpos=0),3),
            np.round(distance_between(self.pos, target.pos, normalize=False, justpos=1),3),

            # obstacles
            env.is_collision(point_l),
            env.is_collision(point_r),
            env.is_collision(point_u),
            env.is_collision(point_d),

            ]

        # within- and between-population communication
        for pop, pop_type in [(env.preds.agents,'pred'), (env.preys.agents,'prey')]:
            if self._type == 'pred' and pop_type == 'pred' and not C.COMMUNICATE_WITHIN_POP.pred\
                or self._type == 'prey' and pop_type == 'prey' and not C.COMMUNICATE_WITHIN_POP.prey\
                or self._type == 'pred' and pop_type == 'prey' and not C.HEAR_BETWEEN_POP.pred\
                or self._type == 'prey' and pop_type == 'pred' and not C.HEAR_BETWEEN_POP.prey:
                state.extend([0,0,0,0])
                continue
            com_l, com_r, com_u, com_d = 0, 0, 0, 0
            for a in pop:
                if a is None:
                    continue
                dist = distance_between(a.pos, self.pos)
                if a.pos != self.pos and dist <= C.RAD_COM and a.com > 0:
                    com_l += a.com if a.pos.x < self.pos.x else 0
                    com_r += a.com if a.pos.x > self.pos.x else 0
                    com_u += a.com if a.pos.y < self.pos.y else 0
                    com_d += a.com if a.pos.y > self.pos.y else 0
                    break
            state.extend([
                np.round(com_l,3),
                np.round(com_r,3),
                np.round(com_u,3),
                np.round(com_d,3)
            ])
            
        return [float(s) for s in state]

    def get_action(self, state):
        # get agent action based on input state and model
        state_t = torch.tensor(state, dtype=torch.float, device=C.DEVICE)
        values = self.model(state_t)
        com_val = values[4].item()
        action = torch.argmax(values).item()
        action2 = None
        if action == 4:
            values = torch.cat([values[0:action], values[action+1:]])
            action2 = torch.argmax(values).item()
        return action, values, action2, com_val

class Predator(Agent):
    def __init__(self, env=None):
        super(Predator, self).__init__(env)
        self._type = 'pred'
        self.health = C.MAX_HEALTH.pred

class Prey(Agent):
    def __init__(self,env=None):
        super(Prey, self).__init__(env)
        self._type = 'prey'
        self.health = C.MAX_HEALTH.prey