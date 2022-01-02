import random

from population import Population
from helper import distance_between, Point
import constants as C

class Environment:

    def __init__(self):
        self.w = C.WORLD_SIZE[0]
        self.h = C.WORLD_SIZE[1]
        
        self.preds = Population('pred')
        self.preys = Population('prey')
        self.reset()

    def reset(self):
        # resets the environment and agent populations
        self.obstacles = []
        for i in range(C.N_OBST):
            self.obstacles.append(self.random_point())

        self.preds.populate(self)
        self.preys.populate(self)

    def point_exists(self, pt):
        # is point within environment?
        return pt.x >= 0 and pt.x < self.w and pt.y >= 0 and pt.y < self.h

    def point_free(self, pt):
        # is the point occupied?
        return self.point_exists(pt) and not \
            (pt in self.preds.get_positions() or pt in self.preys.get_positions() or pt in self.obstacles)

    def random_point(self):
        # get a random non-occupied point in the environment
        pt = Point(-1,-1)
        while not self.point_free(pt):
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            pt = Point(x,y)
        return pt

    def play_step(self, agent, action, a2, com):

        # 1. move
        if agent._type == 'pred' or (agent._type == 'prey' and agent.moving == True and C.PREY_MOVE):
            self._move(agent,action, a2, com)

        # 2. collision death
        if self.is_collision(agent.pos): # die from collision
            return True

        # 3. survival reward
        agent.scores['survived'] += 1

        # 4a. predator specific
        if agent._type == 'pred':
            agent.health -= 1
            if agent.health <= 0: # predator dies when not eaten in a while
                return True

            # reward when prey eaten
            for prey in self.preys.agents:
                if agent.pos == prey.pos:
                    if prey.moving:
                        agent.scores['killed'] += 1
                        agent.health = C.MAX_HEALTH.pred
                        agent.scores['eaten'] += 1
                        prey.moving = False
                        #prey.dead = True
                    else:
                        agent.scores['eaten'] += 1
                        hp_diff = C.MAX_HEALTH.pred - agent.health
                        food = min(hp_diff, prey.health)
                        prey.health -= food
                        agent.health += food

        # 4b. prey specific
        else:
            if agent.dead:
                return True
            elif not agent.moving:
                agent.health -= 1 # decay
                if agent.health <= 0: # die
                    agent.dead = False
                    return True
        
        # 5. return
        return False

    def is_collision(self, pt):
        return not self.point_exists(pt) or pt in self.obstacles # hits boundary or obstacle

    def _move(self, agent, action, a2=None, com_val=0):
        com = 0
        move = Point(0,0)

        if action==0: #L
            move = Point(-1,0)
        elif action==1: #R
            move = Point(1,0)
        elif action==2: #U
            move = Point(0,-1)
        elif action==3: #D
            move = Point(0,1)
        elif action==4:
            com = com_val
            #if a2==0:
            #    move = Point(1,0)
            #elif a2==1:
            #    move = Point(0,1)
            #elif a2==2:
            #    move = Point(-1,0)
            #elif a2==3:
            #    move = Point(0,-1) # or stand still
        elif action==5:
            pass # stand still

        newPos = Point(agent.pos.x + move.x, agent.pos.y + move.y)

        if not self.is_collision(newPos):
            agent.pos = newPos
        agent.com = abs(com) if C.ABSOLUTE_COM else com

    def get_closest_distances(self):
        distances = {'pred':[],'prey':[]}
        for pop, pop_type in [(self.preds.agents,'pred'),(self.preys.agents,'prey')]:
            for a in pop:
                pred_dist, prey_dist = -1,-1
                if len(self.preds.agents) > 1:
                    _, pred_dist = self.preds.get_closest_agent(a.pos,og_agent=a)
                if len(self.preys.agents) > 1:
                    _, prey_dist = self.preys.get_closest_agent(a.pos,og_agent=a)
                distances[pop_type].append({'pred':pred_dist,'prey':prey_dist})
        return distances

    def get_indiv_coms(self):
        coms = {'pred':[],'prey':[]}
        for pop, pop_type in [(self.preds.agents,'pred'),(self.preys.agents,'prey')]:
            if pop_type=='prey':
                continue
            for a in pop:
                state = a.get_state(self)[6:10]
                coms[pop_type].append(state)
        return coms

    '''
    def get_avg_distances(self):
        avg_distances = {'pred':{'pred':0,'prey':0},'prey':{'pred':0,'prey':0}}
        n_calcs = {'pred':{'pred':0,'prey':0},'prey':{'pred':0,'prey':0}}
        for pop1, pop_type1 in [(self.preds.agents,'pred'), (self.preys.agents,'prey')]:
            for pop2, pop_type2 in [(self.preds.agents,'pred'), (self.preys.agents,'prey')]:
                for a1 in pop1:
                    for a2 in pop2:
                        if a1._id == a2._id:
                            continue
                        avg_distances[pop_type1][pop_type2] += distance_between(a1.pos,a2.pos)
                        n_calcs[pop_type1][pop_type2] += 1
        avg_distances['pred']['pred'] /= max(n_calcs['pred']['pred'],1)
        avg_distances['pred']['prey'] /= max(n_calcs['pred']['prey'],1)
        avg_distances['prey']['prey'] /= max(n_calcs['prey']['prey'],1)
        avg_distances['prey']['pred'] /= max(n_calcs['prey']['pred'],1)
        return avg_distances
    '''