import numpy as np
import random
import os

from agents import Predator, Prey
import constants as C
from helper import distance_between
from pop_stats import PopStats
import torch as T

class Population:

    def __init__(self, _type):
        self._type = _type
        self.agents = []
        self.dead_agents = []
        
        self.genomes = []
        self.agent_index = 0
        self.stats = PopStats(_type)

        if C.LOAD_MODEL and self.files_exist():
            self.load_models()

    def populate(self,env):
        # Fills agent array with new agents of type
        self.agents = []
        self.repopulate(env)
        return
        
    def repopulate(self,env):

        #if self.is_preds() and C.LEARN.pred or self.is_preys() and C.LEARN.prey:
        #    return

        if self.is_preds():
            N = C.N_PRED
            LEARN = C.LEARN.pred
            MAX_N = C.POP_AMOUNT.pred
        elif self.is_preys():
            N = C.N_PREY
            LEARN = C.LEARN.prey
            MAX_N = C.POP_AMOUNT.prey

        while len(self.agents) < N:
            if self.agent_index >= MAX_N:
                return
            agent = Predator(env) if self.is_preds() else Prey(env)
            if len(self.genomes) > 0:
                idx = self.agent_index
                if not LEARN:
                    idx = int(random.random()*(MAX_N-1))
                agent.model.set_weights(self.genomes[idx])
            self.agent_index += 1
            self.agents.append(agent)

    def files_exist(self):
        model_folder_path = "./models/"+C.MODEL_NAME+"/"+self._type
        if(os.path.exists(model_folder_path)):
            file_name_mdl = os.path.join(model_folder_path, self._type+'0.mdl')
            file_name_stats = os.path.join(model_folder_path, 'stats_'+self._type+'.data')
            if os.path.exists(file_name_mdl) and os.path.exists(file_name_stats):
                return True
        return False

    def calc_fitness(self, agent):
        if self._type == 'pred':
            return agent.scores[C.FITNESS_FUNCTION.pred]
        else:
            return agent.scores[C.FITNESS_FUNCTION.prey]

    def finished(self):
        if self.is_preds():
            return self.agent_index >= C.POP_AMOUNT.pred
        else:
            return self.agent_index >= C.POP_AMOUNT.prey

    def have_sex(self):
        # 1. kill all agents still alive
        self.dead_agents.extend(self.agents)
        self.agents = []

        # 1.5 quit
        if not self.finished():
            return

        # 2. sort based on fitness
        total_score = sum([self.calc_fitness(a) for a in self.dead_agents])
        self.dead_agents.sort(key=lambda a: self.calc_fitness(a), reverse=True)

        # 3. normalize and make cumulative
        cum_score = 0
        fitness = []
        for i, a in enumerate(self.dead_agents):
            cum_score += self.calc_fitness(a)
            if total_score != 0:
                fitness.append(cum_score/total_score)
            else:
                fitness.append(0)

        #print("")
        #for i in fitness:
        #    if i != 1.0 or 0.0:
        #        print(np.round(i,2),end=' ')
        #print("")

        self.genomes = []
        for i in range(C.SUCCESSION_AMOUNT):
            self.genomes.append(self.dead_agents[i].model.get_weights())

        MAX_N = C.POP_AMOUNT.pred if self.is_preds() else C.POP_AMOUNT.prey

        while len(self.genomes) < MAX_N:

            # 4. select pair based on fitness
            selected_indices = (random.random(), random.random())
            selected_pair = [None, None]
            for i, a in enumerate(self.dead_agents):
                if selected_pair[0] is None and selected_indices[0] < fitness[i] and a != selected_pair[1]:
                    selected_pair[0] = a
                if selected_pair[1] is None and selected_indices[1] < fitness[i] and a != selected_pair[0]:
                    selected_pair[1] = a

            # 4.5. select at random if still None
            selected_random = [False, False]
            if selected_pair[0] is None: 
                for a in self.dead_agents:
                    if a is not selected_pair[1]:
                        selected_pair[0] = a
                        selected_random[0] = True
                        break
            if selected_pair[1] is None:   
                for a in self.dead_agents:
                    if a is not selected_pair[0]:
                        selected_pair[1] = a
                        selected_random[1] = True
                        break

            # 5. crossover genes
            if selected_pair[0] is not None and selected_pair[1] is not None:
                selected_genomes = (selected_pair[0].model.get_weights(), selected_pair[1].model.get_weights())

                new_genome = []
                if C.CROSSOVER_TYPE == 'split':
                    crossover_point = random.randint(1,len(selected_genomes[0])-1)
                    new_genome = selected_genomes[0][:crossover_point]
                    new_genome.extend(selected_genomes[1][crossover_point:])
                    #print(f"p1={np.round(genomes[0],2)}\np2={np.round(genomes[1],2)}\ncrossover={crossover_point}\nc={np.round(new_genome,2)}")
                elif C.CROSSOVER_TYPE == 'uniform':
                    for i in range(len(selected_genomes[0])):
                        new_genome.append(selected_genomes[0 if random.random() < 0.5 else 1][i])
                else:
                    raise ValueError("Unrecognised crossover type "+C.CROSSOVER_TYPE)

                # 6. mutation
                max_val, min_val = max(new_genome), min(new_genome)
                for i in range(len(new_genome)):
                    if random.random() < C.MUTATION_CHANCE:
                        #old_gene = new_genome[i]
                        new_genome[i] = -1+random.random()*2
                        #new_genome[i] = min_val + random.random() * abs(max_val)
                        #print(f"mutated {old_gene} to {new_genome[i]}")

                self.genomes.append(new_genome)
    
        # 7. stats
        self.stats.record_pop_scores(self)
        

        # 8. save models
        if C.SAVE_MODEL and (
            self.stats.is_record(C.FITNESS_FUNCTION.pred) or
            self.is_preys() and self.stats.is_record(C.FITNESS_FUNCTION.prey)):
            self.save_models(verbose=True)
            self.stats.save()

        # 9. clear vars
        self.dead_agents = []
        self.agent_index = 0

        return True

    def reset_same(self):
        self.agent_index -= len(self.agents)
        self.dead_agents.extend(self.agents)
        self.agents = []

        if not self.finished():
            return

        self.stats.record_pop_scores(self)

        self.dead_agents = []
        self.agent_index = 0

    def save_models(self,verbose=False):
        # saves population models from dead_agents
        model_folder_path = "./models/"+C.MODEL_NAME+"/"+self._type
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        if len(self.genomes) == 0:
            return
        n_saved = 0
        for i,gen in enumerate(self.genomes):
            a = Predator(None) if self._type == 'pred' else Prey(None)
            a.model.set_weights(self.genomes[i])
            file_name = self._type+str(i)+'.mdl'
            file_name = os.path.join(model_folder_path, file_name)
            a.model.save(file_name)
            n_saved += 1
        #save_id = 0
        #for array in [self.dead_agents, self.agents]:
        #    for agent in array:
        #        file_name = self._type+str(save_id)+'.mdl'
        #        file_name = os.path.join(model_folder_path, file_name)
        #        agent.model.save(file_name)
        #        save_id += 1
        print("S",end='')
        if verbose:
            print("aved "+str(n_saved)+" models of type "+self._type+".")

    def load_models(self):
        # load models to genomes array
        model_folder_path = "./models/"+C.MODEL_NAME+"/"+self._type
        POP_AMOUNT = C.POP_AMOUNT.pred if self.is_preds() else C.POP_AMOUNT.prey
        if(os.path.exists(model_folder_path)):
            agent = Predator() if self.is_preds() else Prey()
            for i in range(POP_AMOUNT):
                file_name = self._type+str(self.agent_index)+'.mdl'
                file_name = os.path.join(model_folder_path, file_name)
                if os.path.exists(file_name):
                    self.genomes.append(agent.model.load(file_name))
                else:
                    print("Loading Failed: model file "+file_name+" doesn't exist.")
                    return
        else:
            print("Loading Failed: folder "+model_folder_path+" doesn't exist.")
            return
        print(f"Loaded {POP_AMOUNT} models of type {self._type}.")

    def get_closest_agent(self, pt, og_agent=None):
        # Get closest agent of type
        closest = (None, 99999)
        for agent in self.agents:
            dist = distance_between(agent.pos, pt)
            if dist < closest[1] and (og_agent != agent):
                closest = (agent, dist)
            
        agent = closest[0]
        if agent is None:
            raise ValueError('Closest agent of type '+self._type+' is None')
        return agent, closest[1]

    def get_agent_by_id(self, _id):
        for agent in self.agents:
            if agent._id == _id:
                return agent
        return None

    def get_positions(self):
        # Return just agent positions
        positions = []
        for agent in self.agents:
            if agent is not None:
                positions.append(agent.pos)
        return positions

    def is_empty(self):
        return len(self.agents) == 0

    def get_amount(self):
        return len(self.agents)

    def remove_agent(self, index):
        self.dead_agents.append(self.agents[index])
        del self.agents[index]

    def is_preds(self):
        return self._type == 'pred'

    def is_preys(self):
        return self._type == 'prey'