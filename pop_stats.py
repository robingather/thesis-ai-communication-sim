import constants as C
from helper import save, load
from torch import is_tensor
import numpy as np

class PopStats:

    def __init__(self, _type):
        
        self._type = _type
        self.stats = {
            'scores':{'killed':[], 'eaten':[], 'survived':[]},
            'records':{'killed':0, 'eaten':0, 'survived':0},
            'it_gen':1, 'it_frames':1,
            'learn_switches':[], 'com_usage':[],
            # new
            'com_investigations':{'target_on_agent':[],'target_in_range_5':[],
            'target_in_range_5-10':[],'target_in_range_10-20':[],'target_out_range_20':[],
            't_l':[],'t_r':[],'t_u':[],'t_d':[], 'C_l>0':[],'C_r>0':[],'C_u>0':[],'C_d>0':[], 'all':[]},
            'x_usage':[]
            }
        self.gen_stats = {
            'scores':{'killed':[], 'eaten':[], 'survived':[]},
            'com_usage':[], 'avg_dist_over_time':[], 'com_value':[],
            'com_states':[], 'com_investigations':None,'com_actions':None
        }
        self.c_com_usage = 0
        self.c_frames = 0
        N_INPUTS = C.N_INPUTS.pred if self._type == 'pred' else C.N_INPUTS.prey
        self.c_com_states = [0 for i in range(N_INPUTS)]
        self.c_com_investigations = {'target_on_agent':[0,0],'target_in_range_5':[0,0],
            'target_in_range_5-10':[0,0],'target_in_range_10-20':[0,0],'target_out_range_20':[0,0],
        't_l':[0,0],'t_r':[0,0],'t_u':[0,0],'t_d':[0,0],
        'C_l>0':[0,0],'C_r>0':[0,0],'C_u>0':[0,0],'C_d>0':[0,0], 'all':[0,0]}
        self.c_avg_distances = {'pred':[],'prey':[]}
        self.c_action_coms = {'L':[0,0,0,0,0],'R':[0,0,0,0,0],'U':[0,0,0,0,0],'D':[0,0,0,0,0],'C':[0,0,0,0,0],'X':[0,0,0,0,0],'Any':[0,0,0,0,0]}
        self.c_com_actions = {'C_L>0':[0,0,0,0,0,0],'C_R>0':[0,0,0,0,0,0],'C_U>0':[0,0,0,0,0,0],
        'C_D>0':[0,0,0,0,0,0],'C>0':[0,0,0,0,0,0],'C==0':[0,0,0,0,0,0],'C_l<0':[0,0,0,0,0,0],'C_r<0':[0,0,0,0,0,0],'C_u<0':[0,0,0,0,0,0],
        'C_d<0':[0,0,0,0,0,0],'C<0':[0,0,0,0,0,0],'All':[0,0,0,0,0,0]}

        self.c_indiv_distances = [] # [it][agent]{'pred','prey'}
        self.c_indiv_coms = [] # [it][agent]{'C_r','C_l','C_u','C_d'}
        self.c_largest_com = []
        self.c_action_counts = [0,0,0,0,0,0]
        self.c_x_usage = 0

        if C.LOAD_MODEL:
            self.load()

    def record_indiv(self, distances, com):
        self.c_indiv_distances.append(distances)
        self.c_indiv_coms.append(com)
        '''
        avg_dist = {'pred':0,'prey':0}
        for pop_type in ['pred','prey']:
            for dists in distances:
                avg_dist[pop_type] += dists[pop_type]
            avg_dist[pop_type] /= len(dists) if len(dists) > 0 else 1
            self.c_avg_distances[pop_type].append(avg_dist[pop_type])
        '''

    def record_one_iteration(self, state, action, com, a2):
        if action==5:
            self.c_x_usage += 1
        if action==4 and com != 0:
            self.c_com_usage += 1
            target_dist = np.sqrt(state[0]**2+state[1]**2)
            if target_dist <= 5:
                self.c_com_investigations['target_on_agent'][0] += com
                self.c_com_investigations['target_on_agent'][1] += 1
            elif target_dist <= 5:
                self.c_com_investigations['target_in_range_5'][0] += com
                self.c_com_investigations['target_in_range_5'][1] += 1
            elif target_dist <= 10:
                self.c_com_investigations['target_in_range_5-10'][0] += com
                self.c_com_investigations['target_in_range_5-10'][1] += 1
            elif target_dist <= 20:
                self.c_com_investigations['target_in_range_10-20'][0] += com
                self.c_com_investigations['target_in_range_10-20'][1] += 1
            elif target_dist > 20:
                self.c_com_investigations['target_out_range_20'][0] += com
                self.c_com_investigations['target_out_range_20'][1] += 1
            if state[0] > 0:
                self.c_com_investigations['t_r'][0] += com
                self.c_com_investigations['t_r'][1] += 1
            if state[0] < 0:
                self.c_com_investigations['t_l'][0] += com
                self.c_com_investigations['t_l'][1] += 1
            if state[1] > 0:
                self.c_com_investigations['t_d'][0] += com
                self.c_com_investigations['t_d'][1] += 1
            if state[1] < 0:
                self.c_com_investigations['t_u'][0] += com
                self.c_com_investigations['t_u'][1] += 1
            if state[6] > 0:
                self.c_com_investigations['C_l>0'][0] += com
                self.c_com_investigations['C_l>0'][1] += 1
            if state[7] > 0:
                self.c_com_investigations['C_r>0'][0] += com
                self.c_com_investigations['C_r>0'][1] += 1
            if state[8] > 0:
                self.c_com_investigations['C_u>0'][0] += com
                self.c_com_investigations['C_u>0'][1] += 1
            if state[9] > 0:
                self.c_com_investigations['C_d>0'][0] += com
                self.c_com_investigations['C_d>0'][1] += 1
            self.c_com_investigations['all'][0] += com
            self.c_com_investigations['all'][1] += 1

        act_vec = [0 for i in range(C.N_INPUTS.pred)]
        act_vec[action] = 1
        
        lbl = None
        if state[6] > 0:
            self.c_com_actions['C_L>0'][action] += 1
        if state[6] < 0:
            self.c_com_actions['C_l<0'][action] += 1
        if state[7] > 0:
            self.c_com_actions['C_R>0'][action] += 1
        if state[7] < 0:
            self.c_com_actions['C_r<0'][action] += 1
        if state[8] > 0:
            self.c_com_actions['C_U>0'][action] += 1
        if state[8] < 0:
            self.c_com_actions['C_u<0'][action] += 1
        if state[9] > 0:
            self.c_com_actions['C_D>0'][action] += 1
        if state[9] < 0:
            self.c_com_actions['C_d<0'][action] += 1
        if state[6]+state[7]+state[8]+state[9] > 0:
            self.c_com_actions['C>0'][action] += 1
        if state[6]+state[7]+state[8]+state[9] < 0:
            self.c_com_actions['C<0'][action] += 1
        if state[6]+state[7]+state[8]+state[9] == 0:
            self.c_com_actions['C==0'][action] += 1
        self.c_com_actions['All'][action] += 1

        '''
        if state[7] > state[6] and state[7] > state[8] and state[7] > state[9]:
            self.c_com_actions['C_r>C'][0] = [a + b for a, b in zip(self.c_com_actions['C_r>0'][0], act_vec)] 
            self.c_com_actions['C_r>C'][1] += 1
        if state[8] > state[7] and state[8] > state[6] and state[8] > state[9]:
            self.c_com_actions['C_u>C'][0] = [a + b for a, b in zip(self.c_com_actions['C_u>0'][0], act_vec)] 
            self.c_com_actions['C_u>C'][1] += 1
        if state[9] > state[7] and state[9] > state[6] and state[9] > state[8]:
            self.c_com_actions['C_d>C'][0] = [a + b for a, b in zip(self.c_com_actions['C_d>0'][0], act_vec)] 
            self.c_com_actions['C_d>C'][1] += 1
        if state[6] < state[7] and state[6] < state[8] and state[6] < state[9]:
            self.c_com_actions['C_l<C'][0] = [a + b for a, b in zip(self.c_com_actions['C_l>0'][0], act_vec)] 
            self.c_com_actions['C_l<C'][1] += 1
        if state[7] < state[6] and state[7] < state[8] and state[7] < state[9]:
            self.c_com_actions['C_r<C'][0] = [a + b for a, b in zip(self.c_com_actions['C_r>0'][0], act_vec)] 
            self.c_com_actions['C_r<C'][1] += 1
        if state[8] < state[7] and state[8] < state[6] and state[8] < state[9]:
            self.c_com_actions['C_u<C'][0] = [a + b for a, b in zip(self.c_com_actions['C_u>0'][0], act_vec)] 
            self.c_com_actions['C_u<C'][1] += 1
        if state[9] < state[7] and state[9] < state[6] and state[9] < state[8]:
            self.c_com_actions['C_d<C'][0] = [a + b for a, b in zip(self.c_com_actions['C_d>0'][0], act_vec)] 
            self.c_com_actions['C_d<C'][1] += 1
        if state[9]+state[8]+state[7]+state[6] > 0:
            self.c_com_actions['C>0'][0] = [a + b for a, b in zip(self.c_com_actions['C>0'][0], act_vec)] 
            self.c_com_actions['C>0'][1] += 1
        if state[9]+state[8]+state[7]+state[6] < 0:
            self.c_com_actions['C<0'][0] = [a + b for a, b in zip(self.c_com_actions['C<0'][0], act_vec)] 
            self.c_com_actions['C<0'][1] += 1
        if state[9]+state[8]+state[7]+state[6] == 0:
            self.c_com_actions['C==0'][0] = [a + b for a, b in zip(self.c_com_actions['C==0'][0], act_vec)] 
            self.c_com_actions['C==0'][1] += 1
        '''
        com_idx = np.argmax([state[6],state[7],state[8],state[9]])

        for a_id, a_lbl in [(0,'L'),(1,'R'),(2,'U'),(3,'D'),(4,'C'),(5,'X')]:
            if action==a_id:
                self.c_action_counts[a_id] += 1
                for i in range(4):
                    self.c_action_coms[a_lbl][i] += state[6+i]
                    self.c_action_coms['Any'][i] += state[6+i]

        self.c_frames += 1
            

    def record_pop_scores(self, pop):
        scores = self.stats['scores']
        scores['killed'].append(0)
        scores['eaten'].append(0)
        scores['survived'].append(0)
        for a in pop.dead_agents:
            scores['killed'][-1] += a.scores['killed']
            scores['eaten'][-1] += a.scores['eaten']
            scores['survived'][-1] += a.scores['survived']
        scores['killed'][-1] /= len(pop.dead_agents)
        scores['eaten'][-1] /= len(pop.dead_agents)
        scores['survived'][-1] /= len(pop.dead_agents)

        #for key in self.c_com_investigations.keys():
        #    self.stats['com_investigations'][key].append(self.c_com_investigations[key])
        
        self.iterate('it_gen')
        self.stats['com_usage'].append(self.c_com_usage/self.c_frames)
        #self.stats['x_usage'].append(self.c_x_usage/self.c_frames)
        self.c_com_usage = 0
        self.c_x_usage = 0
        self.c_frames = 0
        self.c_indiv_distances = []
        self.c_indiv_coms = [] 
        N_INPUTS = C.N_INPUTS.pred if self._type == 'pred' else C.N_INPUTS.prey
        self.c_com_states = [0 for i in range(N_INPUTS)]
        if not C.PLOT_LAST_GEN:
            self.c_com_investigations = {'target_on_agent':[0,0],'target_in_range_5':[0,0],
                'target_in_range_5-10':[0,0],'target_in_range_10-20':[0,0],'target_out_range_20':[0,0],
            't_l':[0,0],'t_r':[0,0],'t_u':[0,0],'t_d':[0,0],
        'C_L>0':[0,0],'C_R>0':[0,0],'C_U>0':[0,0],'C_D>0':[0,0], 'all':[0,0]}
        #self.c_com_actions = {'C_l>0':[[0,0,0,0,0,0],0],'C_r>0':[[0,0,0,0,0,0],0],'C_u>0':[[0,0,0,0,0,0],0],
        #'C_d>0':[[0,0,0,0,0,0],0],'C>0':[[0,0,0,0,0,0],0],'C<0':[[0,0,0,0,0,0],0],'C==0':[[0,0,0,0,0,0],0]}
        self.c_avg_distances = {'pred':[],'prey':[]}

        self.update_records()

    def record_current_gen_scores(self, env):
        scores = {'killed':0, 'eaten':0, 'survived':0}
        coms, com_use = 0, 0
        agents = env.preds.agents if self._type == 'pred' else env.preys.agents
        for a in agents:
            scores['killed'] += a.scores['killed']
            scores['eaten'] += a.scores['eaten']
            scores['survived'] += a.scores['survived']
            coms += a.com.item() if is_tensor(a.com) else a.com
            com_use += 1 if a.com != 0 else 0
        self.gen_stats['scores']['killed'].append(scores['killed'])
        self.gen_stats['scores']['eaten'].append(scores['eaten'])
        self.gen_stats['scores']['survived'].append(scores['survived'])

        self.gen_stats['com_value'].append(coms)
        self.gen_stats['com_usage'].append(com_use)
        self.gen_stats['avg_dist_over_time'] = self.c_avg_distances

        self.gen_stats['com_states'] = self.c_com_states
        self.gen_stats['com_investigations'] = self.c_com_investigations
        


    def reset_current_gen_scores(self):
        #print(self.c_com_states)
        self.gen_stats = {
            'scores':{'killed':[], 'eaten':[], 'survived':[]},
            'com_usage':[], 'avg_dist_over_time':[], 'com_value':[],
            'com_states':[], 'com_investigations':None,'com_actions':None
        }


    def update_records(self):
        scores, records = self.stats['scores'], self.stats['records']
        for key in scores:
            if len(scores[key]) == 0:
                continue
            if scores[key][-1] > records[key]:
                records[key] = scores[key][-1]

    def is_record(self, key):
        if self.is_empty(key):
            return False
        return self.stats['scores'][key][-1] == self.stats['records'][key]

    def save(self):
        folder_path = "./models/"+C.MODEL_NAME+"/"+self._type
        file_name = 'stats_'+self._type+'.data'
        save(folder_path, file_name, self.stats)
        print("s",end='')

    def load(self):
        folder_path = "./models/"+C.MODEL_NAME+"/"+self._type
        file_name = 'stats_'+self._type+'.data'
        loaded_stats = load(folder_path, file_name)
        if loaded_stats is not None:
            self.stats = loaded_stats
            print("Loaded Stats ("+self._type+")")

    def iterate(self, key):
        self.stats[key] += 1

    def get_recent_score(self, key):
        if self.is_empty():
            return None
        return self.stats['scores'][key][-1]

    def get(self, key):
        return self.stats[key]

    def is_empty(self,key):
        return len(self.stats['scores'][key]) == 0

    def is_plottable(self,key):
        return len(self.stats['scores'][key]) > 1