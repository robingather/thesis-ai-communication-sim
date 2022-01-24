from os import kill
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib as mpl
import pygame
import seaborn as sns

from helper import Point, distance_between
import constants as C

# rgb colors
WHITE = (250,250,250)
LIGHT_BLUE = (167, 148, 246)
LIGHT_RED = (242, 121, 125)
LIGHTER_RED = (252, 198, 164)
DARK_RED = (100, 0, 0)
LIGHT_GREEN = (146,247,173)
LIGHTER_PURPLE = (247, 152, 250)
DARK_GREEN = (28,169,66)
BLACK = (0,0,0)
BOTTOM_BG = (0, 36, 81)
FADED_RED = (200,220,168)
DEEP_RED = (51, 102, 242)

pygame.init()
font = pygame.font.Font('ROBOTOSLAB-REGULAR.ttf', 14)
font_s = pygame.font.Font('ROBOTOSLAB-REGULAR.ttf', 10)

class Renderer:

    BOTTOM_SLOTS = 6
    BOTTOM_MARGIN = BOTTOM_SLOTS*25

    def __init__(self,w,h):
        self.w = w*C.BLOCK_SIZE
        self.h = h*C.BLOCK_SIZE
        self.h_full = self.h + self.BOTTOM_MARGIN
        self.display = pygame.display.set_mode((self.w,self.h_full))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('PredAItor')
        sns.set_style("darkgrid")

    def tick(self):
        self.clock.tick(C.SPEED)

    def render(self, env):

        self.render_world(env)
        
        #self.render_stats(env,stats)

        self.render_bottom(env)

        pygame.display.flip()

    def render_world(self, env):
        self.display.fill(LIGHT_GREEN, pygame.Rect(0,0,self.w,self.h))
        BS = C.BLOCK_SIZE
        mBS = BS/4

        for pred in env.preds.agents:
            if pred.com != 0 and C.RENDER_COMMUNICATION:
                self.render_circle(env, pred.pos, C.RAD_COM)

        for i, pred in enumerate(env.preds.agents):
            agent_color = LIGHT_RED#self.mix_colors(LIGHT_RED, LIGHT_GREEN, pred.health/C.MAX_HEALTH.pred)
            if i==0:
                agent_color = DEEP_RED
            pygame.draw.rect(self.display, agent_color, pygame.Rect(pred.pos.x*BS, pred.pos.y*BS, BS, BS))

            if env.preys.get_amount() == 0:
                break

            state = pred.get_state(env)
            if state[2] == 1:
                pygame.draw.rect(self.display, LIGHTER_RED, pygame.Rect(pred.pos.x*BS,pred.pos.y*BS, mBS, BS))
            if state[3] == 1:
                pygame.draw.rect(self.display, LIGHTER_RED, pygame.Rect(pred.pos.x*BS+BS-mBS,pred.pos.y*BS, mBS, BS))
            if state[4] == 1:
                pygame.draw.rect(self.display, LIGHTER_RED, pygame.Rect(pred.pos.x*BS,pred.pos.y*BS, BS, mBS))
            if state[5] == 1:
                pygame.draw.rect(self.display, LIGHTER_RED, pygame.Rect(pred.pos.x*BS,pred.pos.y*BS+BS-mBS, BS, mBS))
            target = env.preys.get_agent_by_id(pred.target_id)
            if target != None:
               pygame.draw.line(self.display, LIGHTER_RED, (pred.pos.x*BS+BS/2,pred.pos.y*BS+BS/2), (pred.pos.x*BS+BS/2, target.pos.y*BS+BS/2))
               pygame.draw.line(self.display, LIGHTER_RED, (pred.pos.x*BS+BS/2,target.pos.y*BS+BS/2), (target.pos.x*BS+BS/2, target.pos.y*BS+BS/2))

            # health text
            #text_pos = Point(pred.pos.x*BS, pred.pos.y*BS-BS/8)
            #self.display.blit(font_s.render(str(pred.health), True, BLACK), [text_pos.x,text_pos.y])

        for obst in env.obstacles:
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(obst.x*BS, obst.y*BS, BS, BS))

        for prey in env.preys.agents:
            agent_color = self.mix_colors(LIGHT_BLUE, LIGHT_GREEN, prey.health/C.MAX_HEALTH.prey)
            pygame.draw.rect(self.display, agent_color, pygame.Rect(prey.pos.x*BS, prey.pos.y*BS, BS, BS))

    def render_circle(self, env, pos, r):
        BS = C.BLOCK_SIZE
        pygame.draw.circle(self.display, FADED_RED, Point(pos.x*BS,pos.y*BS), r*BS)
        return
        BS = C.BLOCK_SIZE
        iR = int(r)+1
        for y_i in range(pos.y-iR,pos.y+iR):
            for x_i in range(pos.x-iR,pos.x+iR):
                pt = Point(x_i,y_i)
                if not env.point_exists(pt) or distance_between(env, pos, pt, asblocks=True) > r:
                    continue
                pygame.draw.rect(self.display, LIGHTER_RED, pygame.Rect(x_i*BS, y_i*BS, BS, BS))

    def render_stats(self, env, stats):
        pred_scores = []
        total_score = 0
        for pred in env.preds.agents:
            pred_scores.append(pred.scores['killed'])
            total_score += pred.scores['killed']

        self.render_text(f"Gen: {stats.get('it_gen')}", 0)
        self.render_text(f"Score: {pred_scores} -> {total_score}", 1)
        self.render_text(f"N: {env.preds.get_amount()} / {env.preys.get_amount()}", 2)
        if env.preds.get_amount() > 0:
            _, vals = env.preds.agents[0].get_action(env.preds.agents[0].get_state(env))
            self.render_text(f"Vals: {np.round(vals.cpu().detach().numpy(),3)}",3)

    def render_bottom(self, env):
        self.display.fill(BOTTOM_BG, pygame.Rect(0,self.h,self.w,self.BOTTOM_MARGIN))

        pred_score = 0
        for pred in env.preds.agents:
            if pred.scores['killed'] > pred_score:
                pred_score = pred.scores['killed']

        prey_score = 0
        for prey in env.preys.agents:
            if prey.scores['survived'] > prey_score:
                prey_score = prey.scores['survived']

        speed_string = 'Slow' if C.SPEED < 10 else ('Med' if C.SPEED < 100 else 'Max')
        dist_pred = env.preds.stats.c_avg_distances['pred']
        dist_pred = np.round(dist_pred[-1],1) if len(dist_pred) > 0 else -1
        dist_prey = env.preys.stats.c_avg_distances['prey']
        dist_prey = np.round(dist_prey[-1],1) if len(dist_prey) > 0 else -1

        values = [
            'PREDATORS STATS',
            'Gen: '+str(env.preds.stats.get('it_gen')),
            'Batch: '+str(env.preds.agent_index-C.N_PRED)+'-'+str(env.preds.agent_index)+' of '+str(C.POP_AMOUNT.pred),
            'Learning: '+str(C.LEARN.pred)+' (L)',
            'Comms: '+str(C.COMMUNICATE_WITHIN_POP.pred)+' (T)',
            'Hear Preys: '+str(C.HEAR_BETWEEN_POP.pred)+' (Y)',

            '',
            'Score: '+str(pred_score),
            'Record: '+str(np.round(env.preds.stats.stats['records']['killed'],2)),
            'N Alive: '+str(env.preds.get_amount()),
            'Frame: '+str(self.human_format(env.preds.stats.get('it_frames'))),
            'Avg Dist: '+str(dist_pred),

            'PREYS STATS',
            'Gen: '+str(env.preys.stats.get('it_gen')),
            'Batch: '+str(env.preys.agent_index-C.N_PRED)+'-'+str(env.preys.agent_index)+' of '+str(C.POP_AMOUNT.prey),
            'Learning: '+str(C.LEARN.prey)+' (O)',
            'Comms: '+str(C.COMMUNICATE_WITHIN_POP.prey)+' (U)',
            'Hear Preds: '+str(C.HEAR_BETWEEN_POP.prey)+' (I)',
            '',
            'Score: '+str(prey_score),
            'Record: '+str(np.round(env.preys.stats.stats['records']['survived'],2)),
            'N Alive: '+str(env.preys.get_amount()),
            'Frame: '+str(self.human_format(env.preys.stats.get('it_frames'))),
            'Avg Dist: '+str(dist_prey),

            'MISC STATS',
            'Render: '+str(True)+' (Tab)',
            'Render Com: '+str(C.RENDER_COMMUNICATION)+' (C)',
            'Speed: '+speed_string+' (< >)',
            'Save: '+str(C.SAVE_MODEL)+' (X/S)',

        ]

        val_idx = 0
        for i in range(5):
            for j in range(self.BOTTOM_SLOTS):
                if val_idx < len(values):
                    value = values[val_idx]
                    txt_y = self.h+8+22*j
                    txt_x = 8+180*i
                    self.display.blit(font.render(value, True, WHITE), [txt_x, txt_y])
                    val_idx += 1

    def render_text(self, text, i):
        text = font.render(text, True, BLACK)
        self.display.blit(text, [5, 5+i*20])
        
    def human_format(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def plot_com(self, env):
        plt.ion()
        plt.figure(8,figsize=(18,6))
        plt.clf()
        plt.tight_layout()
        plt.suptitle('Communication Stats')
        plt.subplots_adjust(wspace=0.55, hspace=0.55)
        fig, axs = plt.subplots(2,4,num=8)
        fig.canvas.set_window_title(C.MODEL_NAME+", Communication Stats")

        cases=list(env.preds.stats.gen_stats['com_investigations'].keys())
        vals=list(env.preds.stats.gen_stats['com_investigations'].values())
        summed_usage = 0
        for v in vals:
            summed_usage += v[1]
        for i in range(len(vals)):
            vals[i] = (vals[i][1] / summed_usage) if summed_usage > 0 else 0

        ax = axs[0,0]
        ax.set_xlabel('Com Usage')
        ax.set_ylabel('Case')
        cases=list(env.preds.stats.gen_stats['com_investigations'].keys())
        vals=list(env.preds.stats.gen_stats['com_investigations'].values())
        summed_usage = 0
        for v in vals:
            summed_usage += v[1]
        ax.set_title('Com Usage For Cases (N='+str(self.human_format(int(summed_usage)))+')')
        for i in range(len(vals)):
            vals[i] = (vals[i][1] / summed_usage) if summed_usage > 0 else 0
        plt.sca(ax)
        sns.barplot(x=vals,y=cases)

        ax = axs[1,0]
        ax.set_title('Avg Com Value For Cases')
        ax.set_xlabel('Avg Com Value')
        ax.set_ylabel('Case')
        cases=list(env.preds.stats.gen_stats['com_investigations'].keys())
        vals=list(env.preds.stats.gen_stats['com_investigations'].values())
        for i in range(len(vals)):
            vals[i] = (vals[i][0]/vals[i][1]) if vals[i][1] > 0 else 0
        plt.sca(ax)
        sns.barplot(x=vals,y=cases)

        vals = env.preds.stats.c_action_coms
        any_tot_amount = sum(vals['Any'][:-1])
        any_a_vals = [0,0,0,0]
        for i in range(4):
            any_a_vals[i] = vals['Any'][i]/(any_tot_amount if any_tot_amount > 0 else 1)

        axes = [[0,1],[1,1],[0,2],[1,2],[0,3],[1,3]]
        for i,key in enumerate(vals.keys()):
            if key == 'Any':
                continue
            ax = axs[axes[i][0],axes[i][1]]
            
            tot_amount = sum(vals[key][:-1])
            a_vals = [0,0,0,0]
            for i in range(4):
                a_vals[i] = (vals[key][i]/(tot_amount if tot_amount > 0 else 1))#-any_a_vals[i]

            plt.sca(ax)
            keys = ['C_L','C_R','C_U','C_D']
            ax.set_title('Com Inputs Influence for Action '+key+' (N='+str(self.human_format(int(tot_amount)))+")")
            ax.set_xlabel('Difference from Avg')
            ax.set_ylabel('Com Input')
            
            #ax.set_xlim(-0.8,0.8)
            print("Action DIST ("+key+")")
            print(keys,a_vals)
            sns.barplot(x=a_vals,y=keys)
            #xabs_max = abs(max(ax.get_xlim(), key=abs))
            #ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)

        # R raw
        '''
        key = 'R'
        ax = axs[0,3]
        tot_amount = sum(vals[key][:-1])
        a_vals = [0,0,0,0]
        for i in range(4):
            a_vals[i] = (vals[key][i]/(tot_amount if tot_amount > 0 else 1))

        plt.sca(ax)
        keys = ['C_L','C_R','C_U','C_D']
        ax.set_title('Com Input Distribution for Action '+key+' (N='+str(self.human_format(int(tot_amount)))+")")
        ax.set_xlabel('Prominence')
        ax.set_ylabel('Com Input')
        
        ax.set_xlim(0,1)
        
        sns.barplot(x=a_vals,y=keys)

        ax = axs[1,3]
        plt.sca(ax)
        keys = ['C_L','C_R','C_U','C_D']
        ax.set_title('Com Input Distribution for Any Action (N='+str(self.human_format(int(any_tot_amount)))+")")
        ax.set_xlabel('Prominence')
        ax.set_ylabel('Com Input')
        
        ax.set_xlim(0,1)
        print("Action DIST (Any)")
        print(keys,any_a_vals)
        sns.barplot(x=any_a_vals,y=keys)

        print("Action counts")
        print(env.preds.stats.c_action_counts)
        '''
        print("Action counts")
        print(env.preds.stats.c_action_counts)


        #xabs_max = abs(max(ax.get_xlim(), key=abs))
        #ax.set_xlim(xmin=-xabs_max, xmax=xabs_max)


        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)

    def plot_preds(self, env):
        plt.ion()
        plt.figure(12,figsize=(8,8))
        plt.clf()
        plt.tight_layout()
        plt.suptitle('Pred Distances and Coms')
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        fig, axs = plt.subplots(4,4,num=12)
        fig.canvas.set_window_title(C.MODEL_NAME+", Distance / Coms")

        pop_dists = env.preds.stats.c_indiv_distances
        pop_coms = env.preds.stats.c_indiv_coms

        for i in range(4):
            for j in range(4):
                k = i*4+j
                ax = axs[i,j]
                coms = []
                dists = []
                for l in range(len(pop_dists)):
                    if len(pop_dists) == 0 or l >= len(pop_dists) or k >= len(pop_dists[l]):
                        return
                    #print(l,k)
                    dists.append(pop_dists[l][k]['pred'])
                    tot_com = pop_coms[l][k][0] + pop_coms[l][k][1] + pop_coms[l][k][2] + pop_coms[l][k][3]
                    coms.append(tot_com > 0)
                for l, val in enumerate(coms):
                    if val:
                        ax.axvline(l, color='lightgray')
                sns.lineplot(x=range(1,len(dists)+1),y=dists,ax=ax)
                ax.set_title(f'pred {k} pred-pred dist')

        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)

    def plot_gen(self, env):

        self.plot_com(env)
        #self.plot_preds(env)

        plt.ion()
        plt.figure(2,figsize=(14,8))
        plt.clf()
        plt.tight_layout()
        plt.suptitle('Distance / Iteration Stats')
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        fig, axs = plt.subplots(2,2,num=2)
        fig.canvas.set_window_title(C.MODEL_NAME+", Distance / Iteration Stats")

        ax = axs[0,0]
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Prey killed',color='mediumblue')
        scores = env.preds.stats.gen_stats['scores']['killed']
        sns.lineplot(x=range(1,len(scores)+1),y=scores,ax=ax)
        ax.set_title('Predator Score')

        ax = axs[1,0]
        ax.set_xlabel('Iteration')
        ax.set_ylabel('\% of pred actions',color='purple')
        usage_pred = env.preds.stats.gen_stats['com_usage']
        
        sns.lineplot(x=range(1,len(usage_pred)+1),y=usage_pred,ax=ax,color='purple')

        ax = axs[1,0].twinx()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('avg com per pred',color='purple')
        value_pred = env.preds.stats.gen_stats['com_value']
        sns.lineplot(x=range(1,len(value_pred)+1),y=value_pred,ax=ax,color='pink')
        
        ax = axs[0,1]
        ax.set_xlabel('Iteration')
        ax.set_title('Pred Distance')
        ax.set_ylabel('Avg Normalized Distance')
        last_dists = env.preds.stats.gen_stats['avg_dist_over_time']
        #print(last_dists)
        sns.lineplot(x=range(1,len(last_dists['pred'])+1),y=last_dists['pred'],ax=ax,color='red')
        sns.lineplot(x=range(1,len(last_dists['prey'])+1),y=last_dists['prey'],ax=ax,color='orange')
        ax.legend(['pred-pred','pred-prey'])

        ax = axs[1,1]
        pop_dists = env.preds.stats.c_indiv_distances
        pop_coms = env.preds.stats.c_indiv_coms
        coms = []
        dists = []
        for i in range(len(pop_dists)):
            dists.append(pop_dists[i][0]['pred'])
            tot_com = pop_coms[i][0][0] + pop_coms[i][0][1] + pop_coms[i][0][2] + pop_coms[i][0][3]
            coms.append(tot_com > 0)
        for i, val in enumerate(coms):
            if val:
                ax.axvline(i, color='lightgray')
        sns.lineplot(x=range(1,len(dists)+1),y=dists,ax=ax)
        ax.set_title('pred 0 pred-pred dist')
        
        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)
        

    def plot(self, env):
        
        if not env.preds.stats.is_plottable('killed'):
            return

        plt.ion()
        plt.figure(1,figsize=(5,7))
        plt.clf()
        plt.tight_layout()
        plt.suptitle('Generation Stats')
        plt.subplots_adjust(wspace=0.35, hspace=0.45)
        fig, axs = plt.subplots(3,1,num=1)
        fig.canvas.set_window_title(C.MODEL_NAME+", Generation Stats")

        # pred score
        ax = axs[0]
        ax.set_xlabel('Generation')
        ax.set_ylabel('Prey eaten',color='orange')
        scores = env.preds.stats.stats['scores']['eaten']
        print("SCORES")
        print(scores)
        sns.lineplot(x=range(1,len(scores)+1),y=scores,ax=ax,color='orange')
        ax.set_title('Predator Score')
        #for pred_gen, prey_gen, val in env.preds.stats.stats['learn_switches']:
        #    ax.axvline(pred_gen, color='green' if val else 'red')
        #for pred_gen, prey_gen, val in env.preys.stats.stats['learn_switches']:
        #    ax.axvline(pred_gen, color='palegreen' if val else 'lightpink')
        

        #ax = axs[0,0].twinx()
        #ax.set_ylabel('Prey eaten',color='orange')
        #usage = env.preds.stats.stats['scores']['eaten']
        #sns.lineplot(x=range(1,len(usage)+1),y=usage,ax=ax,color='orange')
        # coms
        #ax = axs[0].twinx()
        #ax.set_ylabel('Com Usage',color='orange')
        #usage = env.preds.stats.stats['com_usage']
        #sns.lineplot(x=range(1,len(usage)+1),y=usage,ax=ax,color='orange')

        # prey score
        #ax = axs[1]
        #ax.set_xlabel('Generation')
        #ax.set_ylabel('Score',color='mediumblue')
        #scores = env.preys.stats.stats['scores']['survived']
        #sns.lineplot(x=range(1,len(scores)+1),y=scores,ax=ax)
        #ax.set_title('Prey Score')
        # com cases
        
        

        #for pred_gen, prey_gen, val in env.preys.stats.stats['learn_switches']:
        #    ax.axvline(prey_gen, color='green' if val else 'red')
        #for pred_gen, prey_gen, val in env.preds.stats.stats['learn_switches']:
        #    ax.axvline(prey_gen, color='palegreen' if val else 'lightpink')

        # comms
        #ax = axs[1].twinx()
        #ax.set_ylabel('Com Usage',color='orange')
        #usage = env.preys.stats.stats['com_usage']
        #sns.lineplot(x=range(1,len(usage)+1),y=usage,ax=ax,color='orange')

        ax = axs[2]
        ax.set_xlabel('Generation')
        ax.set_ylabel('\% of pred actions',color='purple')

        usage_pred = env.preds.stats.stats['com_usage']
        print("USAGE")
        print(usage_pred)
        sns.lineplot(x=range(1,len(usage_pred)+1),y=usage_pred,ax=ax,color='purple')

        usage_pred = env.preds.stats.stats['x_usage']
        sns.lineplot(x=range(1,len(usage_pred)+1),y=usage_pred,ax=ax,color='pink')

        '''
        ax = axs[2].twinx()
        ax.set_ylabel('\% of prey actions',color='pink')
        ax.set_xlabel('Generation',color='pink')
        ax.tick_params(axis='x',color='pink')
        usage_prey = env.preys.stats.stats['com_usage']
        sns.lineplot(x=list(np.linspace(1,len(usage_pred)+1,len(usage_prey))),y=usage_prey,ax=ax,color='pink')
        ax.set_title('Coms usage')'''

        '''
        ax = axs[0,1]
        ax.set_title('Avg Pred States (stacked)')
        ax.set_xlabel('State')
        ax.set_ylabel('Ratio')
        com_states={'com':env.preds.stats.stats['com_states'][-1]['com'],\
            'no_com':env.preds.stats.stats['com_states'][-1]['no_com']}
        x = range(len(com_states['com']))
        #print(com_states)
        ax.bar(x,com_states['no_com'])
        ax.bar(x,com_states['com'],bottom=com_states['no_com'])
        ax.set_xticks(ticks=x)
        ax.set_xticklabels(labels=C.STATE_LABELS)
        
        ax.legend(['no com','com'])
        ax.set_ylim(ymin=0)
        
        ax = axs[1,1]
        ax.set_xlabel('Iteration')
        ax.set_title('Pred Distance')
        ax.set_ylabel('Avg Normalized Distance')
        last_dists = env.preds.stats.stats['avg_dist_over_time'][-1]
        #print('r',len(last_dists['pred']))
        #print(last_dists['pred'])
        sns.lineplot(x=range(1,len(last_dists['pred'])+1),y=last_dists['pred'],ax=ax,color='red')
        sns.lineplot(x=range(1,len(last_dists['prey'])+1),y=last_dists['prey'],ax=ax,color='orange')
        ax.legend(['pred-pred','pred-prey'])


        ax = axs[2,1]
        ax.set_xlabel('Iteration')
        ax.set_title('Prey Distance')
        ax.set_ylabel('Avg Normalized Distance')
        if len(env.preys.stats.stats['avg_dist_over_time']) > 0:
            last_dists = env.preys.stats.stats['avg_dist_over_time'][-1]
            sns.lineplot(x=range(1,len(last_dists['prey'])+1),y=last_dists['prey'],ax=ax,color='purple')
            sns.lineplot(x=range(1,len(last_dists['pred'])+1),y=last_dists['pred'],ax=ax,color='pink')
            ax.legend(['prey-prey','prey-pred'])
        '''
        plt.ioff()

        # 3d distance
        #ax = fig.add_subplot(111, projection = '3d')

        # convert to plane
        #dists = np.array([np.array(d['pred']) for d in env.preds.stats.stats['avg_dist_over_time']])

        #length = (np.vectorize(len)(dists)).max()
        #dists = np.array([np.pad(d, (0, length-len(d)), 'constant') for d in dists])
        
        #avg_y = np.mean(dists,axis=0) # average distance per iteration across gens
        #x = range(1,len(avg_y)+1) # iteration range


        #ax = axs[3,1]
        #ax.set_title('Pred Distances Across Gens')
        #img = ax.imshow(dists,interpolation='nearest',cmap = 'hot',aspect='auto')
        #plt.sca(ax)
        #plt.colorbar(img,label='Avg Pred-Pred Distance')


        # plot plane
        #for i in range(min(len(dists_pred),10)):
        #    ax.plot(xs = range(1, len(dists_pred[i])+1), ys=dists_pred[i], zs=i)
        #ax.set_xlabel('Iteration')
        #ax.set_ylabel('World')
        #ax.set_zlabel('Generation')
        #ax.view_init(90+30, 270)
        #ax.plot(range(1,len(last_dists['pred'])+1), last_dists['pred'], )

        '''
        ax = axs[3,0]
        ax.set_title('Coms/Behavior Corr.')
        ax.set_xlabel('Generation')      
        ax.set_yticks([])
        usage_pred = env.preds.stats.stats['com_usage']
        p1 = sns.lineplot(x=range(1,len(usage_pred)+1),y=usage_pred,ax=ax,color='purple',label='com')

        ax = axs[3,0].twinx() 
        ax.set_xticks([])
        ax.set_yticks([])
        #dist_gens = np.mean(dists,axis=1)
        #p2 = sns.lineplot(x=range(1,len(dist_gens)+1),y=dist_gens,ax=ax,color='red',label='dist')

        ax = axs[3,0].twinx()
        ax.set_xticks([])
        ax.set_yticks([])
        scores = env.preds.stats.stats['scores']['killed']
        p3 = sns.lineplot(x=range(1,len(scores)+1),y=scores,ax=ax, color='mediumblue',label='score')
        #ax.legend(handles=[p1,p2,p3])
        #fig.canvas.manager.window.move(0,0)
        '''
        plt.show(block=False)
        plt.pause(.1)

        plt.ion()
        plt.figure(88,figsize=(17,4))
        #plt.clf()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35,right=0.85,left=0.05)

        ax = plt.gca()
        ax.set_xlabel('Generation')
        ax.set_ylabel('Prevalence')
        ax.set_title('Communication cases over generations (no-com)')
        #ax.set_title('Correlation Com Case, Usage, and Score (no-com)')
        cases = env.preds.stats.stats['com_investigations']
        tot_case = cases['all']
        for key in cases.keys():
            if key == 'all':
                continue
            #if key != 'target_on_agent':
            #    continue
            case = cases[key]
            com_amounts = [c[0] / (tot_case[i][0] if tot_case[i][0] > 0 else 1) for i,c in enumerate(case)]
            case_amounts = [c[1] / (tot_case[i][1] if tot_case[i][1] > 0 else 1) for i,c in enumerate(case)]
            sns.lineplot(x=range(1,len(case_amounts[:200])+1),y=case_amounts[:200],ax=ax)
        
        scores = [s/max(scores) for s in scores]
        #sns.lineplot(x=range(1,len(usage_pred)+1),y=usage_pred,ax=ax,color='purple')
        #sns.lineplot(x=range(1,len(scores)+1),y=scores,ax=ax,color='orange')
        plt.legend(list(cases.keys())[:-1],bbox_to_anchor=(1.01, 1))
        #plt.legend(['target_on_agent','com usage','score'])
        plt.ioff()
        plt.show(block=False)
        
        plt.pause(.1)
        

    def mix_colors(self, c1, c2, perc):
        c3 = [0,0,0]
        for i in range(3):
            c3[i] = int(c1[i] + (c2[i] - c1[i]) * (1-perc))
        return tuple(c3)