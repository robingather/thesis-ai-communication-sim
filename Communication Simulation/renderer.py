import matplotlib.pyplot as plt
import numpy as np
import pygame
import seaborn as sns
from helper import Point
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
font = pygame.font.Font('ROBOTOSLAB-REGULAR.TTF', 14)
font_s = pygame.font.Font('ROBOTOSLAB-REGULAR.TTF', 10)

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

    def tick(self):
        self.clock.tick(C.SPEED)

    def render(self, env):
        self.render_world(env)
        self.render_bottom(env)
        pygame.display.flip()

    def render_world(self, env):
        # render the simulation world
        self.display.fill(LIGHT_GREEN, pygame.Rect(0,0,self.w,self.h))
        BS = C.BLOCK_SIZE
        mBS = BS/4

        for pred in env.preds.agents:
            if pred.com != 0 and C.RENDER_COMMUNICATION:
                self.render_circle(env, pred.pos, C.RAD_COM)

        for pred in env.preds.agents:
            agent_color = LIGHT_RED
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

        for obst in env.obstacles:
            pygame.draw.rect(self.display, DARK_GREEN, pygame.Rect(obst.x*BS, obst.y*BS, BS, BS))

        for prey in env.preys.agents:
            agent_color = self.mix_colors(LIGHT_BLUE, LIGHT_GREEN, prey.health/C.MAX_HEALTH.prey)
            pygame.draw.rect(self.display, agent_color, pygame.Rect(prey.pos.x*BS, prey.pos.y*BS, BS, BS))

    def render_circle(self, env, pos, r):
        BS = C.BLOCK_SIZE
        pygame.draw.circle(self.display, FADED_RED, Point(pos.x*BS,pos.y*BS), r*BS)

    def render_bottom(self, env):
        # render stats on the bottom
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
        # plot communication statistics
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
        sns.barplot(x=vals[:-1],y=cases[:-1])

        ax = axs[1,0]
        ax.set_title('Avg Com Value For Cases')
        ax.set_xlabel('Avg Com Value')
        ax.set_ylabel('Case')
        cases=list(env.preds.stats.gen_stats['com_investigations'].keys())
        vals=list(env.preds.stats.gen_stats['com_investigations'].values())
        for i in range(len(vals)):
            vals[i] = (vals[i][0]/vals[i][1]) if vals[i][1] > 0 else 0
        plt.sca(ax)
        sns.barplot(x=vals[:-1],y=cases[:-1])

        # Correlation
        vals = env.preds.stats.c_com_actions.copy()
        for key in vals.keys():
            vals[key] = vals[key][:4]

        any_tot_amount = sum(vals['C>0'])
        if any_tot_amount != 0:
            vals['C>0'] = [x/any_tot_amount for x in vals['C>0']]

        ax = axs[0,3]
        plt.sca(ax)
        sns.barplot(x=vals['C>0'],y=C.ACTION_LABELS[:4])

        axes = [[0,1],[1,1],[0,2],[1,2],[0,3],[1,3]]
        for i,key in enumerate(vals.keys()):
            if not key in ['C_L>0','C_R>0','C_U>0','C_D>0','C==0']:
                continue
            ax = axs[axes[i][0],axes[i][1]]
            
            vals[key] = vals[key]
            tot_amount = sum(vals[key])
            if tot_amount != 0:
                vals[key] = [x/tot_amount for x in vals[key]]
                
                vals[key] = [x-vals['C>0'][i] for i,x in enumerate(vals[key])]
                print(key+' '+str(vals[key]))

            plt.sca(ax)
            ax.set_title('Actions correlation with '+key+' (N='+str(self.human_format(int(tot_amount)))+")")
            ax.set_xlabel('Difference from Avg')
            ax.set_ylabel('Action')
            sns.barplot(x=vals[key],y=C.ACTION_LABELS[:4])

        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)

    def plot_preds(self, env):
        # plot predator distance statistics (unused)
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
        # plot one generation statistics
        self.plot_com(env)
        #self.plot_preds(env)

    def plot(self, env):
        # plot main generational statistics
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
        
        # com use
        ax = axs[2]
        ax.set_xlabel('Generation')
        ax.set_ylabel('\% of pred actions',color='purple')

        com_usage = env.preds.stats.stats['com_usage']
        print("USAGE")
        print(com_usage)
        #x_usage = env.preds.stats.stats['x_usage']
        #sns.lineplot(x=range(1,len(com_usage)+1),y=np.array(com_usage)/np.array(x_usage),ax=ax,color='purple')
        sns.lineplot(x=range(1,len(com_usage)+1),y=com_usage,ax=ax,color='pink')

        plt.ioff()
        plt.show(block=False)
        plt.pause(.1)
        
    def mix_colors(self, c1, c2, perc):
        # mix one color with a percentage of another
        c3 = [0,0,0]
        for i in range(3):
            c3[i] = int(c1[i] + (c2[i] - c1[i]) * (1-perc))
        return tuple(c3)