from environment import Environment
from renderer import Renderer
import constants as C
from input import Input
from helper import print_state

class Controller:

    def __init__(self):
        self.env = Environment()
        self.renderer = Renderer(self.env.w, self.env.h)
        self.input = Input()
    
    def is_gameover(self):
        return self.env.preds.is_empty() or self.env.preys.is_empty()

    def run(self):

        while C.GEN_AMOUNT < 0 or self.env.preds.stats.get('it_gen') <= C.GEN_AMOUNT:

            # Predator iteration loop
            if not self.is_gameover():
                for i, pred in enumerate(self.env.preds.agents):

                    state = pred.get_state(self.env)
                    action, vals, a2, com = pred.get_action(state)
                    self.env.preds.stats.record_one_iteration(state, action, com,a2)
                    self.env.preds.stats.c_frames += 1
                    terminated = self.env.play_step(pred,action, a2, com)

                    #if C.SHOW_WORLD and i == 0:
                    #    print_state(state, action, vals)

                    if terminated:
                        self.env.preds.remove_agent(i)

                    if C.LEARN.pred:
                        self.env.preds.stats.iterate('it_frames')

                    if self.is_gameover():
                        break

                    

            # Prey iteration loop
            if not self.is_gameover():
                for i, prey in enumerate(self.env.preys.agents):

                    terminated = False
                    if prey.moving:
                        state = prey.get_state(self.env)
                        action, _, a2, com = prey.get_action(state)
                        self.env.preys.stats.record_one_iteration(state, action, com,a2)
                        terminated = self.env.play_step(prey,action,a2,com)
                    else:
                        terminated = self.env.play_step(prey,None,None,None)

                    if terminated:
                        self.env.preys.remove_agent(i)
                    if C.LEARN.prey:
                        self.env.preys.stats.iterate('it_frames')

                    if self.is_gameover():
                        break

            # Render
            if C.SHOW_WORLD:
                self.renderer.render(self.env)
                if C.SPEED < 10000:
                    self.renderer.tick()
            
            # Stats
            if C.PLOT_LAST_GEN:
                self.env.preds.stats.record_current_gen_scores(self.env)
                self.env.preys.stats.record_current_gen_scores(self.env)
            if not self.is_gameover(): 
                pass
                #distances = self.env.get_closest_distances()
                #coms = self.env.get_indiv_coms()
                #self.env.preds.stats.record_indiv(distances['pred'],coms['pred'])
                #self.env.preys.stats.record_indiv(distances['prey'],coms['prey'])

            # Input
            self.input.process_inputs(self.env,self.renderer)
            
            # World
            self.env.preds.repopulate(self.env)
            self.env.preys.repopulate(self.env)

            # End of world logic
            if self.is_gameover():
            
                # train populations
                if self.env.preds.is_empty():
                
                    if C.LEARN.pred:
                        self.env.preds.have_sex()
                    else:
                        self.env.preds.reset_same()

                if self.env.preys.is_empty(): 

                    if C.LEARN.prey:
                        self.env.preys.have_sex()
                    else:
                        self.env.preys.reset_same()

                # new world
                #print("GAMEOVER")
                #self.env.preds.stats.record_world_stats()
                #self.env.preys.stats.record_world_stats()
                self.env.reset()

                # output
                #if (self.env.preds.stats.is_record(C.FITNESS_FUNCTION.pred)\
                #    or self.env.preys.stats.is_record(C.FITNESS_FUNCTION.prey)) and not C.SHOW_WORLD: 
                #    self.renderer.plot(self.env)
                if C.SHOW_WORLD or C.PLOT_GRAPHS:
                    self.renderer.plot(self.env)
                    if C.PLOT_LAST_GEN:
                        self.renderer.plot_gen(self.env)
                self.env.preds.stats.reset_current_gen_scores()
                self.env.preys.stats.reset_current_gen_scores()
                print('|',end='')

        if self.env.preds.stats.get('it_gen') >= C.GEN_AMOUNT:
            if C.GEN_AMOUNT != 0:
                self.env.preds.save_models()
                self.env.preys.save_models()
                self.env.preys.stats.save()
                self.env.preds.stats.save()
            self.renderer.plot(self.env)