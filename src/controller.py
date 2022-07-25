from environment import Environment
from renderer import Renderer
import constants as C
from input import Input

class Controller:

    def __init__(self):
        self.env = Environment()
        self.renderer = Renderer(self.env.w, self.env.h)
        self.input = Input()
    
    def is_gameover(self):
        # is either population's genenome array exhausted?
        return self.env.preds.is_empty() or self.env.preys.is_empty()

    def run(self):
        # main loop
        while C.GEN_AMOUNT < 0 or self.env.preds.stats.get('it_gen') <= C.GEN_AMOUNT:

            # predator iteration loop
            if not self.is_gameover():
                for i, pred in enumerate(self.env.preds.agents):

                    state = pred.get_state(self.env)
                    action, _, a2, com = pred.get_action(state)
                    self.env.preds.stats.record_one_iteration(state, action, com,a2)
                    self.env.preds.stats.c_frames += 1
                    terminated = self.env.play_step(pred,action, a2, com)

                    if terminated:
                        self.env.preds.remove_agent(i)

                    if C.LEARN.pred:
                        self.env.preds.stats.iterate('it_frames')

                    if self.is_gameover():
                        break

            # prey iteration loop
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

            # render
            if C.SHOW_WORLD:
                self.renderer.render(self.env)
                if C.SPEED < 10000:
                    self.renderer.tick()
            
            # stats
            if C.PLOT_LAST_GEN:
                self.env.preds.stats.record_current_gen_scores(self.env)
                self.env.preys.stats.record_current_gen_scores(self.env)

            # input
            self.input.process_inputs(self.env,self.renderer)
            
            # world
            self.env.preds.repopulate(self.env)
            self.env.preys.repopulate(self.env)

            # if either population has no more agents to spawn
            if self.is_gameover():
            
                # train populations
                if self.env.preds.is_empty():
                    if C.LEARN.pred:
                        self.env.preds.reproduce()
                    else:
                        self.env.preds.reset_same()
                if self.env.preys.is_empty(): 
                    if C.LEARN.prey:
                        self.env.preys.reproduce()
                    else:
                        self.env.preys.reset_same()

                self.env.reset()

                # render and plot
                if C.SHOW_WORLD or C.PLOT_GRAPHS:
                    self.renderer.plot(self.env)
                    if C.PLOT_LAST_GEN:
                        self.renderer.plot_gen(self.env)
                self.env.preds.stats.reset_current_gen_scores()
                self.env.preys.stats.reset_current_gen_scores()
                print('|',end='')

        # save populations if generation limit enabled
        if self.env.preds.stats.get('it_gen') >= C.GEN_AMOUNT:
            if C.GEN_AMOUNT != 0:
                self.env.preds.save_models()
                self.env.preys.save_models()
                self.env.preys.stats.save()
                self.env.preds.stats.save()
            self.renderer.plot(self.env)