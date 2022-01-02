import pygame
from agents import Predator

from constants import PopPair
import constants as C

class Input:

    def __init__(self):
        self.switch_learn_next_gen = PopPair(False, False)

    def process_inputs(self,env, renderer):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    C.SHOW_WORLD = not C.SHOW_WORLD
                if event.key == pygame.K_LEFT:
                    C.SPEED = 1 if C.SPEED == 10 or C.SPEED == 1 else 10
                if event.key == pygame.K_RIGHT:
                    C.SPEED = 100_000_000 if C.SPEED == 10 or C.SPEED == 100_000_000 else 10
                if event.key == pygame.K_s:
                    env.preds.save_models(verbose=True)
                    env.preys.save_models(verbose=True)
                    env.preys.stats.save()
                    env.preds.stats.save()
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_c:
                    C.RENDER_COMMUNICATION = not C.RENDER_COMMUNICATION
                if event.key == pygame.K_o:
                    if C.LEARN.pred and env.preds.stats.get('it_gen') > 1:
                        C.LEARN = PopPair(C.LEARN.pred, not C.LEARN.prey)
                    env.preys.stats.stats['learn_switches'].append((env.preds.stats.get('it_gen'),env.preys.stats.get('it_gen'),C.LEARN.prey))
                if event.key == pygame.K_l:
                    if C.LEARN.prey and env.preys.stats.get('it_gen') > 1:
                        C.LEARN = PopPair(not C.LEARN.pred, C.LEARN.prey)
                    env.preds.stats.stats['learn_switches'].append((env.preds.stats.get('it_gen'),env.preys.stats.get('it_gen'),C.LEARN.pred))
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_x:
                    C.SAVE_MODEL = not C.SAVE_MODEL
                if event.key == pygame.K_t:
                    C.COMMUNICATE_WITHIN_POP = PopPair(not C.COMMUNICATE_WITHIN_POP.pred, C.COMMUNICATE_WITHIN_POP.prey)
                if event.key == pygame.K_u:
                    C.COMMUNICATE_WITHIN_POP = PopPair(C.COMMUNICATE_WITHIN_POP.pred, not C.COMMUNICATE_WITHIN_POP.prey)
                if event.key == pygame.K_y:
                    C.HEAR_BETWEEN_POP = PopPair(not C.HEAR_BETWEEN_POP.pred, C.HEAR_BETWEEN_POP.prey)
                if event.key == pygame.K_i:
                    C.HEAR_BETWEEN_POP = PopPair(C.HEAR_BETWEEN_POP.pred, not C.HEAR_BETWEEN_POP.prey)
                if event.key == pygame.K_p:
                    renderer.plot(env)
                    if C.PLOT_LAST_GEN:
                        renderer.plot_gen(env)
        