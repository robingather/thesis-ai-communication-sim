import pygame
from constants import PopPair
import constants as C

class Input:
    def process_inputs(self,env, renderer):
        # Handles all input events
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB: # enable/disable rendering
                    C.SHOW_WORLD = not C.SHOW_WORLD
                if event.key == pygame.K_LEFT: # simulation speed
                    C.SPEED = 1 if C.SPEED == 10 or C.SPEED == 1 else 10
                if event.key == pygame.K_RIGHT: # simulation speed
                    C.SPEED = 100_000_000 if C.SPEED == 10 or C.SPEED == 100_000_000 else 10
                if event.key == pygame.K_s: # save models
                    env.preds.save_models(verbose=True)
                    env.preys.save_models(verbose=True)
                    env.preys.stats.save()
                    env.preds.stats.save()
                if event.key == pygame.K_c: # enable/disable communication rendering
                    C.RENDER_COMMUNICATION = not C.RENDER_COMMUNICATION
                if event.key == pygame.K_o: # enable/disable prey learning
                    if C.LEARN.pred and env.preds.stats.get('it_gen') > 1:
                        C.LEARN = PopPair(C.LEARN.pred, not C.LEARN.prey)
                    env.preys.stats.stats['learn_switches'].append((env.preds.stats.get('it_gen'),env.preys.stats.get('it_gen'),C.LEARN.prey))
                if event.key == pygame.K_l: # enable/disable pred learning
                    if C.LEARN.prey and env.preys.stats.get('it_gen') > 1:
                        C.LEARN = PopPair(not C.LEARN.pred, C.LEARN.prey)
                    env.preds.stats.stats['learn_switches'].append((env.preds.stats.get('it_gen'),env.preys.stats.get('it_gen'),C.LEARN.pred))
                if event.key == pygame.K_ESCAPE: # quit game
                    pygame.quit()
                    quit()
                if event.key == pygame.K_x: # enable/disable model saving
                    C.SAVE_MODEL = not C.SAVE_MODEL
                if event.key == pygame.K_t: # enable/disable predator within-pop communication
                    C.COMMUNICATE_WITHIN_POP = PopPair(not C.COMMUNICATE_WITHIN_POP.pred, C.COMMUNICATE_WITHIN_POP.prey)
                if event.key == pygame.K_u: # enable/disable prey within-pop communication
                    C.COMMUNICATE_WITHIN_POP = PopPair(C.COMMUNICATE_WITHIN_POP.pred, not C.COMMUNICATE_WITHIN_POP.prey)
                if event.key == pygame.K_y: # enable/disable predator between-pop communication
                    C.HEAR_BETWEEN_POP = PopPair(not C.HEAR_BETWEEN_POP.pred, C.HEAR_BETWEEN_POP.prey)
                if event.key == pygame.K_i: # enable/disable prey between-pop communication
                    C.HEAR_BETWEEN_POP = PopPair(C.HEAR_BETWEEN_POP.pred, not C.HEAR_BETWEEN_POP.prey)
                if event.key == pygame.K_p: # show plot
                    renderer.plot(env)
                    if C.PLOT_LAST_GEN:
                        renderer.plot_gen(env)