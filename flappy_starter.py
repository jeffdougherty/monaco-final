import os
import sys
import random
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE

NUM_FRAMES = 10000

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

N_PARAMS = 8

ACTION_MAP = {
    1: 119,
    0: None
}

FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP)

def normalize(obs):
    x = [
        obs['player_y']/HEIGHT,
        obs['player_vel']/HEIGHT,
        obs['next_pipe_dist_to_player']/WIDTH,
        obs['next_pipe_top_y']/HEIGHT,
        obs['next_pipe_bottom_y']/HEIGHT,
        obs['next_next_pipe_dist_to_player']/WIDTH,
        obs['next_next_pipe_top_y']/HEIGHT,
        obs['next_next_pipe_bottom_y']/HEIGHT
    ]

    return np.array(x)


def agent(x, w):
    '''
    Perceptron agent flaps if x dot w is >= 1

    x is the observed state
    w is the weight vector
    '''
    return 0 if x @ w < 0.5 else 1


def initialize(n_agents=10):
    '''
    Initialize the population
    '''

    agents = []

    dist_mean = 0.5     #Mean of normal distribution
    dist_sd = 0.25      #SD of normal distribution

    for i in range(n_agents):
        this_agent = [np.random.normal(dist_mean, dist_sd) for j in range(N_PARAMS)]
        agents.append(this_agent)

    return agents

    #return [np.zeros(N_PARAMS) for _ in range(n_agents)]


def fitness(w, seed=SEED, headless=True):
    '''
    Evaluate the fitness of an agent with the game

    game is a PLE game
    agent is an agent function
    '''
    # disable rendering if headless
    if headless:
        display_screen=False
        force_fps=True
    else:
        display_screen=True
        force_fps=False

    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0

    while True:
        if game.game_over():
            break

        x = normalize(game.getGameState())
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        # TODO: your fitness logic goes here

    return agent_score


def crossover(w1, w2):
    '''
    Generate an offspring from two agents
    '''
    crossover_pt = random.randint(N_PARAMS)    #Each agent is N_PARAMS long

    for i in range(crossover_pt, N_PARAMS):
        swap = w1[i]
        w1[i] = w2[i]
        w2[i] = swap

    return w1, w2


def mutate(w):
    '''
    Apply random mutations to an agent's genome
    '''
    # TODO: your mutation logic goes here
    return ...


def train_agent(n_agents=10, n_epochs=100, headless=True):
    '''
    Train a flappy bird using a genetic algorithm
    '''
    #
    # TODO: genetic algorithm steps below

    # initialization
    population = initialize(n_agents)

    for i in range(n_epochs):
        # evaluate fitness
        fits = [fitness(w, headless=headless) for w in population]

        # selection
        winners = ...

        # crossover
        children = ...

        # mutation
        children = ...

        # insertion
        population = ...

    # return the best agent found
    best_agent = ...

    return best_agent


def main(w, seed=SEED, headless=True):
    '''
    Let an agent play flappy bird
    '''
    if headless:
        display_screen=False
        force_fps=True
    else:
        display_screen=True
        force_fps=False

    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0
    num_frames = 0

    while True:
        if game.game_over():
            break

        x = normalize(game.getGameState())
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        if reward > 0:
            agent_score += 1

        num_frames += 1

    print('Frames  :', num_frames)
    print('Score   :', agent_score)


if __name__ == '__main__':
    np.random.seed(1234)
    w = train_agent()
    main(w)

    # For human play, use 'W' key to flap
    # main(np.zeros(8))
