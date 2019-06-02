import os
import sys
import random
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from copy import deepcopy

NUM_FRAMES = 10000

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

N_PARAMS = 8

MUTATE = True

ALGO_TO_USE = 0         #0 = One that originally showed learning #1 = Jessie's  Even numbers Jessie, Odd numbers Jeff

ACTION_MAP = {
    1: 119,
    0: None
}

DIST_MEAN = 0.0  # Mean of normal distribution
DIST_SD = 0.5  # SD of normal distribution

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

    for i in range(n_agents):
        this_agent = [np.random.normal(DIST_MEAN, DIST_SD) for j in range(N_PARAMS)]
        agents.append(this_agent)

    return agents

def fitness(w, seed=SEED, headless=False):

    '''Evaluate the fitness of an agent with the game
    game is a PLE game
    agent is an agent function'''

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
    r_vals = []

    while True:
        if game.game_over():
            break

        obs = game.getGameState()
        x = normalize(obs)
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        #Score: Pipes traversed * r, see notes
        # agent_score += 1
        if ALGO_TO_USE == 0:
            agent_score += 1  # Number of frames traveled
            safe_zone_width = x[3] - x[4]
            safe_zone_center = safe_zone_width / 2
            r_vals.append((safe_zone_width - abs(safe_zone_center - x[0])) / safe_zone_width)
        elif ALGO_TO_USE == 1:
            center = abs(obs['next_pipe_top_y'] - obs['next_pipe_bottom_y'] / 2)
            target = obs['next_pipe_top_y'] + center

            player_success = 1 - abs(target - obs['player_y'])/target
            r_vals.append(player_success)
    if ALGO_TO_USE == 0:
        agent_score = agent_score * np.mean(r_vals)
    elif ALGO_TO_USE == 1:
        agent_score = np.mean(r_vals)
    # agent_score = agent_score * np.mean(r_vals)
    return agent_score

def crossover(w1, w2):
    '''
    Generate an offspring from two agents
    '''
    crossover_pt = random.randint(0, N_PARAMS-1)    #Each agent is N_PARAMS long

    for i in range(crossover_pt, N_PARAMS):
        swap = w1[i]
        w1[i] = w2[i]
        w2[i] = swap

    return [w1, w2]


def mutate(w):
    '''
    Apply random mutations to an agent's genome
    '''
    #Given in assignment p(mutate) should be 0.5
    #Rather than mess around with floats, just flip a coin

    mut = random.randint(0,1)
    if mut == 1:        #One means yes, mutate!  Binary joke, ga-harf, ga-harf
        #First pass at mutation: pick one position in array and randomize.
        mut_posit = random.randint(0, N_PARAMS-1)
        w[mut_posit] = np.random.normal(DIST_MEAN, DIST_SD)

        #Other possibilities: generate small random quantity around mean of 0, add to one/each position
    return w


def train_agent(n_agents=10, n_epochs=10, headless=True):
    '''
    Train a flappy bird using a genetic algorithm
    '''
    #
    # TODO: genetic algorithm steps below

    res = open('training_results.txt', 'w')

    # initialization
    population = initialize(n_agents)

    top_2 = []          #Defining here so PyCharm will shut up

    for g in range(n_epochs):
        # evaluate fitness
        agent_dict = {}
        fit_list = []
        top_2 = []
        parents = []

        for w in population:                        #Want both a way to sort fitness scores and a quick way to find agent associated with that fitness score
            #Note: this approach could have problems if we get agents with identical fitness.  Cross that bridge if we come to it.
            w_fit = fitness(w, headless=headless)   #Fitness for this agent
            fit_list.append(w_fit)

            if w_fit not in agent_dict:             #No longer assuming that fitnesses are unique, since they apparently aren't
                agent_dict[w_fit] = []
            agent_dict[w_fit].append(w)


            '''if w_fit not in agent_dict:
                agent_dict[w_fit] = [w]'''
        fit_list.sort(reverse=True)                             #Put fitness list in order

        '''for i in range(4):
            this_fitness = fit_list[i]
            winners_fitness.append(this_fitness)     #Fitness of top 4 on list
            if i <= 1:                              #Also track top 2 separately, get actual agents so we can clone and cross
                top_2.append(agent_dict[this_fitness])'''

        for i in range(4):
            this_fitness = fit_list[i]
            this_fitness_list = agent_dict[this_fitness]
            while len(this_fitness_list) > 0 and len(top_2) < 2 and len(parents) < 4:
                if len(this_fitness_list) > 1:
                    random.shuffle(this_fitness_list)
                this_agent = this_fitness_list.pop()
                top_2.append(this_agent)
                parents.append(this_agent)
            while len(this_fitness_list) > 0 and len(parents) < 4:
                if len(this_fitness_list) > 1:
                    random.shuffle(this_fitness_list)
                parents.append(this_fitness_list.pop())


        # crossover
        #children = [deepcopy(i) for i in top_2]
        clones = [deepcopy(i) for i in top_2] #Clones for the top 2, will not be subjected to mutation
        children = []
        children = children + crossover(top_2[0], top_2[1])  #Children of the top 2
        for _ in range(4):
            '''parents = np.random.choice(winners_fitness, 2, False)       #Choose two different winners at random from the top 4
            children += crossover(agent_dict[parents[0]], agent_dict[parents[1]])'''
            random.shuffle(parents)
            children = children + crossover(parents[0], parents[1])



        # mutation
        if MUTATE:
            children = [mutate(_) for _ in children]

        # insertion
        population = children
        population = population + clones

        result_string = "Generation " + str(g) +" Best Score: " + str(fit_list[0]) + "\n"

        print(result_string)
        res.write(result_string)

    # return the best agent found
    best_agent = top_2[0]       #The top agent of the last generation to undergo fitness evaluation
    res.close()
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
    human_play = False
    np.random.seed(1234)
    if not human_play:
        w = train_agent()
        main(w)
    else:
    # For human play, use 'W' key to flap
        main(np.zeros(8), headless=False)
