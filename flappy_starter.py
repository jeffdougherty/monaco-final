import random
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from copy import deepcopy

# import tensorflow as tf


NUM_FRAMES = 10000

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

N_PARAMS = 8
N_AGENTS = 10

MUTATE = True

ALGO_TO_USE = 2  # 1 = Jeff,  2= Jessie's (article's)

ACTION_MAP = {
    1: 119,
    0: None
}

DIST_MEAN = .1  # Mean of normal distribution
DIST_SD = 1     # SD of normal distribution

FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP)




def normalize(obs):
    x = [
        obs['player_y'] / HEIGHT,
        obs['player_vel'] / HEIGHT,
        obs['next_pipe_dist_to_player'] / WIDTH,
        obs['next_pipe_top_y'] / HEIGHT,
        obs['next_pipe_bottom_y'] / HEIGHT,
        obs['next_next_pipe_dist_to_player'] / WIDTH,
        obs['next_next_pipe_top_y'] / HEIGHT,
        obs['next_next_pipe_bottom_y'] / HEIGHT
    ]

    return np.array(x)


def agent(x, w):
    """
    Perceptron agent flaps if x dot w is >= 1

    x is the observed state
    w is the weight vector
    """
    return 0 if x @ w < 0.5 else 1


def initialize(n_agents=N_AGENTS):
    """
    Population of 10 agents, each drawn from a normal distribution
    """

    agents = []

    for i in range(n_agents):
        # this_agent = np.zeros(8)
        # this_agent = [1.1828843144200194, 0.047874517980502934, 0.0, -1.061810124546465, 1.0520386833146527, -0.06356522803788328, -1.061810124546465, 0.32965948666649525]
        this_agent = [np.random.normal(DIST_MEAN, DIST_SD) for j in range(N_PARAMS)]
        agents.append(this_agent)
        # agents.append(this_agent)

    return agents


def eval_fitness(w, seed=SEED, headless=False):
    """
    Evaluate the fitness of an agent with the game
    game is a PLE game
    agent is an agent function
    """

    # disable rendering if headless
    if headless:
        display_screen = False
        force_fps = True
    else:
        display_screen = True
        force_fps = False

    # game init
    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0
    r_vals = []
    dist_traveled = 0

    while True:
        if game.game_over():
            break

        obs = game.getGameState()
        # print(obs)
        x = normalize(obs)
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        # Score: Pipes traversed * r, see notes
        if ALGO_TO_USE == 1:
            center = abs(obs['next_pipe_top_y'] - obs['next_pipe_bottom_y'] / 2)
            target = obs['next_pipe_top_y'] + center

            player_success = 1 - abs(target - obs['player_y']) / target
            r_vals.append(player_success)


        # distance traveled - next_pipe_dist_to_player
        elif ALGO_TO_USE == 2:

            if reward >= 0:
                # next_pipe_dist_to_player[t - 1] - next_pipe_dist_to_player[t] = 4
                dist_traveled += 4
                # (this is consistent throughout game)

    # agent's fitness

    if ALGO_TO_USE == 1:

        agent_score = np.mean(r_vals)


    elif ALGO_TO_USE == 2:
        agent_score = dist_traveled - obs['next_pipe_dist_to_player']

    return agent_score


def crossover(parents):
    """
    Single-Point Crossover:
    Generate an offspring from two agents, w1 and w2
    """
    w1 = parents[0]
    w2 = parents[1]

    # Each agent is N_PARAMS long

    crossover_pt = random.randint(0, N_PARAMS - 1)
    swap = w1[crossover_pt]
    w1[crossover_pt] = w2[crossover_pt]
    w2[crossover_pt] = swap

    # the chosen one
    chosen_one = [w1, w2][random.randint(0, 1)]

    return [chosen_one]


def mutate(w):
    """
    Apply random mutations to an agent's genome
    """
    # Given in assignment p(mutate) should be 0.5

    # Hit a ton of plateaus, wanted to introduce a different kind of disruption
    mut = random.randint(0, 1)

    # if mut == 1:  # One means yes, mutate!  Binary joke, ga-harf, ga-harf
        # First pass at mutation: pick one position in array and randomize.
    # mut_posit = random.randint(0, N_PARAMS - 1)
    # w[mut_posit] = np.random.normal(DIST_MEAN, DIST_SD)

    # else:
        # add noise to all weights
    noise = np.random.normal(DIST_MEAN, DIST_SD, 8)
    w += noise

    # Other possibilities: generate small random quantity around mean of 0, add to one/each position
    return w


def train_agent(n_agents=N_AGENTS, n_epochs=1000, headless=False):
    """
    Train a flappy bird using a genetic algorithm
    """
    # TODO: genetic algorithm steps below

    # manually picked best weights thus far
    # superstar = [
    #     [1.1828843144200194, 0.047874517980502934, 0.0, -1.061810124546465, 1.0520386833146527, -0.06356522803788328,
    #      -1.061810124546465, 0.32965948666649525]]
    # superstar = superstar[0]

    res = open('training_results.txt', 'w')

    # --Initialization--
    # population = initialize(n_agents - 1)
    population = initialize(n_agents)
    # population.append(superstar)

    best_weights = []
    best_fitness = 0

    for g in range(n_epochs):

        # to evaluate fitness
        agent_dict = {}
        fit_list = []
        winners = []

        # Want both a way to sort fitness scores and a quick way
        # to find agent associated with that fitness score
        for w in population:
            # Fitness for this agent

            w_fit = eval_fitness(w, headless=headless)
            fit_list.append(w_fit)
            print(w_fit, w)

            # one measure of fitness could have multiple
            # sets of weights associated with it

            if w_fit not in agent_dict:
                agent_dict[w_fit] = []

            agent_dict[w_fit].append(w)

            # print('consistent best fitness:', best_fitness ==)
            # print('consistent best weights:', best_weights ==)

            # verify best fitness & associated weights

            if best_fitness < w_fit:
                best_fitness = w_fit
                best_weights = agent_dict[w_fit][:]
                print('best_fitness', best_fitness)
                print('best_weights', best_weights)
                res.write('\n' + '*FITNESS  ' + str(best_fitness) + '\n\n')
                res.write('\n' + '*WEIGHTS' + '\n' + str(best_weights) + '\n\n')

            res.write(str(w) + '\n' + str(w_fit) + '\n')

        fit_list.sort(reverse=True)  # Put fitness list in order

        # -- Selection --
        # Choose the 4 best agents for mating = "winners"
        best = 0
        count = 0

        while count < 4:

            # grab top agent metrics
            fitness = fit_list[best]
            weight_list = agent_dict[fitness][:]

            ### CHECK weight_list len

            random.shuffle(weight_list)

            # when we have multiple sets per fitness
            while len(weight_list) > 0 and len(winners) != 4:
                choice = deepcopy(weight_list[0])
                weight_list = weight_list[1:][:]

                winners.append(choice)

                count += 1

            best += 1

        top_2 = deepcopy(winners[:2])

        # -- Crossover --
        # Create 6 children total from the winner pool
        # * 1 child of top 2 winners
        # * 2 direct clones of top 2 winners
        # * 3 offspring of randomly chosen 2 of the 4 winners

        #Think we want to do it differently to accom diff numbers of agents, and keep our agent pool big:
        # * 1 child of top 2 winners
        # * 2 direct clones of top 2 winners
        # * (n_agents-3) offspring of randomly chosen 2 of 4 winners

        # Clones for the top 2 - not subject to mutation
        clones = [deepcopy(i) for i in top_2]  #Make sure we make a deep copy
        # print('clones==top_2', clones == top_2)

        # Children of the top 2
        #children = crossover([top_2[0], top_2[1]]) + clones #As written, this mutates the clones too!

        children = crossover([top_2[0], top_2[1]])

        # creates (n_agents - 3) offspring

        for _ in range(n_agents-3):
            random.shuffle(winners)
            children = children + crossover([winners[0], winners[1]])  #+= doesn't work on lists the way we think it does

        # -- Mutation --
        offspring = []

        for c in range(len(children)):

            if random.randint(0, 1):

                offspring.append(mutate(children[c]))

            else:

                offspring.append(children[c])

            #offspring.append(mutant)


        # -- Insertion --
        #population = clones + winners + offspring  #As written, this mutates the clones as well, and puts all 4 winners into the next gen
        population = offspring + clones

        print('consistent best weights:', best_weights[0] == population[0])
        print(len(best_weights[0]) == len(population[0]))
        print('best_weights', best_weights)


        result_string = "Generation " + str(g) + " Best Score: " + str(fit_list[0]) + "\n"

        print(result_string)
        res.write(result_string)

    # RETURN: The top agent of the last generation to undergo fitness evaluation
    best_agent = top_2[0]
    print('best_agent', best_agent)
    print('consistency:', best_weights[0] == best_agent)

    
    res.close()
    return best_agent


def main(w, seed=SEED, headless=False):
    """
    Let an agent play flappy bird
    """
    if headless:
        display_screen = False
        force_fps = True
    else:
        display_screen = True
        force_fps = False

    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0
    num_frames = 0

    while True:
        if game.game_over():
            break

        obs = game.getGameState()
        x = normalize(obs)
        # print(obs)
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        if reward > 0:
            agent_score += 1

        num_frames += 1

    print('Frames  :', num_frames)
    print('Score   :', agent_score)


if __name__ == '__main__':
    human_play = 0

    # added bc population reaches equilibrium v fast (aka stops learning & can't improve)

    # np.random.seed(random.randint(0, 100))
    np.random.seed(1234)

    if not human_play:
        w = train_agent()
        print(w)
        main(w)

    else:
        # For human play, use 'W' key to flap
        main(np.zeros(8), headless=False)
