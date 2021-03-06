import random
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from copy import deepcopy
from statistics import mean

import inspect

# Game setup
NUM_FRAMES = 10000

WIDTH = 288
HEIGHT = 512
GAP = 100
SEED = 1234

# Weight params
POS_Y = 0
Y_VEL = 1
NPIPE_DIST = 2
NPIPE_TOP_Y = 3
NPIPE_BOTTOM_Y = 4
NNPIPE_DIST = 5
NNPIPE_TOP_Y = 6
NNPIPE_BOTTOM_Y = 7


N_PARAMS = 8
N_AGENTS = 10

# Mutation 
P_MUT = 0.5         #Probability of mutation
MUTATE_ALL = True   #When mutation is rolled, apply to all genes, or just one?
MUT_MEAN = 0        #Mean and SD of the normal distribution used for mutation
MUT_SD = 1.2

ALGO_TO_USE = 2
# 1 = Distance in both X and Y 2 = Distance in Y only

ACTION_MAP = {
    1: 119,
    0: None
}

ALTER_SEED = True   #If true, use a different random seed for each generation

PRINT_INDIV = False

FLAPPYBIRD = FlappyBird(width=WIDTH, height=HEIGHT, pipe_gap=GAP)




def ezprint(var):
    """
    Because I'm tired of writing multi-arg print statements
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print([var_name for var_name, var_val in callers_local_vars if var_val is var], len(var), var)



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

        this_agent = [np.random.uniform(-1,1) for _ in range(N_PARAMS)]
        agents.append(this_agent)
        # agents.append(this_agent)

    return agents


def eval_fitness(w, seed=SEED, headless=True):
    """
    Evaluate the fitness of an agent with the game
    game is a PLE game
    agent is an agent function
    """

    # disable rendering if headless
    display_screen = not(headless)
    force_fps = headless

    # game init
    game = PLE(FLAPPYBIRD, display_screen=display_screen, force_fps=force_fps, rng=seed)
    game.init()
    game.reset_game()
    FLAPPYBIRD.rng.seed(seed)

    agent_score = 0

    while True:

        if game.game_over():
            break

        obs = game.getGameState()
        x = normalize(obs)
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        if reward > 0:

            agent_score += 1
            if PRINT_INDIV:
                print('SCORED', agent_score)

    if ALGO_TO_USE == 1:
        #Sense both X distance to next pipe and Y distance to a notional "target" halfway between the next pipe's top and bottom Y
        pipe_proximity_score = 1.0 - x[2]
        agent_score += pipe_proximity_score
        target = get_target(x)
        dist_from_target = abs(target-x[0])
        agent_score -= dist_from_target
        if PRINT_INDIV:
            print("distance to next pipe:", x[2], "pipe proximity score:", pipe_proximity_score)
            print("distance", dist_from_target, "distance score:", dist_from_target)
            print("agnent score:", agent_score)

    elif ALGO_TO_USE == 2:
        #Sense safe zone in Y only
        target = get_target(x)
        dist_from_target = abs(target-x[0])
        dist_score = 1-dist_from_target
        if PRINT_INDIV:
            print("distance", dist_from_target, "distance score:", dist_score)
        agent_score += dist_score

    return agent_score

def get_target(x):
    #Find point halfway between top and bottom Y
    safe_zone_normalized = x[4] - x[3]  # Relative proportion of top y - bottom y
    target = (safe_zone_normalized / 2) + x[3]
    if PRINT_INDIV:
        print("Target:", target, "player Y", x[0])
    return target

def crossover(parents):
    """
    Single-Point Crossover: combining 2 slices of parent attributes 
    - exchanging all values *from crossover_pt to end of string*
    """
    
    # pick crossover point at random 

    crossover_pt = random.randint(0, N_PARAMS - 1)
    return parents[0][:crossover_pt] + parents[1][crossover_pt:]


def mutate(w):
    """
    Apply mutations to an agent's genome with probability 0.5
    """
    mut = random.uniform(0,1)
    if not MUTATE_ALL:
        if mut <= P_MUT:
        
            mut_point = random.randint(0, N_PARAMS - 1)
            w[mut_point] += np.random.normal(MUT_MEAN, MUT_SD)
    

    else:

        if mut <= P_MUT:
        
           for i in range(N_PARAMS):
               w[i] += np.random.normal(MUT_MEAN, MUT_SD)
    
    return w

def train_agent(n_agents=N_AGENTS, n_epochs=200, headless=True):
    """
    Train a flappy bird using a genetic algorithm
    """

    res = open('training_results.csv', 'w')

    # --Initialization--
    population = initialize(n_agents)

    best_fitness = 0
    winners = []

    for g in range(n_epochs):

        # to evaluate fitness
        agent_dict = {}
        fit_list = []       #List of fitnesses used as indexes to agents for each generation
        winners = []        #Agents chosen to reproduce for next generation

        if ALTER_SEED:
            this_seed = np.random.randint(1, 10000)
        else:
            this_seed = SEED

        # Want both a way to sort fitness scores and a quick way
        # to find agent associated with that fitness score
        for w in population:
            # Fitness for this agent
            w_fit = eval_fitness(w, seed=this_seed, headless=headless)
            fit_list.append(w_fit)
            #print(w_fit, w)

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

        fit_mean = mean(fit_list)

        fit_list.sort(reverse=True)  # Put fitness list in order

        # -- Selection --
        # Choose the 4 best agents for mating = "winners"
        best = 0

        while len(winners) < 4:

            # grab top agent metrics
            fitness = fit_list[best]
            agent_list = agent_dict[fitness]

            ### CHECK weight_list len
            if len(agent_list) > 1:
                random.shuffle(agent_list)

            while len(agent_list) > 0 and len(winners) < 4:
                choice = agent_list.pop(0)                 #Remove from weight list and pop to choice
                winners.append(choice)

            best += 1                                       #Agents get added to winners in order of fitness


        top_2 = deepcopy(winners[:2])                   #Top 2 agents, will be reproduced mutated and put in pool


        ''''# -- Crossover --
        # Create 6 children total from the winner pool
        # * 4 winners from the previous generation
        # * 3 offspring of randomly chosen 2 of the 4 winners
        # * 2 direct clones of top 2 winners
        # * 1 child of top 2 winners
        # * And a partridge in a pear tree'''

        #Start initializing children
        #Start with the 4 winners that are going to be carried forward into the next generation

        children = [deepcopy(p) for p in winners]
        
        if n_agents == 10:
            #Add offspring of top 2 winners
            children.append(mutate(crossover(top_2)))
            #Now that we've used the top 2 to produce children, can mutate them
            for agent in top_2:
                children.append(mutate(agent))


            for _ in range(4):
                random.shuffle(winners)
                children.append(mutate(crossover([winners[0], winners[1]])))


        result_string = "Generation " + str(g) + " Best Score: " + str(fit_list[0]) + "\n"

        print(result_string)

        csv_string = str(g) + "," + str(fit_list[0]) + "," + str(fit_mean) +"\n"
        res.write(csv_string)


        population = children

    # RETURN: The top agent of the last generation to undergo fitness evaluation
    best_agent = winners[0]

    print('best_agent', best_agent)

    
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
        action = agent(x, w)

        reward = game.act(ACTION_MAP[action])

        if reward > 0:
            agent_score += 1

        num_frames += 1

    print('Frames  :', num_frames)
    print('Score   :', agent_score)


if __name__ == '__main__':
    human_play = False
    eval_agent = None

    # added bc population reaches equilibrium v fast (aka stops learning & can't improve)

    # np.random.seed(random.randint(0, 100))
    np.random.seed(1234)

    if not human_play and not eval_agent:
        w = train_agent(headless=True)
        print(w)
        literally_nothing = input('Press enter to continue with main() & observe behavior: ')
        main(w, headless=False)

    elif not human_play:
        main(eval_agent, headless=False)

    else:
        # For human play, use 'W' key to flap
        main(np.zeros(8), headless=False)
