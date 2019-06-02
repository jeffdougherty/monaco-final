import random
import numpy as np

#Toy GA example to make sure I'm implementing them right.
#Agents are arrays of 5 random integers
#Goal is to maximize sum of the ints

N_PARAMS = 5

MUTATE = True
P_MUT = 1

def main(num_agents=5, gens=5):
    f = open('test-results.txt', 'w')
    agents = []
    #Initialize
    for _ in range(num_agents):
        agents.append([random.randint(1, 100) for _ in range(N_PARAMS)])

    max_last = 0

    for g in range(gens):
        agent_dict = {}
        fit_list = []
        top_2 = []
        winners_fitness = []

        for w in agents:
            this_fit = fitness(w)
            fit_list.append(this_fit)
            agent_dict[this_fit] = w
        fit_list.sort(reverse=True)

        for i in range(4):
            winners_fitness.append(fit_list[i])
            if i <= 1:
                top_2.append(agent_dict[fit_list[i]])

        res_string = "Generation " + str(g) + " Max sum: " + str(fit_list[0]) + " Agent: " + str(top_2[0]) + "\n"
        print(res_string)
        f.write(res_string)

        agents = [top_2[0], top_2[1]]   #Copy of top_2
        clones = [top_2[0], top_2[1]]   #Make another copy that won't be subject to mutation
        agents += crossover(top_2[0], top_2[1]) #Children of the top 2

        for _ in range(4):

            parents = np.random.choice(winners_fitness, 2,
                                           False)  # Choose two different winners at random from the top 4
            agents += crossover(agent_dict[parents[0]], agent_dict[parents[1]])

        if MUTATE:
            agents = [mutate(_) for _ in agents]

        agents += clones


    f.close()

def fitness(w):
    return sum(w)

def crossover(w1, w2):
    '''
    Generate an offspring from two agents
    '''
    crossover_pt = random.randint(0, N_PARAMS)    #Each agent is N_PARAMS long

    for i in range(crossover_pt, N_PARAMS):
        swap = w1[i]
        w1[i] = w2[i]
        w2[i] = swap

    return [w1, w2]

def mutate(w):
    if random.random() <= P_MUT:
        mut_pos = random.randint(0, N_PARAMS - 1)

        if random.random() <= 0.5:
            w[mut_pos] += 5
        else:
            w[mut_pos] -= 5
    return w

main()