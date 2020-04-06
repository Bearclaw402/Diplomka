## some genetic algorithm stuff
import numpy
import random

# Converting each solution from matrix to vector.
def mat_to_vector(mat_pop_weights):
    pop_weights_vector = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        curr_vector = []
        for layer_idx in range(mat_pop_weights.shape[1]):
            vector_weights = numpy.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
            curr_vector.extend(vector_weights)
        pop_weights_vector.append(curr_vector)
    return numpy.array(pop_weights_vector)

# Converting each solution from vector to matrix.
def vector_to_mat(vector_pop_weights, mat_pop_weights):
    mat_weights = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        start = 0
        end = 0
        for layer_idx in range(mat_pop_weights.shape[1]):
            end = end + mat_pop_weights[sol_idx, layer_idx].size
            curr_vector = vector_pop_weights[sol_idx, start:end]
            mat_layer_weights = numpy.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
            mat_weights.append(mat_layer_weights)
            start = end
    return numpy.reshape(mat_weights, newshape=mat_pop_weights.shape)

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint32(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, mutation_percent):
    num_mutations = numpy.uint32((mutation_percent*offspring_crossover.shape[1])/100)
    mutation_indices = numpy.array(random.sample(range(0, offspring_crossover.shape[1]), num_mutations))
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, mutation_indices] = offspring_crossover[idx, mutation_indices] + random_value
    return offspring_crossover

def fitness():
    pass

def doGenAlg(input_weights, num_parents, num_generations, mutation_probability):
    accuracies = numpy.empty(shape=(num_generations))
    pop_weights_mat = numpy.array(input_weights)
    pop_weights_vector = mat_to_vector(pop_weights_mat)

    for generation in range(num_generations):
        print("Generation : ", generation)

        # converting the solutions from being vectors to matrices.
        pop_weights_mat = vector_to_mat(pop_weights_vector,
                                           pop_weights_mat)

        # Measuring the fitness of each chromosome in the population.
        fitness = fitness()

        accuracies[generation] = fitness[0]
        print("Fitness")
        print(fitness)

        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(pop_weights_vector,

                                        fitness.copy(),

                                        num_parents)
        print("Parents")
        print(parents)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                           offspring_size=(
                                           pop_weights_vector.shape[0] - parents.shape[0], pop_weights_vector.shape[1]))

        print("Crossover")
        print(offspring_crossover)

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover,
                                         mutation_percent=mutation_probability)
        print("Mutation")
        print(offspring_mutation)

        # Creating the new population based on the parents and offspring.
        pop_weights_vector[0:parents.shape[0], :] = parents
        pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

    pop_weights_mat = vector_to_mat(pop_weights_vector, pop_weights_mat)
    best_weights = pop_weights_mat[0, :]
    return best_weights