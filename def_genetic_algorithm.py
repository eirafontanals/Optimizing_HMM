'''
GENETIC ALGORITHM FOR THE OPTIMIZATION OF THE PARAMETERS OF A HMM
'''
# Imports
import math
import random
import matplotlib.pyplot as plt

# Create a class for the candidate solutions
class Solution(object):
    def __init__(self, initial_logits, legend_vector, states, categories):
        # Create a variable to store the input list with the names of the states
        self.states = states
        # Create a variable to store the input list with the names of the categories
        self.categories = categories
        # Initialize the solution: store the initial logits in the variable 'vector_logits' (chromosome)
        self.vector_logits = initial_logits
        # 'legend_vector' is a list that indicates how many emission and transition probabilities per state are encoded in the vector (chromosome) change: [num_emission_prob, num_transition_prob]
        self.legend_vector = legend_vector
        # # Compute theh likelihood with the initial logits
        self.likelihood = self.compute_likelihood()

    # Create a function (method) to retrieve the logit located at a specific position in the list
    def logit_at_position(self,position):
        # return the logit by indexing through the position
        return self.vector_logits[position]
    
    def get_logits(self):
        # return the vector of logits
        return self.vector_logits

    def update_logits(self, new_logits):
        self.vector_logits = new_logits

    def get_legend(self):
        # return the legend vector
        return self.legend_vector
    
    # Define a function to decode the solution
    def logit_to_prob(self):
        # Index to know until which position of the vector of logits has already been converted into probability (the index marks the next position to be converted)
        i = 0

        # STEP 1: EMISSION PROBABILITIES
        # Create the dictionary where the emission probabilities will be stored:
        emission = {}
        # Iterate through the states:
        for state in self.states:
            # Store the logits that will be used in this iteration in the variable 'logits'
            logits = self.vector_logits[i:i+self.legend_vector[0]]

            # Transform the logits into emission probabilities. Given there will be k probabilities in total:
            #   (1) calculate the first k-1 emission probabilities (the ones that are encoded in the vector)
            for logit in logits:
                try:
                    math.exp(logit)
                except Exception as e:
                    print(f"Error with value {logit}, couldn't compute exp")
                    print(f"Current solution: {self.vector_logits}")
                
            numerator = [math.exp(logit) for logit in logits]
            denominator = 1 + sum(numerator)
            probabilities = [value/denominator for value in numerator]
            #   (2) calculate the kth emission probability (the one that was not included in the vector)
            probabilities.append(1-sum(probabilities))

            # Create a dictionary with the emission probabilities of the current state:
            d = {}
            for x in range(len(self.categories)):
                d[self.categories[x]] = probabilities[x]
            # Store the dictionary in the associated state:
            emission[state] = d

            # Update the index until the last logit that has been used
            i += self.legend_vector[0]
        
        # STEP 2: TRANSITION PROBABILITIES (Important: in this function it is assumed that there are only 2 different states)
        # Create the dictionary where the transition probabilities will be stored:
        transition = {}
        # Iterate through the states:
        for state in self.states:
            logit = self.vector_logits[i]

            # Transform the logits into transition probabilities. Given there will be k probabilities in total:
            #   (1) calculate the first k-1 transition probabilities (the ones that are encoded in the vector)
            numerator = math.exp(logit)
            denominator = 1 + numerator
            probabilities = [numerator/denominator]
            #   (2) calculate the kth transition probability (the one that was not included in the vector)
            probabilities.append(1.0-probabilities[0])

            # Create a dictionary with the transition probabilities of the current state:
            d = {}
            for x in range(len(self.states)):
                d[self.states[x]] = probabilities[x]
            # Store the dictionary in the associated state:
            transition[state] = d

            # Update the index until the last logit that has been used
            i += 1
    
        return emission, transition

    
    def forward_algorithm(self, sequence, prior_probabilities, emission, transition):
        prior_probabilities = {"F":0.5, "L":0.5}
        '''
        scaling factor
        '''
        f_i = prior_probabilities.copy()

        '''
        the probability turns out to be the multiplication of the scaling factor.
        Since we will use the log, it is the sum of the log of the scaling factor.
        Since the initial s0 adds to 1, the log of it is 0. We do not need to do anything else        
        '''
        S = 0
        
        for i in range(len(sequence)): #FIRST LOOP: iterate through the positions of the sequence
            # initialize the scaling factor to 0
            s_ii = 0
            # create an empty dictionary that will store the probability of getting to each possible state in position i of the sequence and generating the observed character (nucleotide)
            f_ii = {}
            '''
            comes from i
            '''
            for key_ii in f_i: # SECOND LOOP: iterate through all the possible states in position i
                '''
                goes to ii
                '''
                # initialize the probability of the current cell (which is defined by the current state (key_ii) and the current position in the sequence (i)) to 0
                a = 0
                for key_i in f_i: # THIRD LOOP: iterate through all the possible previous states (i.e. from which previous state we came to position i)
                    '''
                    probability that I move from key_i to key_ii in the position i+1
                    multiplied by the stored scaled f_i at key i 
                    '''
                    a = a + transition[key_i][key_ii]*f_i[key_i] # we add the computation for all possibilities of previous states
                
                '''
                Probability of, being at key_ii getting the category observed at sequence i
                '''
                a = a*emission[key_ii][sequence[i]] # multiply 'a' by the probability of producing the observed category given the current state.
                '''
                before re-scaling by the total s_ii
                '''
                f_ii[key_ii] = a # new probability for the current state in position i.
                '''
                Update the s_ii
                '''
                s_ii = s_ii + a # update the scaling factor (which is the sum of the probabilities in each possible state in position i)
                
            # Add the logarithm of the scaling factor into the variable 'S' (at the end, when we finish all the iterations, it will be equal to the log probability of the sequence given the HMM)
            S = S + math.log(s_ii)

            '''
            Once we have computed all the f_ii, we need to re_scale by the s_ii and set this as the new f_i
            '''    
            for key_ii in f_ii: # rescale the probability (value) we have in each cell (each possible state in the current position of the sequence) by the scaling factor s_ii
                f_i[key_ii] = f_ii[key_ii]/s_ii # we divide each probability by the total sum of all the probabilities (so they add up to 1)       
        
        # return the log probability of the sequence
        return(S)

    def make_copy(self, vector_logits):
        # The only parameter that changes during the initialization of the object is 'vector_logits' (the other parameters are copies of the current ones)
        return Solution(vector_logits,self.legend_vector,self.states, self.categories)

    # Define a function to estimate the likelihood using the forward algorithm (scale algorithm, Rabiner)
    def compute_likelihood(self, sequence = [1, 2, 3, 4, 5, 6, 5, 4, 1, 2, 3], prior_probabilities =  {"F":0.5, "L":0.5}):
        emission, transition = self.logit_to_prob()
        likelihood = self.forward_algorithm(sequence, prior_probabilities, emission, transition)
        self.likelihood = likelihood
        return likelihood


class GeneticAlgorithm(object):
    def __init__(self, observed_sequence, states, categories, population_size):
        '''
        Inputs that the constructor needs to receive:
            1. 'states': is a list containing the names of the different states of the HMM. In the dishonest casino problem: states = ['F', 'L'])
            2. 'categories': is a list containing the names of the different categories of the HMM. In the dishonest casino problem: categories = [1,2,3,4,5,6]
            3. 'observed_sequence'
        '''
        # Create a variable to store the input list with the names of the states
        self.states = states
        # Create a variable to store the number of states of the HMM
        self.num_states = len(states)
        # Create a variable to store the input list with the names of the categories
        self.categories = categories
        # Create a variable to store the number of states of the HMM
        self.num_categories = len(categories)

        self.sequence = observed_sequence
        self.population_size = population_size
        # Create a list 'self.current_solutions' whose elements (items) will be the candidate solutions (Solution objects)
        self.current_solutions = []
        # Plot
        self.best_likelihoods = []
    
    # Define a function to generate (initialize) a certain number of solutions inside the population
    def initialize_population(self):
        # Compute the solution size: i.e., the length (number of logits) of each solution. In the dishonest casino problem: solution_size = (6-1)*2 + (2-1)*2 = 12
        solution_size = (self.num_categories - 1)*self.num_states + (self.num_states-1)*self.num_states
        # Compute at which points of the vector of logits each type of probability (emission or transition) of a given state ends
        emission_per_state = (self.num_categories - 1)
        transition_per_state = (self.num_states-1)
        self.legend_vector = [emission_per_state, transition_per_state]
        # For loop repeated as many times as solutions will be created inside the population
        for x in range(self.population_size):
            # Create a variable to store a list that is going to contain the logits of the current solution
            vector_logits = []
            # For loop repeated as many times as elements will be inside the current solution being created
            for _ in range(solution_size):
                # Draw a logit (random sample from the normal distribution with mean 0 and SD 1) and add it to the solution
                vector_logits.append(random.gauss(0, 1))
            # Create a new entry in the dictionary of solutions for the created solution
            self.current_solutions.append(Solution(vector_logits, self.legend_vector, self.states, self.categories))
        # Return the dictionary containing the initialized population of solutions
        return self.current_solutions
    
    def evaluate_population(self, discard_rate = 0.5):
        # For each solution, compute the likelihood (it is stored inside the Solution object as an attribute)
        for sol in range(len(self.current_solutions)):
            self.current_solutions[sol].compute_likelihood(self.sequence, {"F":0.5, "L":0.5})
        # Rank the solutions from highest to smallest log-likelihood
        self.current_solutions = sorted(self.current_solutions, key=lambda x: x.likelihood, reverse=True)
        number_solutions = len(self.current_solutions)
        self.current_solutions = self.current_solutions[:int(number_solutions*discard_rate)]
        self.best_likelihoods.append(math.exp(self.current_solutions[0].likelihood))
        return self.current_solutions
    
    def crossover(self, parent1, parent2):
        # Randomly pick the crossover point
        crossover_point = random.randint(0, min(len(parent1.get_logits()), len(parent2.get_logits())) - 1)

        # Perform the crossover to generate the 2 childs
        child1_vector = parent1.get_logits()[:crossover_point] + parent2.get_logits()[crossover_point:]
        child1 = parent1.make_copy(child1_vector)
        child2_vector = parent2.get_logits()[:crossover_point] + parent1.get_logits()[crossover_point:]
        child2 = parent2.make_copy(child2_vector)

        return child1, child2

    def mutation(self, solution, sd=0.1):
        logits = solution.get_logits()

        # Pick a random position from the vector (chromosome)
        index = random.randint(0, len(logits) - 1)

        # Take a random sample from the normal distribution whose mean is the picked position and whose SD is a value we choose
        mutation = random.gauss(logits[index], sd)

        # Add the sampled value to the value at the chosen index
        logits[index] = mutation

        solution.update_logits(logits)

        return solution

    def next_generation(self, p_crossover = 0.5, p_mutation = 0.5):
        new_population = []
        # While the population size is not reached, match 2 solutions
        while(len(new_population) < self.population_size):
            # Randomly choose 2 parents from the population
            parent1, parent2 = random.sample(self.current_solutions, 2)
            # Crossover is only performed in probability
            if random.random() < p_crossover:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            # Mutation is only performed in probability
            if random.random() < p_mutation:
                self.mutation(child1)
            if random.random() < p_crossover:
                self.mutation(child2)
            # Add the 2 childs to the new population
            new_population.append(child1)
            new_population.append(child2)
        # Replace the old population by the new one (next_generation)
        self.current_solutions = new_population
        return self.current_solutions

    def get_all_logits(self):
        result = []
        for sol in range(len(self.current_solutions)):
            result.append(self.current_solutions[sol].get_logits())
        return result

    # Create a function (method) to retrieve the logits of a specific solution
    def get_logits_solution(self,index):
        # The logits are retrieved by indexing through the name (number) of the solution
        return self.current_solutions[index]

    def get_trajectory_likelihoods(self):
        for x in self.best_likelihoods:
            print(x)
        return self.best_likelihoods

def main():

    # Create an object of the class GeneticAlgorithm
    genetic_algorithm = GeneticAlgorithm([1, 2, 3, 4, 5, 6, 5, 4, 1, 2, 3, 5, 2, 4, 5, 5, 1], ['F', 'L'], [1,2,3,4,5,6], 20)
    # (1) Create an initial population of solutions
    genetic_algorithm.initialize_population()
    for _ in range(2000):
        # (2) Calculate the fitness (likelihood) of each solution, rank them accordingly and discard the ones with worst likelihood
        genetic_algorithm.evaluate_population()
        genetic_algorithm.next_generation(p_crossover=0.5, p_mutation=0.5)
    y_axis = genetic_algorithm.get_trajectory_likelihoods()
    x_axis = range(2000)
    plt.plot(x_axis, y_axis)
    plt.title('Optimization Process Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Best Likelihood')
    plt.show()

if __name__ == "__main__":
    main()
