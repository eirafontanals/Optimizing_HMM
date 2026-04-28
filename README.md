# Optimizing Hidden Markov Model Parameters Using Genetic Algorithms

This repository contains a Python script that optimizes the parameters of a Hidden Markov Model (HMM) using a genetic algorithm. The aim of this project is to demonstrate how genetic algorithms can be applied to fine-tune the parameters of an HMM for better performance in specific tasks.

## Project Overview

A Hidden Markov Model (HMM) is a statistical model that assumes an underlying process governed by unobservable (hidden) states. These models are widely used in various fields, including time-series analysis, speech recognition, and bioinformatics.

The script in this repository uses a **genetic algorithm (GA)** to optimize the parameters of an HMM, specifically:
- Transition probabilities between states
- Emission probabilities (probabilities of observing a particular output)

By applying the genetic algorithm, the model searches for the best parameter values that maximize the likelihood of the observed data, improving the model's performance in real-world applications.

### The Dishonest Casino Problem

In the **Dishonest Casino** scenario, we want to optimize the parameters of a Hidden Markov Model that simulates a casino game with biased probabilities. The goal is to find the best transition and emission probabilities that maximize the likelihood of observed sequences. This problem involves using the Genetic Algorithm to tune the parameters of the HMM model.

## Algorithms Used

The genetic algorithm is a metaheuristic that mimics the process of natural selection. It has been developed to solve optimization problems in situations where the objective function is complex or unknown.

In our implementation:
1. **Initial Population**: Random sets of parameters (logits) are generated.
2. **Evaluation**: The fitness of each individual solution is assessed based on its likelihood.
3. **Selection**: The best-performing solutions are kept, and the worst are discarded.
4. **Crossover**: Two parent solutions are combined to generate new solutions.
5. **Mutation**: Random alterations are applied to the parameters.

### Key Steps of the Genetic Algorithm
1. **Create an Initial Population**: The initial population of candidate solutions is created by sampling random logits from a normal distribution centered at 0 with a standard deviation of 1.
2. **Evaluate Population**: For each solution, the logits are decoded into probabilities (transition and emission probabilities) and the likelihood of the observed sequence is computed using the forward algorithm.
3. **Selection**: The population is sorted by fitness (likelihood), and the worst half of the solutions are discarded.
4. **Crossover**: Randomly selected pairs of solutions are combined by a crossover operation, exchanging parts of their genetic material (logit vectors).
5. **Mutation**: A random mutation is applied to the selected solutions, introducing slight changes in the parameters.
6. **Next Generation**: The new population is created and evaluated in the next generation.

### Mathematical Considerations
When applying the genetic algorithm to HMMs, the transition and emission probabilities must satisfy the following conditions:
- The parameters must sum to 1 (since they represent probabilities).
- The parameters must remain between 0 and 1.

To ensure these conditions hold after applying the genetic algorithm, we use a **logit transformation** of the parameters before encoding them into the genetic algorithm.

### Solution Class and Forward Algorithm
The **Solution** class represents each individual solution in the population. It includes methods for decoding logits into probabilities, computing likelihoods, and applying the forward algorithm to evaluate the fitness of the solution. The **forward algorithm** computes the likelihood of observing a given sequence based on the HMM parameters.

## Results

### Visualization of the Likelihood

In the first stages of the genetic algorithm, we observe rapid growth in the likelihood values as the algorithm explores a wide range of possible solutions. This exploration phase involves testing solutions that are not yet optimal but may lead to better solutions later.

As the algorithm progresses, the rate of growth slows down as the algorithm begins to **specialize in regions with more optimal solutions**.

Below are the visualizations of the likelihood values for each generation and the convergence of the algorithm over time.

### Terminal Printout of Likelihood:

On the left, we can observe the printout in the terminal showing the **best likelihood for each of the generations**, providing us with a detailed view of the optimization process. On the right, we have the plot of these values over time.

## Discussion

The results show that the genetic algorithm is an effective approach to optimize the parameters of the HMM. Initially, the algorithm explores a wide range of solutions, but as it progresses, it refines the search and converges to optimal solutions.

This approach can be applied to other optimization problems where the objective function is complex or not easily derivable.

## Conclusions

In conclusion, we have successfully implemented the genetic algorithm to optimize the parameters of a Hidden Markov Model for the "Dishonest Casino" problem. We demonstrated that the genetic algorithm can effectively explore the solution space, improve the solution quality over generations, and converge to optimal parameters. This project highlights the power of the genetic algorithm in solving complex optimization problems.

## How to Run the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Optimizing_HMM.git
   cd Optimizing_HMM
   ```

2. Install the required Python packages:

   ```bash
   pip install numpy hmmlearn matplotlib scipy
   ```

3. Run the Python script:

   ```bash
   python def_genetic_algorithm.py
   ```

This will execute the genetic algorithm, and the optimized parameters will be printed (or saved, depending on how you choose to handle output).

