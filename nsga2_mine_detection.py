"""
COMP5012 - Computational Intelligence
Multi-Objective Optimisation for Underwater Mine Detection using NSGA-II
Author: Apostolos Zymvrakakis
University of Plymouth
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cost_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n = int(lines[0].strip())
    costs = []
    for line in lines[1:]:
        costs.extend([int(x) for x in line.split()])

    cost_matrix = np.array(costs).reshape(n, n)
    return cost_matrix


def generate_probability_matrix(n, seed=42):
    np.random.seed(seed)
    return np.random.beta(5, 2, size=(n, n))


# =============================================================================
# SOLUTION CLASS
# =============================================================================

class Solution:
    def __init__(self, permutation, cost_matrix, prob_matrix):
        self.permutation = np.array(permutation)
        self.cost = sum(cost_matrix[drone, area]
                        for area, drone in enumerate(self.permutation))
        self.detection = sum(prob_matrix[drone, area]
                             for area, drone in enumerate(self.permutation))
        self.rank = None
        self.crowding_distance = 0


def dominates(sol1, sol2):
    better_cost = sol1.cost <= sol2.cost
    better_detection = sol1.detection >= sol2.detection
    strictly_better = (sol1.cost < sol2.cost) or (sol1.detection > sol2.detection)
    return better_cost and better_detection and strictly_better


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def tournament_selection(population, tournament_size=2):
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda x: (x.rank, -x.crowding_distance))
    return selected[0]


def pmx_crossover(parent1, parent2):
    n = len(parent1.permutation)
    child1 = np.full(n, -1)
    child2 = np.full(n, -1)

    cx1, cx2 = sorted(random.sample(range(n), 2))

    child1[cx1:cx2 + 1] = parent1.permutation[cx1:cx2 + 1]
    child2[cx1:cx2 + 1] = parent2.permutation[cx1:cx2 + 1]

    def fill(child, donor):
        for i in range(n):
            if child[i] == -1:
                value = donor[i]
                while value in child:
                    idx = np.where(donor == value)[0][0]
                    value = child[idx] if child[idx] != -1 else donor[idx]
                    if value == donor[i]:
                        for v in donor:
                            if v not in child:
                                value = v
                                break
                child[i] = value

    fill(child1, parent2.permutation)
    fill(child2, parent1.permutation)
    return child1, child2


def swap_mutation(permutation, mutation_rate):
    mutated = permutation.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# =============================================================================
# NSGA-II CORE FUNCTIONS
# =============================================================================

def non_dominated_sorting(population):
    n = len(population)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(population[i], population[j]):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif dominates(population[j], population[i]):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current + 1
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(population, front_indices):
    if len(front_indices) <= 2:
        for i in front_indices:
            population[i].crowding_distance = float('inf')
        return

    for i in front_indices:
        population[i].crowding_distance = 0

    for obj in ['cost', 'detection']:
        sorted_idx = sorted(front_indices, key=lambda x: getattr(population[x], obj))
        min_val = getattr(population[sorted_idx[0]], obj)
        max_val = getattr(population[sorted_idx[-1]], obj)

        population[sorted_idx[0]].crowding_distance = float('inf')
        population[sorted_idx[-1]].crowding_distance = float('inf')

        if max_val - min_val > 0:
            for i in range(1, len(sorted_idx) - 1):
                prev = getattr(population[sorted_idx[i - 1]], obj)
                next_val = getattr(population[sorted_idx[i + 1]], obj)
                population[sorted_idx[i]].crowding_distance += (next_val - prev) / (max_val - min_val)


def nsga2(cost_matrix, prob_matrix, pop_size, generations, mutation_rate,
          crossover_rate=0.9, track_history=True):
    n = len(cost_matrix)

    population = []
    for _ in range(pop_size):
        perm = np.random.permutation(n)
        population.append(Solution(perm, cost_matrix, prob_matrix))

    history = {'generation': [], 'pareto_size': [], 'best_cost': [], 'best_detection': []}

    fronts = non_dominated_sorting(population)
    for front in fronts:
        crowding_distance(population, front)

    for gen in range(generations):
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            if random.random() < crossover_rate:
                c1, c2 = pmx_crossover(p1, p2)
            else:
                c1, c2 = p1.permutation.copy(), p2.permutation.copy()

            c1 = swap_mutation(c1, mutation_rate)
            c2 = swap_mutation(c2, mutation_rate)

            offspring.append(Solution(c1, cost_matrix, prob_matrix))
            if len(offspring) < pop_size:
                offspring.append(Solution(c2, cost_matrix, prob_matrix))

        combined = population + offspring
        fronts = non_dominated_sorting(combined)

        new_pop = []
        front_idx = 0
        while len(new_pop) + len(fronts[front_idx]) <= pop_size:
            for i in fronts[front_idx]:
                new_pop.append(combined[i])
            front_idx += 1
            if front_idx >= len(fronts):
                break

        if len(new_pop) < pop_size and front_idx < len(fronts):
            crowding_distance(combined, fronts[front_idx])
            remaining = sorted(fronts[front_idx],
                               key=lambda x: combined[x].crowding_distance, reverse=True)
            for i in remaining[:pop_size - len(new_pop)]:
                new_pop.append(combined[i])

        population = new_pop

        fronts = non_dominated_sorting(population)
        for front in fronts:
            crowding_distance(population, front)

        if track_history:
            pareto = [p for p in population if p.rank == 0]
            history['generation'].append(gen)
            history['pareto_size'].append(len(pareto))
            history['best_cost'].append(min(p.cost for p in population))
            history['best_detection'].append(max(p.detection for p in population))

    return population, history


# =============================================================================
# CONFIGURATION - Experiment Parameters
# =============================================================================

# Experiment A: Statistical Validation (Multiple Runs)
N_RUNS = 30
POP_SIZE_A = 200
GENERATIONS_A = 200
MUTATION_RATE_A = 0.1

# Experiment B: Parameter Sensitivity
POP_SIZES = [50, 100, 200]
GENERATION_VALUES = [50, 100, 200]
MUTATION_RATES = [0.05, 0.1, 0.2]
N_RUNS_B = 5

# Experiment C: Convergence Analysis
GENERATIONS_C = 200
POP_SIZE_C = 200

# General Settings
CROSSOVER_RATE = 0.9
RANDOM_SEED = 42


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiment_A(cost_matrix, prob_matrix):
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Statistical Validation (30 runs)")
    print("=" * 60)
    print(f"Parameters: pop={POP_SIZE_A}, gen={GENERATIONS_A}, mut={MUTATION_RATE_A}")
    print("-" * 60)

    results = {'pareto_sizes': [], 'best_costs': [], 'best_detections': [], 'times': []}

    for run in range(N_RUNS):
        np.random.seed(run * 42)
        random.seed(run * 42)

        start = time.time()
        population, _ = nsga2(cost_matrix, prob_matrix,
                              POP_SIZE_A, GENERATIONS_A, MUTATION_RATE_A,
                              track_history=False)
        elapsed = time.time() - start

        pareto = [p for p in population if p.rank == 0]

        results['pareto_sizes'].append(len(pareto))
        results['best_costs'].append(min(p.cost for p in pareto))
        results['best_detections'].append(max(p.detection for p in pareto))
        results['times'].append(elapsed)

        print(f"  Run {run + 1}/{N_RUNS}: Pareto={len(pareto)}, "
              f"Cost=[{min(p.cost for p in pareto):.0f}], "
              f"Time={elapsed:.2f}s")

    print("\n" + "-" * 60)
    print("STATISTICS:")
    print(f"  Pareto Size: {np.mean(results['pareto_sizes']):.1f} ± {np.std(results['pareto_sizes']):.1f}")
    print(f"  Best Cost:   {np.mean(results['best_costs']):.0f} ± {np.std(results['best_costs']):.0f}")
    print(f"  Best Det:    {np.mean(results['best_detections']):.2f} ± {np.std(results['best_detections']):.2f}")
    print(f"  Avg Time:    {np.mean(results['times']):.2f}s")

    return results


def run_experiment_B(cost_matrix, prob_matrix):
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Parameter Sensitivity")
    print("=" * 60)

    results = {}

    print("\n--- B1: Effect of Population Size ---")
    for pop_size in POP_SIZES:
        pareto_sizes = []
        for run in range(N_RUNS_B):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix,
                                  pop_size, 100, 0.1, track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))

        results[f'pop_{pop_size}'] = np.mean(pareto_sizes)
        print(f"  Pop={pop_size}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")

    print("\n--- B2: Effect of Generations ---")
    for gen in GENERATION_VALUES:
        pareto_sizes = []
        for run in range(N_RUNS_B):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix,
                                  100, gen, 0.1, track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))

        results[f'gen_{gen}'] = np.mean(pareto_sizes)
        print(f"  Gen={gen}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")

    print("\n--- B3: Effect of Mutation Rate ---")
    for mut in MUTATION_RATES:
        pareto_sizes = []
        for run in range(N_RUNS_B):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix,
                                  100, 100, mut, track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))

        results[f'mut_{mut}'] = np.mean(pareto_sizes)
        print(f"  Mut={mut}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")

    return results


def run_experiment_C(cost_matrix, prob_matrix):
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Convergence Analysis")
    print("=" * 60)
    print(f"Parameters: pop={POP_SIZE_C}, gen={GENERATIONS_C}")
    print("-" * 60)

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    population, history = nsga2(cost_matrix, prob_matrix,
                                POP_SIZE_C, GENERATIONS_C, 0.1,
                                track_history=True)

    sizes = history['pareto_size']
    converged_gen = GENERATIONS_C
    for i in range(len(sizes) - 10):
        if abs(sizes[i] - sizes[-1]) < 5:
            converged_gen = i
            break
    print(f"  Convergence at generation: ~{converged_gen}")

    print(f"\nResults after {GENERATIONS_C} generations:")
    print(f"  Pareto Front Size: {history['pareto_size'][-1]}")
    print(f"  Best Cost: {history['best_cost'][-1]:.0f}")
    print(f"  Best Detection: {history['best_detection'][-1]:.2f}")

    return history, population


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(results_A, results_B, history_C, population_C):
    # Figure 2: Multiple Runs Results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(results_A['pareto_sizes'], bins=10, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(results_A['pareto_sizes']), color='red', linestyle='--',
                    label=f'Mean: {np.mean(results_A["pareto_sizes"]):.1f}')
    axes[0].set_xlabel('Pareto Front Size')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Pareto Front Sizes')
    axes[0].legend()

    axes[1].hist(results_A['best_costs'], bins=10, color='coral', edgecolor='black')
    axes[1].set_xlabel('Best Cost')
    axes[1].set_title('Distribution of Best Costs')

    axes[2].hist(results_A['best_detections'], bins=10, color='forestgreen', edgecolor='black')
    axes[2].set_xlabel('Best Detection')
    axes[2].set_title('Distribution of Best Detections')

    plt.tight_layout()
    plt.savefig('exp_A_multiple_runs.png', dpi=300)
    plt.close()
    print("Saved: exp_A_multiple_runs.png")

    # Figure 3: Convergence Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history_C['generation'], history_C['pareto_size'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Pareto Front Size')
    axes[0, 0].set_title('Pareto Front Size Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_C['generation'], history_C['best_cost'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Best Cost')
    axes[0, 1].set_title('Best Cost Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_C['generation'], history_C['best_detection'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Best Detection')
    axes[1, 0].set_title('Best Detection Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Final Pareto Front
    pareto = [p for p in population_C if p.rank == 0]
    costs = [p.cost for p in pareto]
    dets = [p.detection for p in pareto]
    sorted_points = sorted(zip(costs, dets))
    x, y = zip(*sorted_points)

    axes[1, 1].scatter(x, y, c='red', s=30)
    axes[1, 1].plot(x, y, 'r-', alpha=0.5)
    axes[1, 1].set_xlabel('Cost (Minimize)')
    axes[1, 1].set_ylabel('Detection (Maximize)')
    axes[1, 1].set_title(f'Final Pareto Front ({len(pareto)} solutions)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp_C_convergence.png', dpi=300)
    plt.close()
    print("Saved: exp_C_convergence.png")

    # Figure 4: Parameter Sensitivity Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pop_vals = [results_B.get(f'pop_{p}', 0) for p in POP_SIZES]
    axes[0].bar(range(len(POP_SIZES)), pop_vals, color='steelblue')
    axes[0].set_xticks(range(len(POP_SIZES)))
    axes[0].set_xticklabels(POP_SIZES)
    axes[0].set_xlabel('Population Size')
    axes[0].set_ylabel('Pareto Front Size')
    axes[0].set_title('Effect of Population Size')

    gen_vals = [results_B.get(f'gen_{g}', 0) for g in GENERATION_VALUES]
    axes[1].bar(range(len(GENERATION_VALUES)), gen_vals, color='coral')
    axes[1].set_xticks(range(len(GENERATION_VALUES)))
    axes[1].set_xticklabels(GENERATION_VALUES)
    axes[1].set_xlabel('Generations')
    axes[1].set_title('Effect of Generations')

    mut_vals = [results_B.get(f'mut_{m}', 0) for m in MUTATION_RATES]
    axes[2].bar(range(len(MUTATION_RATES)), mut_vals, color='forestgreen')
    axes[2].set_xticks(range(len(MUTATION_RATES)))
    axes[2].set_xticklabels(MUTATION_RATES)
    axes[2].set_xlabel('Mutation Rate')
    axes[2].set_title('Effect of Mutation Rate')

    plt.tight_layout()
    plt.savefig('exp_B_parameters.png', dpi=300)
    plt.close()
    print("Saved: exp_B_parameters.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMP5012 - Multi-Objective Optimisation for Underwater Mine Detection")
    print("Algorithm: NSGA-II | Dataset: OR-Library assign100.txt")
    print("=" * 70)

    print("\nLoading data...")
    cost_matrix = load_cost_matrix("assign100.txt")
    prob_matrix = generate_probability_matrix(100, seed=42)
    print(f"  Cost matrix: {cost_matrix.shape}")
    print(f"  Probability matrix: {prob_matrix.shape}")

    # Run experiments
    results_A = run_experiment_A(cost_matrix, prob_matrix)
    results_B = run_experiment_B(cost_matrix, prob_matrix)
    history_C, population_C = run_experiment_C(cost_matrix, prob_matrix)

    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    create_visualizations(results_A, results_B, history_C, population_C)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nFiles created:")
    print("  - exp_A_multiple_runs.png")
    print("  - exp_B_parameters.png")
    print("  - exp_C_convergence.png")
