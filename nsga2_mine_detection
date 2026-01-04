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


def load_cost_matrix(filepath: str = "assign100.txt") -> np.ndarray:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    values = []
    for line in lines[1:]:
        values.extend([int(x) for x in line.split()])
    return np.array(values[:n*n]).reshape(n, n)


def generate_probability_matrix(n: int, alpha: float = 5, beta: float = 2, 
                                 seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.beta(alpha, beta, size=(n, n))


class Solution:
    def __init__(self, permutation: np.ndarray, cost_matrix: np.ndarray, 
                 prob_matrix: np.ndarray):
        self.permutation = np.array(permutation)
        n = len(permutation)
        self.cost = sum(cost_matrix[drone, area] 
                       for area, drone in enumerate(self.permutation))
        self.detection = sum(prob_matrix[drone, area] 
                            for area, drone in enumerate(self.permutation))
        self.rank = None
        self.crowding_distance = 0.0


def dominates(sol1: Solution, sol2: Solution) -> bool:
    better_cost = sol1.cost <= sol2.cost
    better_detection = sol1.detection >= sol2.detection
    strictly_better = (sol1.cost < sol2.cost) or (sol1.detection > sol2.detection)
    return better_cost and better_detection and strictly_better


def tournament_selection(population: List[Solution], tournament_size: int = 2) -> Solution:
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda x: (x.rank if x.rank is not None else float('inf'), 
                                  -x.crowding_distance))
    return selected[0]


def pmx_crossover(parent1: Solution, parent2: Solution, 
                  cost_matrix: np.ndarray, prob_matrix: np.ndarray) -> Tuple[Solution, Solution]:
    n = len(parent1.permutation)
    p1 = parent1.permutation
    p2 = parent2.permutation
    
    cx1, cx2 = sorted(random.sample(range(n), 2))
    
    child1 = np.full(n, -1, dtype=int)
    child2 = np.full(n, -1, dtype=int)
    
    child1[cx1:cx2+1] = p1[cx1:cx2+1]
    child2[cx1:cx2+1] = p2[cx1:cx2+1]
    
    def fill_child(child, segment_parent, other_parent):
        used = set(child[child != -1])
        other_idx = 0
        for i in range(n):
            if child[i] == -1:
                while other_parent[other_idx] in used:
                    other_idx += 1
                child[i] = other_parent[other_idx]
                used.add(other_parent[other_idx])
                other_idx += 1
        return child
    
    child1 = fill_child(child1, p1, p2)
    child2 = fill_child(child2, p2, p1)
    
    return (Solution(child1, cost_matrix, prob_matrix), 
            Solution(child2, cost_matrix, prob_matrix))


def swap_mutation(solution: Solution, mutation_rate: float,
                  cost_matrix: np.ndarray, prob_matrix: np.ndarray) -> Solution:
    if random.random() < mutation_rate:
        perm = solution.permutation.copy()
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return Solution(perm, cost_matrix, prob_matrix)
    return solution


def non_dominated_sorting(population: List[Solution]) -> List[List[int]]:
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


def crowding_distance(population: List[Solution], front_indices: List[int]) -> None:
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
                prev_val = getattr(population[sorted_idx[i-1]], obj)
                next_val = getattr(population[sorted_idx[i+1]], obj)
                population[sorted_idx[i]].crowding_distance += \
                    (next_val - prev_val) / (max_val - min_val)


def nsga2(cost_matrix: np.ndarray, prob_matrix: np.ndarray,
          pop_size: int, generations: int, mutation_rate: float,
          crossover_rate: float = 0.9, track_history: bool = True) -> Tuple[List[Solution], dict]:
    
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
                c1, c2 = pmx_crossover(p1, p2, cost_matrix, prob_matrix)
            else:
                c1 = Solution(p1.permutation.copy(), cost_matrix, prob_matrix)
                c2 = Solution(p2.permutation.copy(), cost_matrix, prob_matrix)
            
            c1 = swap_mutation(c1, mutation_rate, cost_matrix, prob_matrix)
            c2 = swap_mutation(c2, mutation_rate, cost_matrix, prob_matrix)
            
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)
        
        combined = population + offspring
        fronts = non_dominated_sorting(combined)
        
        new_pop = []
        front_idx = 0
        
        while front_idx < len(fronts) and len(new_pop) + len(fronts[front_idx]) <= pop_size:
            for i in fronts[front_idx]:
                new_pop.append(combined[i])
            front_idx += 1
        
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


def run_experiment_A(cost_matrix: np.ndarray, prob_matrix: np.ndarray,
                     n_runs: int = 30, pop_size: int = 200, 
                     generations: int = 200, mutation_rate: float = 0.1) -> dict:
    
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Statistical Validation (30 runs)")
    print("=" * 60)
    print(f"Parameters: pop={pop_size}, gen={generations}, mut={mutation_rate}")
    print("-" * 60)
    
    results = {'pareto_sizes': [], 'best_costs': [], 'best_detections': [], 'times': []}
    
    for run in range(n_runs):
        np.random.seed(run * 42)
        random.seed(run * 42)
        
        start = time.time()
        population, _ = nsga2(cost_matrix, prob_matrix, pop_size, generations, 
                             mutation_rate, track_history=False)
        elapsed = time.time() - start
        
        pareto = [p for p in population if p.rank == 0]
        
        results['pareto_sizes'].append(len(pareto))
        results['best_costs'].append(min(p.cost for p in pareto))
        results['best_detections'].append(max(p.detection for p in pareto))
        results['times'].append(elapsed)
        
        print(f"  Run {run+1:2d}/{n_runs}: Pareto={len(pareto):3d}, "
              f"Cost={min(p.cost for p in pareto):5.0f}, "
              f"Det={max(p.detection for p in pareto):.2f}, "
              f"Time={elapsed:.1f}s")
    
    print("\n" + "-" * 60)
    print("STATISTICS:")
    print(f"  Pareto Size: {np.mean(results['pareto_sizes']):.1f} ± {np.std(results['pareto_sizes']):.1f}")
    print(f"  Best Cost:   {np.mean(results['best_costs']):.0f} ± {np.std(results['best_costs']):.0f}")
    print(f"  Best Det:    {np.mean(results['best_detections']):.2f} ± {np.std(results['best_detections']):.2f}")
    print(f"  Avg Time:    {np.mean(results['times']):.2f}s")
    
    return results


def run_experiment_B(cost_matrix: np.ndarray, prob_matrix: np.ndarray,
                     n_runs: int = 5) -> dict:
    
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Parameter Sensitivity")
    print("=" * 60)
    
    results = {}
    
    print("\n--- B1: Effect of Population Size ---")
    for pop_size in [50, 100, 200]:
        pareto_sizes = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix, pop_size, 100, 0.1, 
                                 track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))
        
        results[f'pop_{pop_size}'] = np.mean(pareto_sizes)
        print(f"  Pop={pop_size}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")
    
    print("\n--- B2: Effect of Generations ---")
    for gen in [50, 100, 200]:
        pareto_sizes = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix, 100, gen, 0.1,
                                 track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))
        
        results[f'gen_{gen}'] = np.mean(pareto_sizes)
        print(f"  Gen={gen}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")
    
    print("\n--- B3: Effect of Mutation Rate ---")
    for mut in [0.05, 0.1, 0.2]:
        pareto_sizes = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            random.seed(run * 42)
            population, _ = nsga2(cost_matrix, prob_matrix, 100, 100, mut,
                                 track_history=False)
            pareto = [p for p in population if p.rank == 0]
            pareto_sizes.append(len(pareto))
        
        results[f'mut_{mut}'] = np.mean(pareto_sizes)
        print(f"  Mut={mut}: Pareto={np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")
    
    return results


def run_experiment_C(cost_matrix: np.ndarray, prob_matrix: np.ndarray,
                     pop_size: int = 200, generations: int = 200) -> Tuple[dict, List[Solution]]:
    
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Convergence Analysis")
    print("=" * 60)
    print(f"Parameters: pop={pop_size}, gen={generations}")
    print("-" * 60)
    
    np.random.seed(42)
    random.seed(42)
    
    population, history = nsga2(cost_matrix, prob_matrix, pop_size, generations, 0.1,
                               track_history=True)
    
    sizes = history['pareto_size']
    converged_gen = generations
    for i in range(len(sizes) - 10):
        if abs(sizes[i] - sizes[-1]) < 5:
            converged_gen = i
            break
    
    print(f"\nResults after {generations} generations:")
    print(f"  Pareto Front Size: {history['pareto_size'][-1]}")
    print(f"  Best Cost: {history['best_cost'][-1]:.0f}")
    print(f"  Best Detection: {history['best_detection'][-1]:.2f}")
    print(f"  Convergence at generation: ~{converged_gen}")
    
    return history, population


def create_visualizations(results_A: dict, results_B: dict, 
                          history_C: dict, population_C: List[Solution]) -> None:
    
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
    plt.savefig('figure2_multiple_runs.png', dpi=300)
    plt.close()
    print("Saved: figure2_multiple_runs.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    pop_sizes = [50, 100, 200]
    pop_vals = [results_B.get(f'pop_{p}', 0) for p in pop_sizes]
    axes[0].bar(range(len(pop_sizes)), pop_vals, color='steelblue')
    axes[0].set_xticks(range(len(pop_sizes)))
    axes[0].set_xticklabels(pop_sizes)
    axes[0].set_xlabel('Population Size')
    axes[0].set_ylabel('Pareto Front Size')
    axes[0].set_title('Effect of Population Size')
    
    gen_vals_list = [50, 100, 200]
    gen_vals = [results_B.get(f'gen_{g}', 0) for g in gen_vals_list]
    axes[1].bar(range(len(gen_vals_list)), gen_vals, color='coral')
    axes[1].set_xticks(range(len(gen_vals_list)))
    axes[1].set_xticklabels(gen_vals_list)
    axes[1].set_xlabel('Generations')
    axes[1].set_title('Effect of Generations')
    
    mut_rates = [0.05, 0.1, 0.2]
    mut_vals = [results_B.get(f'mut_{m}', 0) for m in mut_rates]
    axes[2].bar(range(len(mut_rates)), mut_vals, color='forestgreen')
    axes[2].set_xticks(range(len(mut_rates)))
    axes[2].set_xticklabels(mut_rates)
    axes[2].set_xlabel('Mutation Rate')
    axes[2].set_title('Effect of Mutation Rate')
    
    plt.tight_layout()
    plt.savefig('figure3_parameters.png', dpi=300)
    plt.close()
    print("Saved: figure3_parameters.png")
    
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
    
    pareto = [p for p in population_C if p.rank == 0]
    costs = [p.cost for p in pareto]
    dets = [p.detection for p in pareto]
    sorted_points = sorted(zip(costs, dets))
    if sorted_points:
        x, y = zip(*sorted_points)
        axes[1, 1].scatter(x, y, c='red', s=30)
        axes[1, 1].plot(x, y, 'r-', alpha=0.5)
    axes[1, 1].set_xlabel('Cost (Minimize)')
    axes[1, 1].set_ylabel('Detection (Maximize)')
    axes[1, 1].set_title(f'Final Pareto Front ({len(pareto)} solutions)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure4_convergence.png', dpi=300)
    plt.close()
    print("Saved: figure4_convergence.png")


if __name__ == "__main__":
    print("=" * 70)
    print("COMP5012 - Multi-Objective Optimisation for Underwater Mine Detection")
    print("Algorithm: NSGA-II | Dataset: OR-Library assign100.txt")
    print("=" * 70)
    
    print("\nLoading data...")
    try:
        cost_matrix = load_cost_matrix("assign100.txt")
        print(f"  Cost matrix loaded: {cost_matrix.shape}")
    except FileNotFoundError:
        print("  ERROR: assign100.txt not found!")
        print("  Please ensure assign100.txt is in the same directory.")
        exit(1)
    
    prob_matrix = generate_probability_matrix(100, seed=42)
    print(f"  Probability matrix generated: {prob_matrix.shape}")
    print(f"  Cost range: [{cost_matrix.min()}, {cost_matrix.max()}]")
    print(f"  Detection range: [{prob_matrix.min():.3f}, {prob_matrix.max():.3f}]")
    
    results_A = run_experiment_A(cost_matrix, prob_matrix, n_runs=30)
    results_B = run_experiment_B(cost_matrix, prob_matrix, n_runs=5)
    history_C, population_C = run_experiment_C(cost_matrix, prob_matrix)
    
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    create_visualizations(results_A, results_B, history_C, population_C)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment A (30 runs):")
    print(f"  Pareto Size: {np.mean(results_A['pareto_sizes']):.1f} ± {np.std(results_A['pareto_sizes']):.1f}")
    print(f"  Best Cost:   {np.mean(results_A['best_costs']):.0f} ± {np.std(results_A['best_costs']):.0f}")
    print(f"  Best Det:    {np.mean(results_A['best_detections']):.2f} ± {np.std(results_A['best_detections']):.2f}")
    
    print(f"\nFiles created:")
    print("  - figure2_multiple_runs.png")
    print("  - figure3_parameters.png")
    print("  - figure4_convergence.png")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
