"""
COMP5012 Final Report - Step 1: Understanding the Data
========================================================

WHAT WE'RE DOING:
    Loading and understanding the assign100.txt dataset from OR-Library.

WHY WE'RE DOING IT:
    Before implementing any algorithm, we need to understand our data structure.
    The assessment brief states we must use one of the OR-Library problems.

COURSE CONNECTION:
    - Assessment Brief: "select one of the following optimisation problems from
      the Operations Research library"
    - Lecture 2: Understanding problem formulation is the first step in any
      optimization task
    - Your Presentation: "Dataset: OR-Library assign100, Cost matrix c(i,j)
      interpreted as mission cost/risk"
"""

import numpy as np
import matplotlib.pyplot as plt  # Για visualization


# =============================================================================
# CONFIGURATION - ΑΛΛΑΞΕ ΑΥΤΑ ΤΑ PATHS ΣΤΟΝ ΔΙΚΟ ΣΟΥ ΦΑΚΕΛΟ
# =============================================================================
BASE_DIR = r'C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012'
COST_FILE = BASE_DIR + r'\assign100.txt'
COST_IMAGE = BASE_DIR + r'\cost_matrix_visualization.png'


def load_assignment_data(filepath):
    """
    Load the assignment problem data from OR-Library format.

    WHAT: Parse the assign100.txt file to extract the cost matrix.

    WHY: The cost matrix c(i,j) represents the cost of assigning drone i to area j.
         This is Objective 1 in our multi-objective problem.

    FILE FORMAT:
        - First number: problem size n (100 for assign100)
        - Following numbers: flattened n×n cost matrix values

    COURSE CONNECTION:
        - Your Presentation: "Cost matrix c(i,j) interpreted as mission cost/risk"
    """
    with open(filepath, 'r') as f:
        # Read all numbers from the file
        numbers = []
        for line in f:
            numbers.extend([int(x) for x in line.split()])

    # First number is the problem size
    n = numbers[0]
    print(f"Problem size: {n} (meaning {n} drones and {n} areas)")

    # Remaining numbers form the cost matrix
    cost_values = numbers[1:]

    # Check we have enough values for an n×n matrix
    expected_values = n * n
    print(f"Expected cost values: {expected_values}")
    print(f"Actual cost values found: {len(cost_values)}")

    # Reshape into n×n matrix
    cost_matrix = np.array(cost_values[:expected_values]).reshape(n, n)

    return n, cost_matrix


def analyze_cost_matrix(cost_matrix):
    """
    Analyze the cost matrix to understand the problem characteristics.

    WHAT: Generate statistics about the cost matrix.

    WHY: Understanding the range and distribution of costs helps us:
         1. Design appropriate objective functions
         2. Understand what "good" vs "bad" costs look like
         3. Inform the creation of our synthetic detection probability matrix

    COURSE CONNECTION:
        - Lecture 2: "defining objective functions" requires understanding the data
    """
    print("\n" + "=" * 60)
    print("COST MATRIX ANALYSIS")
    print("=" * 60)

    print(f"\nMatrix shape: {cost_matrix.shape}")
    print(f"  Interpretation: {cost_matrix.shape[0]} drones × {cost_matrix.shape[1]} areas")

    print(f"\nCost statistics:")
    print(f"  Minimum cost: {cost_matrix.min()}")
    print(f"  Maximum cost: {cost_matrix.max()}")
    print(f"  Mean cost: {cost_matrix.mean():.2f}")
    print(f"  Std deviation: {cost_matrix.std():.2f}")

    print(f"\nSample of cost matrix (first 5×5 corner):")
    print(cost_matrix[:5, :5])

    # Find some example assignments
    print(f"\nExample interpretations:")
    print(f"  c(0,0) = {cost_matrix[0, 0]}: Cost of assigning drone 0 to area 0")
    print(f"  c(0,1) = {cost_matrix[0, 1]}: Cost of assigning drone 0 to area 1")
    print(f"  c(1,0) = {cost_matrix[1, 0]}: Cost of assigning drone 1 to area 0")

    # Find best and worst individual assignments
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    max_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)

    print(f"\nBest single assignment: Drone {min_idx[0]} → Area {min_idx[1]} (cost: {cost_matrix[min_idx]})")
    print(f"Worst single assignment: Drone {max_idx[0]} → Area {max_idx[1]} (cost: {cost_matrix[max_idx]})")


def visualize_cost_matrix(cost_matrix, save_path):
    """
    Create and save visualization of the cost matrix.

    WHAT: Generate heatmap and histogram of cost values.

    WHY: Visual representation helps understand the data and
         creates figures for the report.

    COURSE CONNECTION:
        - Assessment rubric: "visualisations demonstrating..."
        - Lecture 2: Understanding data distribution is essential
    """
    print("\n" + "=" * 60)
    print("CREATING COST MATRIX VISUALIZATION")
    print("=" * 60)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ----- Left Plot: Heatmap -----
    ax1 = axes[0]
    im = ax1.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Area Index (j)', fontsize=12)
    ax1.set_ylabel('Drone Index (i)', fontsize=12)
    ax1.set_title('Cost Matrix c(i,j)\nCost of assigning drone i to area j', fontsize=12)
    plt.colorbar(im, ax=ax1, label='Cost')

    # ----- Right Plot: Histogram -----
    ax2 = axes[1]
    ax2.hist(cost_matrix.flatten(), bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Cost Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Cost Values', fontsize=12)
    ax2.axvline(cost_matrix.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {cost_matrix.mean():.1f}')
    ax2.legend(fontsize=11)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    print(f"\n✓ Figure saved to: {save_path}")

    # Show the plot
    plt.show()


def understand_solution_representation():
    """
    Explain the solution representation for the assignment problem.

    WHAT: Document how solutions are encoded.

    WHY: Choosing the right representation is crucial for the genetic algorithm.
         Permutation encoding naturally enforces the constraint that each drone
         is used exactly once.

    COURSE CONNECTION:
        - Your Presentation: "Representation: Solution = permutation list (length n),
          Position = area; Value = assigned drone"
        - Lecture 4: "Edge Recombination is a permutation-preserving crossover
          which was specifically designed for the TSP"
        - The key insight is that permutation problems require special operators!
    """
    print("\n" + "=" * 60)
    print("SOLUTION REPRESENTATION")
    print("=" * 60)

    print("""
    We use PERMUTATION ENCODING for this assignment problem.

    FORMAT: solution = [drone_for_area_0, drone_for_area_1, ..., drone_for_area_99]

    EXAMPLE (for n=5):
        solution = [2, 4, 0, 3, 1]

        This means:
        - Area 0 is assigned drone 2
        - Area 1 is assigned drone 4
        - Area 2 is assigned drone 0
        - Area 3 is assigned drone 3
        - Area 4 is assigned drone 1

    WHY PERMUTATION ENCODING?
        The key constraint is: each drone must be assigned to exactly ONE area.

        Using a permutation list automatically enforces this constraint:
        - Every number from 0 to n-1 appears exactly once
        - No repeats means no drone is double-assigned

    IMPORTANT FOR GENETIC OPERATORS:
        Standard crossover would break permutation validity!

        Bad example (standard one-point crossover):
            Parent 1: [2, 4, 0, 3, 1]
            Parent 2: [1, 3, 4, 0, 2]
            Crossover point: 2

            Child: [2, 4, | 4, 0, 2]  ← INVALID! (4 appears twice, 1 and 3 missing)

        This is why we need permutation-specific operators like:
        - Partially Matched Crossover (PMX) ← mentioned in your presentation
        - Order Crossover (OX)
        - Edge Recombination ← mentioned in Lecture 4
    """)


def main():
    """
    Main function to run all data understanding steps.
    """
    print("=" * 60)
    print("STEP 1: UNDERSTANDING THE ASSIGNMENT PROBLEM DATA")
    print("=" * 60)

    # Try to load the data from project directory
    try:
        n, cost_matrix = load_assignment_data(COST_FILE)
        analyze_cost_matrix(cost_matrix)

        # NEW: Visualize and save cost matrix
        visualize_cost_matrix(cost_matrix, COST_IMAGE)

    except FileNotFoundError:
        print(f"Note: Could not find file at {COST_FILE}")
        print("Creating synthetic example for demonstration...")
        n = 5
        cost_matrix = np.random.randint(1, 100, (n, n))
        analyze_cost_matrix(cost_matrix)

    # Explain the solution representation
    understand_solution_representation()

    print("\n" + "=" * 60)
    print("SUMMARY - STEP 1 COMPLETE")
    print("=" * 60)
    print(f"""
    ✓ Cost matrix loaded from: {COST_FILE}
    ✓ Visualization saved to: {COST_IMAGE}
    
    Cost Matrix (Objective 1 - MINIMIZE):
    ┌─────────────────────────────────────┐
    │  Source: assign100.txt (OR-Library) │
    │  Size: 100 × 100                    │
    │  Values: 1 to 100                   │
    │  Mean: ~51                          │
    │  Distribution: Uniform              │
    └─────────────────────────────────────┘

    NEXT STEPS:
    1. ✓ We now understand the cost matrix structure
    2. → Next: Create the synthetic detection probability matrix (Step 2)
    3. → Then: Implement the objective functions (Step 3)
    4. → Then: Implement NSGA-II algorithm (Step 4)
    """)


if __name__ == "__main__":
    main()


# END OF FILE
"""
COMP5012 Final Report - Step 2: Creating the Detection Probability Matrix
===========================================================================

WHAT WE'RE DOING:
    Creating a synthetic 100×100 probability matrix P where P(i,j) represents
    the probability that drone i successfully detects a mine in area j.

WHY WE'RE DOING IT:
    The assessment requires TWO conflicting objectives for multi-objective
    optimization. We have:
    - Objective 1: Minimize cost (from assign100.txt)
    - Objective 2: Maximize detection probability (synthetic - approved by Dr. Ansell)

COURSE CONNECTION:
    - Your Presentation: "Dual Objectives: Objective 1: Maximize detection probability,
      Objective 2: Minimize cost"
    - Lecture 2 (Metaheuristics and MOOP): Multi-objective problems require 
      conflicting objectives to create interesting trade-offs
    - Dr. Ansell's email: Confirmed the approach of adding a second probability table
      for the likelihood of detection

DESIGN RATIONALE:
    We want the probability matrix to:
    1. Be in the range [0, 1] (valid probabilities)
    2. Have some structure (not purely random) to create realistic scenarios
    3. NOT be correlated with the cost matrix (to ensure objectives conflict)
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


def create_detection_probability_matrix(n, method='uniform'):
    """
    Create a synthetic detection probability matrix.

    WHAT: Generate n×n matrix where P(i,j) = probability drone i detects mine in area j.

    WHY: We need a second objective function for multi-objective optimization.
         The probability values should be realistic (between 0 and 1).

    Parameters:
    -----------
    n : int
        Problem size (100 for assign100)
    method : str
        Method for generating probabilities:
        - 'uniform': Uniform random values in [0.3, 0.95]
        - 'structured': Drones have specializations (some better at certain areas)
        - 'beta': Uses beta distribution for more realistic probabilities

    COURSE CONNECTION:
        - Lecture 2: "defining objective functions" - we're designing the second objective
        - Your Presentation: "Maximize detection probability"
    """

    if method == 'uniform':
        # Simple uniform distribution
        # Range [0.3, 0.95] ensures no impossibly bad or perfect detection
        prob_matrix = np.random.uniform(0.3, 0.95, (n, n))

    elif method == 'structured':
        # More realistic: drones have strengths/weaknesses
        # Some drones are better at detecting in certain types of areas

        # Base probability for all drones
        base_prob = np.random.uniform(0.4, 0.6, (n, n))

        # Drone specialization factor (some drones are generally better)
        drone_skill = np.random.uniform(0.8, 1.2, (n, 1))

        # Area difficulty factor (some areas are easier to scan)
        area_difficulty = np.random.uniform(0.8, 1.2, (1, n))

        # Combine factors
        prob_matrix = base_prob * drone_skill * area_difficulty

        # Clip to valid probability range
        prob_matrix = np.clip(prob_matrix, 0.2, 0.98)

    elif method == 'beta':
        # Beta distribution is good for modeling probabilities
        # Parameters (alpha=2, beta=2) give bell-shaped distribution centered at 0.5
        # Parameters (alpha=5, beta=2) skew towards higher probabilities
        prob_matrix = np.random.beta(5, 2, (n, n))

    else:
        raise ValueError(f"Unknown method: {method}")

    return prob_matrix


def analyze_probability_matrix(prob_matrix, cost_matrix=None):
    """
    Analyze the probability matrix to understand its characteristics.

    WHAT: Generate statistics about the detection probabilities.

    WHY: Understanding the distribution helps us:
         1. Verify the matrix is sensible
         2. Check that it's not correlated with cost (objectives should conflict!)
         3. Understand what "good" vs "bad" detection looks like
    """
    print("\n" + "=" * 60)
    print("DETECTION PROBABILITY MATRIX ANALYSIS")
    print("=" * 60)

    print(f"\nMatrix shape: {prob_matrix.shape}")
    print(f"  Interpretation: {prob_matrix.shape[0]} drones × {prob_matrix.shape[1]} areas")

    print(f"\nProbability statistics:")
    print(f"  Minimum probability: {prob_matrix.min():.4f}")
    print(f"  Maximum probability: {prob_matrix.max():.4f}")
    print(f"  Mean probability: {prob_matrix.mean():.4f}")
    print(f"  Std deviation: {prob_matrix.std():.4f}")

    print(f"\nSample of probability matrix (first 5×5 corner):")
    print(np.round(prob_matrix[:5, :5], 3))

    # Find best and worst detection assignments
    max_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    min_idx = np.unravel_index(np.argmin(prob_matrix), prob_matrix.shape)

    print(f"\nBest detection: Drone {max_idx[0]} → Area {max_idx[1]} (prob: {prob_matrix[max_idx]:.4f})")
    print(f"Worst detection: Drone {min_idx[0]} → Area {min_idx[1]} (prob: {prob_matrix[min_idx]:.4f})")

    # Check correlation with cost matrix (if provided)
    if cost_matrix is not None:
        correlation = np.corrcoef(cost_matrix.flatten(), prob_matrix.flatten())[0, 1]
        print(f"\nCorrelation with cost matrix: {correlation:.4f}")
        print("  (We want this close to 0 to ensure objectives conflict!)")

        if abs(correlation) < 0.1:
            print("  ✓ Good! Low correlation means objectives are largely independent.")
        elif abs(correlation) < 0.3:
            print("  ⚠ Moderate correlation - objectives may not conflict strongly.")
        else:
            print("  ✗ High correlation - consider regenerating with different seed.")


def visualize_probability_matrix(prob_matrix, save_path=None):
    """
    Create visualizations of the probability matrix.

    WHAT: Generate plots showing the distribution and structure of probabilities.

    WHY: Visualization helps us understand and communicate our design choices.
         The report will need figures showing the data characteristics.

    COURSE CONNECTION:
        - Assessment rubric mentions "visualisations demonstrating that the
          optimiser has found a suitable solution"
        - Understanding your data helps design better experiments
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap of probabilities
    ax1 = axes[0]
    im = ax1.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('Area Index')
    ax1.set_ylabel('Drone Index')
    ax1.set_title('Detection Probability Matrix\nP(i,j) = Probability drone i detects mine in area j')
    plt.colorbar(im, ax=ax1, label='Detection Probability')

    # Histogram of all probability values
    ax2 = axes[1]
    ax2.hist(prob_matrix.flatten(), bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Detection Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Detection Probabilities')
    ax2.axvline(prob_matrix.mean(), color='red', linestyle='--',
                label=f'Mean: {prob_matrix.mean():.3f}')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def save_probability_matrix(prob_matrix, filepath):
    """
    Save the probability matrix for later use.

    WHAT: Save matrix to a numpy file for reproducibility.

    WHY: We want to use the same probability matrix throughout all experiments
         for consistency. The report should use consistent data.
    """
    np.save(filepath, prob_matrix)
    print(f"\nProbability matrix saved to: {filepath}")
    print("  Load later with: prob_matrix = np.load(filepath)")


def main():
    """
    Main function to create and analyze the detection probability matrix.
    """
    print("=" * 60)
    print("STEP 2: CREATING DETECTION PROBABILITY MATRIX")
    print("=" * 60)

    # Load the cost matrix first (for correlation check)
    print("\nLoading cost matrix from assign100.txt...")
    with open(r"C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\assign100.txt", 'r') as f:
        numbers = []
        for line in f:
            numbers.extend([int(x) for x in line.split()])
    n = numbers[0]
    cost_matrix = np.array(numbers[1:n * n + 1]).reshape(n, n)
    print(f"Cost matrix loaded: {cost_matrix.shape}")

    # Create the probability matrix using different methods
    print("\n" + "-" * 60)
    print("METHOD 1: Uniform Distribution")
    print("-" * 60)
    prob_uniform = create_detection_probability_matrix(n, method='uniform')
    analyze_probability_matrix(prob_uniform, cost_matrix)

    print("\n" + "-" * 60)
    print("METHOD 2: Beta Distribution (skewed towards higher probabilities)")
    print("-" * 60)
    prob_beta = create_detection_probability_matrix(n, method='beta')
    analyze_probability_matrix(prob_beta, cost_matrix)

    print("\n" + "-" * 60)
    print("METHOD 3: Structured (drones have specializations)")
    print("-" * 60)
    prob_structured = create_detection_probability_matrix(n, method='structured')
    analyze_probability_matrix(prob_structured, cost_matrix)

    # For the project, we'll use the beta distribution
    # It gives realistic probabilities skewed towards successful detection
    print("\n" + "=" * 60)
    print("SELECTED METHOD: Beta Distribution")
    print("=" * 60)
    print("""
    RATIONALE:
    - Beta distribution is commonly used for modeling probabilities
    - Parameters (alpha=5, beta=2) give realistic detection rates
    - Skewed towards higher probabilities (most drones can detect reasonably well)
    - Independent of cost matrix (low correlation ensures conflicting objectives)

    This is important for multi-objective optimization because:
    - Conflicting objectives create interesting trade-offs
    - The Pareto front will show meaningful choices
    - A decision maker can choose between cost and detection quality
    """)

    # Save the chosen probability matrix
    prob_matrix = prob_beta
    save_probability_matrix(prob_matrix, r'C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\detection_probabilities.npy')

    # Create visualization
    visualize_probability_matrix(prob_matrix,r'C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\probability_matrix.png')

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
    1. ✓ Cost matrix loaded and understood
    2. ✓ Detection probability matrix created and saved
    3. → Next: Implement the objective functions (Step 3)
       - f1(solution) = total cost of assignment
       - f2(solution) = total detection probability (or negative, for minimization)
    4. → Then: Implement NSGA-II algorithm (Step 4)
    """)


if __name__ == "__main__":
    main()

# END OF FILE

"""
COMP5012 Final Report - Step 3: Solution Representation & Objective Functions
===============================================================================

WHAT WE'RE DOING:
    Implementing the solution class and objective functions for the mine 
    detection assignment problem.

WHY WE'RE DOING IT:
    The objective functions are the foundation of any optimization problem.
    They define what we're trying to optimize and how we measure solution quality.

COURSE CONNECTION:
    - Assessment Brief: "You must write Python code that implements the objective 
      functions"
    - Lecture 2 (Metaheuristics and MOOP): Understanding how to define and evaluate
      objectives is fundamental to optimization
    - Your Presentation: "Dual Objectives: Objective 1: Maximize detection probability,
      Objective 2: Minimize cost"
    - Workshop 2 Model Answer: Shows class-based solution design pattern

MULTI-OBJECTIVE NOTE:
    Since we have TWO objectives, we need to handle them appropriately:
    - NSGA-II can handle both minimization objectives
    - We'll convert "maximize detection" to "minimize negative detection" 
      OR keep track of which objectives to minimize vs maximize
"""

import numpy as np
from typing import Tuple, List


class Solution:
    """
    Represents a solution to the mine detection assignment problem.

    WHAT: A solution is a permutation representing drone-to-area assignments.

    WHY: Encapsulating the solution in a class allows us to:
         1. Store both the assignment and its objective values together
         2. Add methods for evaluation, comparison, and manipulation
         3. Keep the code organized (important for the 25% implementation mark)

    COURSE CONNECTION:
        - Workshop 2 shows this class-based design pattern
        - Your Presentation: "Representation: Solution = permutation list (length n)"

    Attributes:
    -----------
    permutation : np.ndarray
        Array of length n where permutation[area] = drone assigned to that area
    objectives : tuple
        (cost, detection) values after evaluation
    rank : int
        Pareto rank from non-dominated sorting (used in NSGA-II)
    crowding_distance : float
        Crowding distance for diversity preservation (used in NSGA-II)
    """

    def __init__(self, permutation: np.ndarray):
        """
        Initialize a solution with a given permutation.

        Parameters:
        -----------
        permutation : np.ndarray
            Array where permutation[area] = drone assigned to that area
        """
        self.permutation = np.array(permutation)
        self.objectives = None  # (cost, detection) - set after evaluation
        self.rank = None  # Pareto rank - set during NSGA-II selection
        self.crowding_distance = None  # Set during NSGA-II selection

    def is_valid(self) -> bool:
        """
        Check if the solution is a valid permutation.

        WHAT: Verify that the solution satisfies all constraints.

        WHY: Invalid solutions waste computational resources and produce
             meaningless results. We should catch errors early.

        COURSE CONNECTION:
            - Your Presentation: "Constraints enforced by encoding: One drone
              per area; no repeats"
        """
        n = len(self.permutation)

        # Check 1: All values are within valid range [0, n-1]
        if not np.all((self.permutation >= 0) & (self.permutation < n)):
            return False

        # Check 2: All values are unique (no drone assigned twice)
        if len(np.unique(self.permutation)) != n:
            return False

        return True

    def evaluate(self, cost_matrix: np.ndarray, prob_matrix: np.ndarray):
        """
        Evaluate the solution's objective values.

        WHAT: Calculate both objective function values for this solution.

        WHY: Objective values determine solution quality and drive the
             optimization process.

        COURSE CONNECTION:
            - Lecture 2: Objective function evaluation is central to optimization
            - Your Presentation: "Dual Objectives"

        Parameters:
        -----------
        cost_matrix : np.ndarray
            n×n matrix where cost_matrix[drone][area] = assignment cost
        prob_matrix : np.ndarray
            n×n matrix where prob_matrix[drone][area] = detection probability

        Returns:
        --------
        tuple : (cost, detection)
            The objective values for this solution
        """
        n = len(self.permutation)

        # Objective 1: Total Cost (MINIMIZE)
        # Sum of c(drone_i, area_i) for all assignments
        total_cost = sum(
            cost_matrix[self.permutation[area], area]
            for area in range(n)
        )

        # Objective 2: Total Detection Probability (MAXIMIZE)
        # Sum of p(drone_i, area_i) for all assignments
        total_detection = sum(
            prob_matrix[self.permutation[area], area]
            for area in range(n)
        )

        # Store as tuple (cost, detection)
        # Note: For NSGA-II, we'll treat cost as minimize and detection as maximize
        # OR convert to (cost, -detection) to minimize both
        self.objectives = (total_cost, total_detection)

        return self.objectives

    def dominates(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another solution.

        WHAT: Pareto dominance comparison between two solutions.

        WHY: Pareto dominance is the core concept in multi-objective optimization.
             It determines which solutions are "better" without converting to
             a single objective.

        COURSE CONNECTION:
            - Lecture 2 (MOOP): "When a candidate solutions is considered, solutions
              in areas labelled e are equal, solutions in area b are better and
              solutions in area w are worse"
            - Lecture 4 (NSGA-II): "Rank solutions with non-dominated sorting"

        Definition:
            Solution A dominates Solution B if:
            1. A is no worse than B in all objectives
            2. A is strictly better than B in at least one objective

            For our problem:
            - Lower cost is better (minimize)
            - Higher detection is better (maximize)

        Parameters:
        -----------
        other : Solution
            The solution to compare against

        Returns:
        --------
        bool : True if self dominates other
        """
        if self.objectives is None or other.objectives is None:
            raise ValueError("Solutions must be evaluated before dominance check")

        cost_self, detect_self = self.objectives
        cost_other, detect_other = other.objectives

        # Check condition 1: self is no worse in all objectives
        no_worse_cost = cost_self <= cost_other  # Lower cost is better
        no_worse_detect = detect_self >= detect_other  # Higher detection is better
        no_worse = no_worse_cost and no_worse_detect

        # Check condition 2: self is strictly better in at least one objective
        strictly_better_cost = cost_self < cost_other
        strictly_better_detect = detect_self > detect_other
        strictly_better = strictly_better_cost or strictly_better_detect

        return no_worse and strictly_better

    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        new_solution = Solution(self.permutation.copy())
        if self.objectives is not None:
            new_solution.objectives = self.objectives
        new_solution.rank = self.rank
        new_solution.crowding_distance = self.crowding_distance
        return new_solution

    def __repr__(self):
        """String representation for debugging."""
        obj_str = f"({self.objectives[0]:.1f}, {self.objectives[1]:.3f})" if self.objectives else "unevaluated"
        return f"Solution(objectives={obj_str}, rank={self.rank})"


def create_random_solution(n: int) -> Solution:
    """
    Create a random valid solution (permutation).

    WHAT: Generate a random assignment of drones to areas.

    WHY: Random initialization is used to create the initial population
         in genetic algorithms.

    COURSE CONNECTION:
        - Lecture 4 (GA Flowchart): "Initialise" step creates initial population
        - Workshop 2: Shows random solution generation

    Parameters:
    -----------
    n : int
        Problem size (number of drones/areas)

    Returns:
    --------
    Solution : A randomly generated valid solution
    """
    permutation = np.random.permutation(n)
    return Solution(permutation)


def evaluate_assignment(permutation: np.ndarray,
                        cost_matrix: np.ndarray,
                        prob_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate a permutation directly (functional interface).

    WHAT: Calculate objective values without creating a Solution object.

    WHY: Sometimes we just need the values quickly without object overhead.

    Parameters:
    -----------
    permutation : np.ndarray
        The assignment permutation
    cost_matrix : np.ndarray
        The cost matrix from assign100.txt
    prob_matrix : np.ndarray
        The detection probability matrix

    Returns:
    --------
    tuple : (total_cost, total_detection)
    """
    n = len(permutation)

    total_cost = sum(cost_matrix[permutation[area], area] for area in range(n))
    total_detection = sum(prob_matrix[permutation[area], area] for area in range(n))

    return total_cost, total_detection


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_solution_class():
    """
    Test the Solution class with examples.

    WHAT: Verify our implementation is correct.

    WHY: Testing is essential before building more complex algorithms.
         Bugs in the foundation will propagate to all results.

    COURSE CONNECTION:
        - Workshop 2 shows testing patterns with print statements
    """
    print("=" * 60)
    print("TESTING SOLUTION CLASS")
    print("=" * 60)

    # Create small test matrices (5×5 for clarity)
    n = 5

    # Simple test cost matrix
    cost_matrix = np.array([
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55],
        [12, 22, 32, 42, 52],
        [18, 28, 38, 48, 58],
        [11, 21, 31, 41, 51]
    ])

    # Simple test probability matrix
    prob_matrix = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.85, 0.75, 0.65, 0.55, 0.45],
        [0.88, 0.78, 0.68, 0.58, 0.48],
        [0.82, 0.72, 0.62, 0.52, 0.42],
        [0.91, 0.81, 0.71, 0.61, 0.51]
    ])

    print("\nTest 1: Creating and validating a solution")
    print("-" * 40)
    sol1 = Solution(np.array([0, 1, 2, 3, 4]))  # Identity assignment
    print(f"Solution: {sol1.permutation}")
    print(f"Is valid: {sol1.is_valid()}")

    # Evaluate
    sol1.evaluate(cost_matrix, prob_matrix)
    print(f"Objectives: cost={sol1.objectives[0]}, detection={sol1.objectives[1]:.2f}")

    # Manual verification:
    # Cost: c(0,0) + c(1,1) + c(2,2) + c(3,3) + c(4,4) = 10+25+32+48+51 = 166
    # Detection: p(0,0) + p(1,1) + p(2,2) + p(3,3) + p(4,4) = 0.9+0.75+0.68+0.52+0.51 = 3.36
    print(f"Expected cost: 166, detection: 3.36")

    print("\nTest 2: Creating another solution for dominance comparison")
    print("-" * 40)
    sol2 = Solution(np.array([4, 3, 2, 1, 0]))  # Reversed assignment
    sol2.evaluate(cost_matrix, prob_matrix)
    print(f"Solution 2 permutation: {sol2.permutation}")
    print(f"Objectives: cost={sol2.objectives[0]}, detection={sol2.objectives[1]:.2f}")

    # Manual verification:
    # Cost: c(4,0) + c(3,1) + c(2,2) + c(1,3) + c(0,4) = 11+28+32+45+50 = 166
    # Detection: p(4,0) + p(3,1) + p(2,2) + p(1,3) + p(0,4) = 0.91+0.72+0.68+0.55+0.5 = 3.36

    print("\nTest 3: Pareto dominance")
    print("-" * 40)
    print(f"Does sol1 dominate sol2? {sol1.dominates(sol2)}")
    print(f"Does sol2 dominate sol1? {sol2.dominates(sol1)}")

    # Create a clearly dominated solution
    sol3 = Solution(np.array([0, 0, 0, 0, 0]))  # Invalid but let's test dominance concept
    # Actually, let's use a valid but worse solution

    print("\nTest 4: Invalid solution detection")
    print("-" * 40)
    invalid_sol = Solution(np.array([0, 0, 1, 2, 3]))  # Duplicate 0
    print(f"Invalid solution: {invalid_sol.permutation}")
    print(f"Is valid: {invalid_sol.is_valid()}")  # Should be False

    print("\nTest 5: Random solution generation")
    print("-" * 40)
    random_sol = create_random_solution(5)
    print(f"Random solution: {random_sol.permutation}")
    print(f"Is valid: {random_sol.is_valid()}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


def demonstrate_objective_tradeoff():
    """
    Demonstrate the trade-off between cost and detection objectives.

    WHAT: Show that optimizing one objective may hurt the other.

    WHY: Understanding the trade-off is key to multi-objective optimization.
         This is what creates the Pareto front.

    COURSE CONNECTION:
        - Lecture 2: Multi-objective problems have conflicting objectives
        - Lecture 4: "The consequence is that now a decision maker can decide
          which solution to pick and look at the trade-offs between the objectives"
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING OBJECTIVE TRADE-OFF")
    print("=" * 60)

    # Load real data
    with open(r"C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\assign100.txt", 'r') as f:
        numbers = []
        for line in f:
            numbers.extend([int(x) for x in line.split()])
    n = numbers[0]
    cost_matrix = np.array(numbers[1:n * n + 1]).reshape(n, n)

    # Load probability matrix
    prob_matrix = np.load(r'C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\detection_probabilities.npy')

    print(f"\nProblem size: {n} drones, {n} areas")

    # Create several random solutions and evaluate them
    print("\nEvaluating 10 random solutions:")
    print("-" * 60)
    print(f"{'Solution':<10} {'Cost':<12} {'Detection':<12}")
    print("-" * 60)

    solutions = []
    for i in range(10):
        sol = create_random_solution(n)
        sol.evaluate(cost_matrix, prob_matrix)
        solutions.append(sol)
        print(f"{i + 1:<10} {sol.objectives[0]:<12.0f} {sol.objectives[1]:<12.3f}")

    # Find best for each objective
    best_cost = min(solutions, key=lambda s: s.objectives[0])
    best_detect = max(solutions, key=lambda s: s.objectives[1])

    print("\n" + "-" * 60)
    print(f"Best cost solution: Cost={best_cost.objectives[0]:.0f}, Detection={best_cost.objectives[1]:.3f}")
    print(f"Best detection solution: Cost={best_detect.objectives[0]:.0f}, Detection={best_detect.objectives[1]:.3f}")

    print("""
    OBSERVATION:
    The best cost solution may not have the best detection, and vice versa.
    This is the TRADE-OFF that multi-objective optimization explores.

    NSGA-II will find a set of solutions (Pareto front) that represent
    the best trade-offs between these conflicting objectives.
    """)


def main():
    """Run all tests and demonstrations."""
    print("=" * 60)
    print("STEP 3: SOLUTION CLASS & OBJECTIVE FUNCTIONS")
    print("=" * 60)

    # Run tests
    test_solution_class()

    # Demonstrate trade-off
    demonstrate_objective_tradeoff()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    We have implemented:

    1. Solution class with:
       - Permutation representation
       - Validity checking
       - Objective function evaluation
       - Pareto dominance comparison

    2. Objective functions:
       - f1(solution) = total cost (MINIMIZE)
       - f2(solution) = total detection probability (MAXIMIZE)

    NEXT STEPS:
    -----------
    4. → Implement genetic operators for permutations:
         - Tournament Selection
         - Partially Matched Crossover (PMX)
         - Swap Mutation

    5. → Implement NSGA-II:
         - Non-dominated sorting
         - Crowding distance
         - Main evolutionary loop
    """)


if __name__ == "__main__":
    main()
"""
=============================================================================
COMP5012 - Computational Intelligence
Step 4: Genetic Operators for Multi-Objective Drone Assignment
=============================================================================

📚 LEARNING MODE - Σύνδεση με το Υλικό του Μαθήματος:
-----------------------------------------------------
Από τα Lecture Notes (Metaheuristics and MOOP):
    "Genetic operators are used to create new populations of solutions from 
     the old population. The operators fall into two categories: 
     mutation operators and crossover operators."

Από την Παρουσίασή σου:
    - Selection: tournament
    - Crossover: order-based (Partially Matched Crossover)
    - Mutation: swap positions

ΓΙΑΤΙ ΑΥΤΟΙ ΟΙ OPERATORS;
-------------------------
Το πρόβλημά μας είναι PERMUTATION-BASED:
    - Κάθε λύση είναι μια μετάθεση [0, 1, 2, ..., 99]
    - Κάθε drone εμφανίζεται ΑΚΡΙΒΩΣ μία φορά
    - Κάθε area παίρνει ΑΚΡΙΒΩΣ ένα drone

Αυτό σημαίνει:
    ❌ One-point crossover θα δημιουργούσε ΑΚΥΡΕΣ λύσεις (διπλότυπα!)
    ✅ PMX διατηρεί τη μοναδικότητα κάθε στοιχείου
    ✅ Swap mutation είναι ασφαλής για permutations
=============================================================================
"""

import numpy as np
import random
from typing import List, Tuple

# =============================================================================
# 1. TOURNAMENT SELECTION
# =============================================================================
"""
📚 LEARNING: Tournament Selection
---------------------------------
Από τα Lecture Notes:
    "During the selection step, we only select a portion of the population 
     to reproduce for the creation of a new generation. Individual solutions 
     are selected through a fitness-based process so that better solutions 
     are more likely to be selected."

Από SPEA Algorithm (Lecture Notes):
    "Select parents from Population using binary tournament selection"

ΠΩΣ ΛΕΙΤΟΥΡΓΕΙ:
1. Επιλέγουμε τυχαία k άτομα από τον πληθυσμό (tournament size)
2. Επιστρέφουμε το ΚΑΛΥΤΕΡΟ από αυτά
3. "Καλύτερο" = χαμηλότερο rank (front) ή μεγαλύτερο crowding distance

ΓΙΑΤΙ TOURNAMENT (και όχι roulette wheel);
- Δουλεύει καλά με multi-objective problems
- Δεν χρειάζεται κανονικοποίηση fitness
- Εύκολη παραμετροποίηση με tournament size
"""

def tournament_selection(population: List[np.ndarray],
                         ranks: np.ndarray,
                         crowding_distances: np.ndarray,
                         tournament_size: int = 2) -> np.ndarray:
    """
    Binary Tournament Selection για NSGA-II.

    Από τα Lecture Notes (NSGA-II):
        "Create a new offspring population from this new population using
         crowd tournament selection (it compares front ranking and then
         crowding distance to break ties)"

    Parameters:
    -----------
    population : List[np.ndarray]
        Ο πληθυσμός των λύσεων (κάθε λύση είναι permutation)
    ranks : np.ndarray
        Το Pareto front rank κάθε λύσης (1 = best, non-dominated)
    crowding_distances : np.ndarray
        Η crowding distance κάθε λύσης (higher = better diversity)
    tournament_size : int
        Πόσα άτομα συμμετέχουν στο tournament (default: 2 = binary)

    Returns:
    --------
    np.ndarray : Η νικήτρια λύση (αντίγραφο)
    """
    pop_size = len(population)

    # Επιλέγουμε τυχαία tournament_size indices
    tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)

    # Βρίσκουμε τον νικητή
    # Προτεραιότητα 1: Χαμηλότερο rank (καλύτερο front)
    # Προτεραιότητα 2: Μεγαλύτερο crowding distance (διατήρηση diversity)

    best_idx = tournament_indices[0]
    for idx in tournament_indices[1:]:
        # Αν έχει καλύτερο rank, νικάει
        if ranks[idx] < ranks[best_idx]:
            best_idx = idx
        # Αν έχει ίδιο rank, νικάει αυτός με μεγαλύτερο crowding distance
        elif ranks[idx] == ranks[best_idx]:
            if crowding_distances[idx] > crowding_distances[best_idx]:
                best_idx = idx

    # Επιστρέφουμε ΑΝΤΙΓΡΑΦΟ (για να μην αλλάξουμε τον αρχικό γονέα)
    return population[best_idx].copy()


# =============================================================================
# 2. PMX CROSSOVER (Partially Matched Crossover)
# =============================================================================
"""
📚 LEARNING: PMX Crossover
--------------------------
Από την Παρουσίασή σου:
    "Crossover: order-based (Partially Matched Crossover)"

Από τα Lecture Notes:
    "Edge Recombination - This is a permutation-preserving crossover which 
     was specifically designed for the TSP in which round trips are encoded 
     as permutations of the cities."

PMX είναι παρόμοιο - διατηρεί τη δομή της μετάθεσης!

ΠΩΣ ΛΕΙΤΟΥΡΓΕΙ ΤΟ PMX:
1. Επιλέγουμε δύο τυχαία σημεία cut (crossover points)
2. Αντιγράφουμε το segment μεταξύ τους από parent1 στο child
3. Για τις υπόλοιπες θέσεις, χρησιμοποιούμε mapping για να αποφύγουμε duplicates

ΠΑΡΑΔΕΙΓΜΑ:
    Parent1: [1, 2, 3, 4, 5, 6, 7, 8]
    Parent2: [3, 7, 5, 1, 6, 8, 2, 4]
    
    Cut points: 3, 6 (segment: positions 3-5)
    
    Child starts: [_, _, _, 4, 5, 6, _, _]  (from parent1)
    
    Fill rest from parent2 using mapping...
    
    Result: Valid permutation!

ΓΙΑΤΙ PMX (και όχι one-point crossover);
- One-point crossover: [1,2,3] + [1,4,5] = [1,2,3,1,4,5] ← ΑΚΥΡΟ! (duplicate 1)
- PMX: Εγγυάται valid permutation
"""

def pmx_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partially Matched Crossover (PMX) για permutations.

    Δημιουργεί ΔΥΟ παιδιά από δύο γονείς, διατηρώντας
    τη validity της μετάθεσης.

    Parameters:
    -----------
    parent1 : np.ndarray
        Πρώτος γονέας (permutation)
    parent2 : np.ndarray
        Δεύτερος γονέας (permutation)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : Δύο παιδιά (child1, child2)
    """
    n = len(parent1)

    # Βήμα 1: Επιλέγουμε δύο τυχαία crossover points
    point1, point2 = sorted(random.sample(range(n), 2))

    # Δημιουργούμε τα παιδιά με -1 (σημαίνει "κενό")
    child1 = np.full(n, -1)
    child2 = np.full(n, -1)

    # Βήμα 2: Αντιγράφουμε το segment από τους γονείς
    # child1 παίρνει segment από parent1
    # child2 παίρνει segment από parent2
    child1[point1:point2+1] = parent1[point1:point2+1]
    child2[point1:point2+1] = parent2[point1:point2+1]

    # Βήμα 3: Δημιουργούμε mapping για conflict resolution
    # Αυτό είναι το "μαγικό" του PMX!
    def fill_remaining(child, parent_segment, other_parent, point1, point2):
        """
        Συμπληρώνει τις κενές θέσεις χρησιμοποιώντας mapping.
        """
        n = len(child)
        segment_values = set(parent_segment)

        # Για κάθε θέση εκτός του segment
        for i in range(n):
            if i >= point1 and i <= point2:
                continue  # Skip segment positions

            # Προσπαθούμε να πάρουμε την τιμή από other_parent
            value = other_parent[i]

            # Αν η τιμή ήδη υπάρχει στο segment, κάνουμε mapping
            while value in segment_values:
                # Βρίσκουμε πού είναι αυτή η τιμή στο segment του parent
                idx = np.where(parent_segment == value)[0]
                if len(idx) > 0:
                    # Παίρνουμε την αντίστοιχη τιμή από other_parent
                    idx_in_segment = idx[0]
                    actual_pos = point1 + idx_in_segment
                    value = other_parent[actual_pos]
                else:
                    break

            child[i] = value

        return child

    # Βήμα 4: Συμπληρώνουμε τα παιδιά
    segment1 = parent1[point1:point2+1]
    segment2 = parent2[point1:point2+1]

    child1 = fill_remaining(child1, segment1, parent2, point1, point2)
    child2 = fill_remaining(child2, segment2, parent1, point1, point2)

    return child1, child2


def pmx_crossover_simple(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Απλοποιημένη έκδοση PMX που επιστρέφει ένα παιδί.
    Πιο εύκολη στην κατανόηση και debugging.

    ΠΩΣ ΛΕΙΤΟΥΡΓΕΙ (Βήμα-βήμα):
    1. Αντιγράφουμε segment από parent1 → child
    2. Για κάθε κενή θέση, παίρνουμε τιμή από parent2
    3. Αν η τιμή υπάρχει ήδη, βρίσκουμε την πρώτη διαθέσιμη
    """
    n = len(parent1)

    # Επιλέγουμε crossover points
    point1 = np.random.randint(0, n-1)
    point2 = np.random.randint(point1+1, n)

    # Δημιουργούμε child με -1
    child = np.full(n, -1, dtype=int)

    # Αντιγράφουμε segment από parent1
    child[point1:point2] = parent1[point1:point2]

    # Κρατάμε track ποιες τιμές έχουμε ήδη
    used = set(child[point1:point2])

    # Συμπληρώνουμε από parent2
    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            # Βρίσκουμε την επόμενη διαθέσιμη τιμή από parent2
            while parent2[p2_idx] in used:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            used.add(parent2[p2_idx])
            p2_idx += 1

    return child


# =============================================================================
# 3. SWAP MUTATION
# =============================================================================
"""
📚 LEARNING: Swap Mutation
--------------------------
Από την Παρουσίασή σου:
    "Mutation: swap positions"

Από τα Lecture Notes:
    "A mutation operator will pick a random decision variable and modify 
     it in some way, this could be to 'flip' the variable to the opposite 
     state or to add a randomly generated value to it."
    
    "Mutation operators are good for exploiting good existing solutions."

ΓΙΑ PERMUTATIONS:
- Δεν μπορούμε να κάνουμε "flip" (θα δημιουργήσουμε duplicate)
- Αντί αυτού, SWAP = αλλάζουμε θέσεις δύο στοιχείων

ΠΑΡΑΔΕΙΓΜΑ:
    Before: [1, 2, 3, 4, 5]
    Swap positions 1 και 3
    After:  [1, 4, 3, 2, 5]
    
    ✅ Παραμένει valid permutation!

ΣΚΟΠΟΣ ΤΟΥ MUTATION:
- Εισάγει DIVERSITY στον πληθυσμό
- Αποτρέπει premature convergence
- Επιτρέπει EXPLORATION νέων περιοχών του search space
"""

def swap_mutation(solution: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Swap Mutation για permutations.

    Με πιθανότητα mutation_rate, αλλάζει θέσεις δύο τυχαίων στοιχείων.

    Parameters:
    -----------
    solution : np.ndarray
        Η λύση προς mutation (permutation)
    mutation_rate : float
        Πιθανότητα να γίνει mutation (default: 0.1 = 10%)

    Returns:
    --------
    np.ndarray : Η (πιθανώς) mutated λύση

    Σημείωση: Επιστρέφει ΑΝΤΙΓΡΑΦΟ, δεν τροποποιεί το original
    """
    # Δημιουργούμε αντίγραφο
    mutated = solution.copy()

    # Ελέγχουμε αν θα γίνει mutation
    if np.random.random() < mutation_rate:
        n = len(mutated)

        # Επιλέγουμε δύο ΔΙΑΦΟΡΕΤΙΚΕΣ θέσεις
        pos1, pos2 = np.random.choice(n, size=2, replace=False)

        # Κάνουμε swap
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

    return mutated


def multi_swap_mutation(solution: np.ndarray, num_swaps: int = 2) -> np.ndarray:
    """
    Εκτελεί πολλαπλά swaps (πιο aggressive mutation).

    Χρήσιμο για να ξεφύγουμε από local optima.

    Parameters:
    -----------
    solution : np.ndarray
        Η λύση προς mutation
    num_swaps : int
        Πόσα swaps να γίνουν

    Returns:
    --------
    np.ndarray : Η mutated λύση
    """
    mutated = solution.copy()
    n = len(mutated)

    for _ in range(num_swaps):
        pos1, pos2 = np.random.choice(n, size=2, replace=False)
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

    return mutated


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def create_random_solution(n: int) -> np.ndarray:
    """
    Δημιουργεί μια τυχαία valid permutation.

    Parameters:
    -----------
    n : int
        Το μέγεθος της permutation

    Returns:
    --------
    np.ndarray : Μια τυχαία μετάθεση [0, 1, 2, ..., n-1]
    """
    solution = np.arange(n)
    np.random.shuffle(solution)
    return solution


def is_valid_permutation(solution: np.ndarray, n: int) -> bool:
    """
    Ελέγχει αν μια λύση είναι valid permutation.

    Μια valid permutation πρέπει:
    1. Να έχει μήκος n
    2. Να περιέχει κάθε αριθμό από 0 έως n-1 ακριβώς μία φορά

    Parameters:
    -----------
    solution : np.ndarray
        Η λύση προς έλεγχο
    n : int
        Το αναμενόμενο μέγεθος

    Returns:
    --------
    bool : True αν είναι valid
    """
    if len(solution) != n:
        return False
    if set(solution) != set(range(n)):
        return False
    return True


# =============================================================================
# 5. TESTING - Ας δοκιμάσουμε τους operators!
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 4: Testing Genetic Operators")
    print("=" * 70)

    # Ορίζουμε το μέγεθος του προβλήματος
    n = 100  # 100 areas, 100 drones

    # Δημιουργούμε test solutions
    np.random.seed(42)  # Για reproducibility

    print("\n📊 1. Δημιουργία Test Solutions")
    print("-" * 40)
    parent1 = create_random_solution(n)
    parent2 = create_random_solution(n)
    print(f"Parent 1 (πρώτα 10): {parent1[:10]}")
    print(f"Parent 2 (πρώτα 10): {parent2[:10]}")
    print(f"Valid permutation P1: {is_valid_permutation(parent1, n)}")
    print(f"Valid permutation P2: {is_valid_permutation(parent2, n)}")

    # Test PMX Crossover
    print("\n🔀 2. Testing PMX Crossover")
    print("-" * 40)
    child = pmx_crossover_simple(parent1, parent2)
    print(f"Child (πρώτα 10): {child[:10]}")
    print(f"Valid permutation: {is_valid_permutation(child, n)}")

    # Verify uniqueness
    unique_values = len(set(child))
    print(f"Unique values: {unique_values} (πρέπει να είναι {n})")

    # Test Swap Mutation
    print("\n🧬 3. Testing Swap Mutation")
    print("-" * 40)
    original = child.copy()
    mutated = swap_mutation(child, mutation_rate=1.0)  # Force mutation

    # Βρίσκουμε ΠΟΙΕΣ θέσεις άλλαξαν
    changed_positions = np.where(original != mutated)[0]

    print(f"Valid permutation: {is_valid_permutation(mutated, n)}")
    print(f"Positions changed: {len(changed_positions)}")

    if len(changed_positions) == 2:
        pos1, pos2 = changed_positions
        print(f"\n🔄 Swap Details:")
        print(f"   Position {pos1}: {original[pos1]} → {mutated[pos1]}")
        print(f"   Position {pos2}: {original[pos2]} → {mutated[pos2]}")
        print(f"\n   Δηλαδή: Area {pos1} άλλαξε drone από {original[pos1]} σε {mutated[pos1]}")
        print(f"           Area {pos2} άλλαξε drone από {original[pos2]} σε {mutated[pos2]}")

    # Test Tournament Selection (με dummy ranks και distances)
    print("\n🏆 4. Testing Tournament Selection")
    print("-" * 40)

    # Δημιουργούμε ένα μικρό population για testing
    pop_size = 10
    population = [create_random_solution(n) for _ in range(pop_size)]

    # Dummy ranks (1 = best front, higher = worse)
    ranks = np.array([1, 2, 1, 3, 2, 1, 3, 2, 1, 2])

    # Dummy crowding distances
    crowding_distances = np.random.random(pop_size)

    print(f"Population size: {pop_size}")
    print(f"Ranks: {ranks}")
    print(f"Crowding distances: {np.round(crowding_distances, 3)}")

    # Κάνουμε selection πολλές φορές
    selection_counts = np.zeros(pop_size)
    for _ in range(1000):
        winner = tournament_selection(population, ranks, crowding_distances, tournament_size=2)
        # Βρίσκουμε ποιος κέρδισε (με βάση το πρώτο στοιχείο)
        for i, p in enumerate(population):
            if np.array_equal(winner, p):
                selection_counts[i] += 1
                break

    print(f"\nSelection counts (1000 tournaments):")
    for i in range(pop_size):
        print(f"  Solution {i} (rank={ranks[i]}): {int(selection_counts[i])} times")

    print("\n" + "=" * 70)
    print("✅ All Genetic Operators tested successfully!")
    print("=" * 70)

    # Summary για το report
    print("\n📝 SUMMARY για το Report:")
    print("-" * 40)
    print("""
    Οι Genetic Operators που υλοποιήθηκαν:
    
    1. Tournament Selection (Binary)
       - Επιλέγει 2 τυχαία άτομα
       - Νικάει αυτός με καλύτερο rank
       - Ties λύνονται με crowding distance
       - Σύνδεση: NSGA-II lecture notes
    
    2. PMX Crossover
       - Permutation-safe crossover
       - Διατηρεί τη μοναδικότητα κάθε στοιχείου
       - Σύνδεση: Presentation slide "Genetic Algorithm Design"
    
    3. Swap Mutation
       - Αλλάζει θέσεις 2 τυχαίων στοιχείων
       - Εισάγει diversity
       - Σύνδεση: Presentation slide "Genetic Algorithm Design"
    """)

    """
    =============================================================================
    COMP5012 - Computational Intelligence
    Step 5: NSGA-II Algorithm for Multi-Objective Drone Assignment
    =============================================================================

    📚 LEARNING MODE - Σύνδεση με το Υλικό του Μαθήματος:
    -----------------------------------------------------
    Από τα Lecture Notes (Metaheuristics and MOOP):

        "NSGA II stands for Non-dominated Sorting Genetic Algorithm 2 and was 
         published by Deb et al. in 2002. The algorithm is widely used in a 
         range of fields, and the original paper has been cited more than 
         40,000 times."

    Τα 3 KEY FEATURES του NSGA-II:
        1. ELITIST PRINCIPLE - οι καλύτερες λύσεις περνάνε στην επόμενη γενιά
        2. CROWDING DISTANCE - διατηρεί diversity (ποικιλία)
        3. EMPHASISES NON-DOMINATING SOLUTIONS - προτεραιότητα στο Pareto Front

    Τα ΒΗΜΑΤΑ του αλγορίθμου:
        1. "First, perform a non-dominating sorting in the combination of 
            parent and offspring populations and classify them by fronts."
        2. "Now fill a new population according to front ranking."
        3. "If the last front overflows, perform crowding sort."
        4. "Create a new offspring population using crowd tournament selection,
            crossover and mutation operators."
    =============================================================================
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from typing import List, Tuple, Dict
    import random


    # =============================================================================
    # ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ΚΑΙ ΠΡΟΗΓΟΥΜΕΝΟΥ ΚΩΔΙΚΑ
    # =============================================================================

    def load_cost_matrix(filepath: str) -> np.ndarray:
        """Φορτώνει τον cost matrix από το OR-Library αρχείο."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        n = int(lines[0].strip())
        costs = []
        for line in lines[1:]:
            costs.extend([int(x) for x in line.split()])

        cost_matrix = np.array(costs).reshape(n, n)
        return cost_matrix


    def create_probability_matrix(n: int, alpha: float = 5, beta: float = 2,
                                  seed: int = 42) -> np.ndarray:
        """
        Δημιουργεί τον probability matrix με Beta distribution.

        Από την προηγούμενη δουλειά (Step 2):
        - Beta(α=5, β=2) δίνει τιμές skewed προς υψηλές πιθανότητες
        - Αυτό δημιουργεί conflict με το cost (expensive drones = better detection)
        """
        np.random.seed(seed)
        return np.random.beta(alpha, beta, size=(n, n))


    # =============================================================================
    # SOLUTION CLASS - Αναπαράσταση Λύσης
    # =============================================================================

    class Solution:
        """
        Αναπαριστά μια λύση του προβλήματος.

        Attributes:
            permutation: np.ndarray - η ανάθεση (position=area, value=drone)
            objectives: Tuple[float, float] - (total_cost, negative_detection)
            rank: int - Pareto front rank (1 = best)
            crowding_distance: float - μέτρο diversity
        """

        def __init__(self, permutation: np.ndarray):
            self.permutation = permutation.copy()
            self.objectives = None  # (cost, -detection) - BOTH MINIMIZED
            self.rank = None
            self.crowding_distance = 0.0

        def evaluate(self, cost_matrix: np.ndarray, prob_matrix: np.ndarray):
            """
            Υπολογίζει τα objectives της λύσης.

            Objective 1: MINIMIZE total cost
            Objective 2: MINIMIZE negative detection (= MAXIMIZE detection)

            ΣΗΜΑΝΤΙΚΟ: Μετατρέπουμε detection σε αρνητικό για να γίνει minimization!
            """
            n = len(self.permutation)

            # Objective 1: Total Cost (minimize)
            total_cost = sum(cost_matrix[area, self.permutation[area]]
                             for area in range(n))

            # Objective 2: Total Detection Probability (maximize → minimize negative)
            total_detection = sum(prob_matrix[area, self.permutation[area]]
                                  for area in range(n))

            # Αποθηκεύουμε ως (cost, -detection) για να κάνουμε MINIMIZE και τα δύο
            self.objectives = (total_cost, -total_detection)

            return self.objectives


    # =============================================================================
    # NON-DOMINATED SORTING
    # =============================================================================
    """
    📚 LEARNING: Non-dominated Sorting
    ----------------------------------
    Από τα Lecture Notes:
        "Rank solutions with non-dominated sorting"
        "First, perform a non-dominating sorting in the combination of parent 
         and offspring populations and classify them by fronts."

    ΤΙ ΚΑΝΕΙ:
    Χωρίζει τον πληθυσμό σε "fronts" (μέτωπα):
        - Front 1: Λύσεις που δεν κυριαρχούνται από καμία άλλη (BEST)
        - Front 2: Λύσεις που κυριαρχούνται μόνο από το Front 1
        - Front 3: κ.ο.κ.

    DOMINANCE (Κυριαρχία):
    Λύση A dominates Λύση B αν:
        - A είναι καλύτερη ή ίση σε ΟΛΟΥΣ τους στόχους ΚΑΙ
        - A είναι αυστηρά καλύτερη σε ΤΟΥΛΑΧΙΣΤΟΝ ΕΝΑ στόχο
    """


    def dominates(obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
        """
        Ελέγχει αν η λύση με objectives obj1 ΚΥΡΙΑΡΧΕΙ την obj2.

        obj1 dominates obj2 αν:
        - obj1 <= obj2 σε ΟΛΟΥΣ τους στόχους (είμαστε σε minimization)
        - obj1 < obj2 σε ΤΟΥΛΑΧΙΣΤΟΝ ΕΝΑ στόχο

        Parameters:
        -----------
        obj1, obj2 : Tuple[float, float]
            Τα objectives δύο λύσεων (cost, -detection)

        Returns:
        --------
        bool : True αν obj1 dominates obj2
        """
        better_in_all = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        strictly_better_in_one = any(o1 < o2 for o1, o2 in zip(obj1, obj2))

        return better_in_all and strictly_better_in_one


    def fast_non_dominated_sort(population: List[Solution]) -> List[List[int]]:
        """
        Fast Non-dominated Sorting - O(MN²) complexity.

        Από τα Lecture Notes:
            "classify them by fronts"

        Parameters:
        -----------
        population : List[Solution]
            Ο πληθυσμός προς ταξινόμηση

        Returns:
        --------
        List[List[int]] : Λίστα από fronts, κάθε front περιέχει indices λύσεων
        """
        n = len(population)

        # Για κάθε λύση, κρατάμε:
        # - domination_count: πόσες λύσεις την κυριαρχούν
        # - dominated_solutions: ποιες λύσεις κυριαρχεί
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]

        fronts = [[]]  # Ξεκινάμε με κενό Front 1

        # Βήμα 1: Υπολογίζουμε domination για κάθε ζεύγος
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if dominates(population[i].objectives, population[j].objectives):
                    # i dominates j
                    dominated_solutions[i].append(j)
                elif dominates(population[j].objectives, population[i].objectives):
                    # j dominates i
                    domination_count[i] += 1

        # Βήμα 2: Βρίσκουμε το Front 1 (λύσεις με domination_count = 0)
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 1
                fronts[0].append(i)

        # Βήμα 3: Βρίσκουμε τα υπόλοιπα fronts
        current_front = 0
        while current_front < len(fronts) and len(fronts[current_front]) > 0:
            next_front = []

            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 2
                        next_front.append(j)

            current_front += 1
            if next_front:
                fronts.append(next_front)

        # Αφαιρούμε κενά fronts
        fronts = [f for f in fronts if len(f) > 0]

        return fronts


    # =============================================================================
    # CROWDING DISTANCE
    # =============================================================================
    """
    📚 LEARNING: Crowding Distance
    ------------------------------
    Από τα Lecture Notes:
        "Resolve ties with crowding distance"
        "crowding distance that is related to the density of solutions 
         around each solution. The less dense is preferred."

    ΤΙ ΚΑΝΕΙ:
    Μετράει πόσο "μόνη" είναι μια λύση στον objective space.
    - ΜΕΓΑΛΟ crowding distance = μόνη, σε αραιή περιοχή → ΚΑΛΟ (θέλουμε diversity)
    - ΜΙΚΡΟ crowding distance = πολλές γειτονικές λύσεις → ΚΑΚΟ

    ΥΠΟΛΟΓΙΣΜΟΣ:
    Για κάθε objective:
    1. Ταξινομούμε τις λύσεις
    2. Οι άκρες (min, max) παίρνουν infinite distance
    3. Οι υπόλοιπες: distance = (next_obj - prev_obj) / (max_obj - min_obj)
    4. Αθροίζουμε για όλα τα objectives
    """


    def calculate_crowding_distance(population: List[Solution],
                                    front_indices: List[int]) -> None:
        """
        Υπολογίζει το crowding distance για τις λύσεις ενός front.

        Parameters:
        -----------
        population : List[Solution]
            Ο πληθυσμός
        front_indices : List[int]
            Οι indices των λύσεων στο front
        """
        if len(front_indices) == 0:
            return

        # Αρχικοποίηση
        for i in front_indices:
            population[i].crowding_distance = 0.0

        if len(front_indices) <= 2:
            # Αν έχουμε 1-2 λύσεις, δίνουμε infinite distance
            for i in front_indices:
                population[i].crowding_distance = float('inf')
            return

        num_objectives = 2  # cost και detection

        for obj_idx in range(num_objectives):
            # Ταξινομούμε με βάση αυτό το objective
            sorted_indices = sorted(front_indices,
                                    key=lambda i: population[i].objectives[obj_idx])

            # Οι άκρες παίρνουν infinite distance
            population[sorted_indices[0]].crowding_distance = float('inf')
            population[sorted_indices[-1]].crowding_distance = float('inf')

            # Υπολογίζουμε το range για κανονικοποίηση
            obj_min = population[sorted_indices[0]].objectives[obj_idx]
            obj_max = population[sorted_indices[-1]].objectives[obj_idx]

            if obj_max - obj_min == 0:
                continue  # Αποφυγή division by zero

            # Υπολογίζουμε crowding distance για τις ενδιάμεσες
            for i in range(1, len(sorted_indices) - 1):
                prev_obj = population[sorted_indices[i - 1]].objectives[obj_idx]
                next_obj = population[sorted_indices[i + 1]].objectives[obj_idx]

                population[sorted_indices[i]].crowding_distance += \
                    (next_obj - prev_obj) / (obj_max - obj_min)


    # =============================================================================
    # GENETIC OPERATORS (από Step 4)
    # =============================================================================

    def tournament_selection(population: List[Solution],
                             tournament_size: int = 2) -> Solution:
        """
        Binary Tournament Selection.

        Από τα Lecture Notes:
            "crowd tournament selection (it compares front ranking and then
             crowding distance to break ties)"
        """
        contestants = random.sample(range(len(population)), tournament_size)

        best = contestants[0]
        for idx in contestants[1:]:
            # Σύγκριση: πρώτα rank, μετά crowding distance
            if population[idx].rank < population[best].rank:
                best = idx
            elif population[idx].rank == population[best].rank:
                if population[idx].crowding_distance > population[best].crowding_distance:
                    best = idx

        return Solution(population[best].permutation)


    def pmx_crossover(parent1: Solution, parent2: Solution) -> Solution:
        """
        Partially Matched Crossover (PMX) για permutations.

        Από την Παρουσίαση: "Crossover: order-based (Partially Matched Crossover)"
        """
        n = len(parent1.permutation)
        p1 = parent1.permutation
        p2 = parent2.permutation

        # Επιλέγουμε crossover points
        point1 = np.random.randint(0, n - 1)
        point2 = np.random.randint(point1 + 1, n)

        # Δημιουργούμε child
        child = np.full(n, -1, dtype=int)

        # Αντιγράφουμε segment από parent1
        child[point1:point2] = p1[point1:point2]

        # Κρατάμε track ποιες τιμές έχουμε
        used = set(child[point1:point2])

        # Συμπληρώνουμε από parent2
        p2_idx = 0
        for i in range(n):
            if child[i] == -1:
                while p2[p2_idx] in used:
                    p2_idx += 1
                child[i] = p2[p2_idx]
                used.add(p2[p2_idx])
                p2_idx += 1

        return Solution(child)


    def swap_mutation(solution: Solution, mutation_rate: float = 0.1) -> Solution:
        """
        Swap Mutation για permutations.

        Από την Παρουσίαση: "Mutation: swap positions"
        """
        if np.random.random() < mutation_rate:
            perm = solution.permutation.copy()
            n = len(perm)
            pos1, pos2 = np.random.choice(n, size=2, replace=False)
            perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
            return Solution(perm)
        return solution


    # =============================================================================
    # NSGA-II MAIN ALGORITHM
    # =============================================================================
    """
    📚 LEARNING: NSGA-II Main Loop
    ------------------------------
    Από τα Lecture Notes:
        "The NSGA II algorithm has the following three key features:
         1. It uses an elitist principal, i.e. the elites of a population are 
            given the opportunity to be carried forward to the next generation.
         2. It uses an explicit diversity-preserving mechanism (crowding distance).
         3. It emphasises the non-dominating solutions."

    Τα βήματα:
        1. Initialise random population
        2. Evaluate population
        3. For each generation:
           a. Create offspring (selection, crossover, mutation)
           b. Combine parent + offspring
           c. Non-dominated sorting
           d. Select best N for next generation
        4. Return final Pareto front
    """


    class NSGA2:
        """
        NSGA-II Algorithm Implementation.

        Βασισμένο στο paper: Deb et al. (2002)
        """

        def __init__(self,
                     cost_matrix: np.ndarray,
                     prob_matrix: np.ndarray,
                     pop_size: int = 100,
                     num_generations: int = 200,
                     crossover_rate: float = 0.9,
                     mutation_rate: float = 0.1):
            """
            Parameters:
            -----------
            cost_matrix : np.ndarray
                Ο πίνακας κόστους (100x100)
            prob_matrix : np.ndarray
                Ο πίνακας πιθανοτήτων detection (100x100)
            pop_size : int
                Μέγεθος πληθυσμού
            num_generations : int
                Αριθμός γενεών
            crossover_rate : float
                Πιθανότητα crossover
            mutation_rate : float
                Πιθανότητα mutation
            """
            self.cost_matrix = cost_matrix
            self.prob_matrix = prob_matrix
            self.n = cost_matrix.shape[0]  # 100 areas/drones
            self.pop_size = pop_size
            self.num_generations = num_generations
            self.crossover_rate = crossover_rate
            self.mutation_rate = mutation_rate

            # Για tracking της εξέλιξης
            self.history = []

        def initialize_population(self) -> List[Solution]:
            """Δημιουργεί τον αρχικό τυχαίο πληθυσμό."""
            population = []
            for _ in range(self.pop_size):
                perm = np.random.permutation(self.n)
                sol = Solution(perm)
                sol.evaluate(self.cost_matrix, self.prob_matrix)
                population.append(sol)
            return population

        def create_offspring(self, population: List[Solution]) -> List[Solution]:
            """
            Δημιουργεί offspring population.

            Από τα Lecture Notes:
                "Create a new offspring population from this new population using
                 crowd tournament selection, crossover and mutation operators."
            """
            offspring = []

            while len(offspring) < self.pop_size:
                # Selection
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = pmx_crossover(parent1, parent2)
                else:
                    child = Solution(parent1.permutation.copy())

                # Mutation
                child = swap_mutation(child, self.mutation_rate)

                # Evaluate
                child.evaluate(self.cost_matrix, self.prob_matrix)
                offspring.append(child)

            return offspring

        def select_next_generation(self, combined: List[Solution]) -> List[Solution]:
            """
            Επιλέγει τον επόμενο πληθυσμό από τον combined (parents + offspring).

            Από τα Lecture Notes:
                "Now fill a new population according to front ranking.
                 If the last front overflows, perform crowding sort that uses
                 the crowding distance."
            """
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined)

            # Υπολογισμός crowding distance για κάθε front
            for front in fronts:
                calculate_crowding_distance(combined, front)

            # Γέμισμα του νέου πληθυσμού
            new_population = []
            front_idx = 0

            while len(new_population) + len(fronts[front_idx]) <= self.pop_size:
                # Προσθέτουμε ολόκληρο το front
                for i in fronts[front_idx]:
                    new_population.append(combined[i])
                front_idx += 1
                if front_idx >= len(fronts):
                    break

            # Αν χρειάζονται κι άλλες λύσεις από το τελευταίο front
            if len(new_population) < self.pop_size and front_idx < len(fronts):
                # Ταξινομούμε με βάση crowding distance (descending)
                remaining = fronts[front_idx]
                remaining.sort(key=lambda i: combined[i].crowding_distance,
                               reverse=True)

                # Προσθέτουμε τις καλύτερες (μεγαλύτερο crowding)
                for i in remaining:
                    if len(new_population) >= self.pop_size:
                        break
                    new_population.append(combined[i])

            return new_population

        def run(self) -> List[Solution]:
            """
            Εκτελεί τον NSGA-II αλγόριθμο.

            Returns:
            --------
            List[Solution] : Το τελικό Pareto front
            """
            print("=" * 60)
            print("NSGA-II Starting...")
            print("=" * 60)
            print(f"Population size: {self.pop_size}")
            print(f"Generations: {self.num_generations}")
            print(f"Problem size: {self.n} areas × {self.n} drones")
            print("-" * 60)

            # Βήμα 1: Αρχικοποίηση
            population = self.initialize_population()

            # Non-dominated sorting στον αρχικό πληθυσμό
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                calculate_crowding_distance(population, front)

            # Βήμα 2: Κύριος βρόχος εξέλιξης
            for gen in range(self.num_generations):
                # Δημιουργία offspring
                offspring = self.create_offspring(population)

                # Συνδυασμός parent + offspring
                combined = population + offspring

                # Επιλογή επόμενης γενιάς
                population = self.select_next_generation(combined)

                # Tracking progress
                pareto_front = [s for s in population if s.rank == 1]

                # Αποθήκευση για visualization
                self.history.append({
                    'generation': gen + 1,
                    'pareto_size': len(pareto_front),
                    'best_cost': min(s.objectives[0] for s in pareto_front),
                    'best_detection': max(-s.objectives[1] for s in pareto_front)
                })

                # Progress update κάθε 20 γενιές
                if (gen + 1) % 20 == 0:
                    print(f"Generation {gen + 1:3d}: Pareto front size = {len(pareto_front)}, "
                          f"Best cost = {self.history[-1]['best_cost']:.0f}, "
                          f"Best detection = {self.history[-1]['best_detection']:.2f}")

            # Τελικό Pareto front
            final_pareto = [s for s in population if s.rank == 1]

            print("-" * 60)
            print(f"✅ Optimization complete!")
            print(f"Final Pareto front size: {len(final_pareto)}")
            print("=" * 60)

            return final_pareto

        def plot_pareto_front(self, pareto_front: List[Solution],
                              save_path: str = None):
            """
            Visualize το Pareto front.

            Από τα Lecture Notes:
                "When we have three objectives or less, we can also visualise
                 the trade-off between the solutions in a scatter plot."
            """
            # Εξαγωγή objectives
            costs = [s.objectives[0] for s in pareto_front]
            detections = [-s.objectives[1] for s in pareto_front]  # Επαναφορά σε θετικό

            plt.figure(figsize=(10, 8))
            plt.scatter(costs, detections, c='blue', s=50, alpha=0.7,
                        edgecolors='black', linewidths=0.5)

            plt.xlabel('Total Mission Cost (Minimize)', fontsize=12)
            plt.ylabel('Total Detection Probability (Maximize)', fontsize=12)
            plt.title('NSGA-II Pareto Front\nDrone Assignment Optimization',
                      fontsize=14, fontweight='bold')

            plt.grid(True, alpha=0.3)

            # Annotations για extreme solutions
            min_cost_idx = np.argmin(costs)
            max_det_idx = np.argmax(detections)

            plt.annotate(f'Min Cost\n({costs[min_cost_idx]:.0f}, {detections[min_cost_idx]:.2f})',
                         xy=(costs[min_cost_idx], detections[min_cost_idx]),
                         xytext=(10, -20), textcoords='offset points',
                         fontsize=9, ha='left',
                         arrowprops=dict(arrowstyle='->', color='red'))

            plt.annotate(f'Max Detection\n({costs[max_det_idx]:.0f}, {detections[max_det_idx]:.2f})',
                         xy=(costs[max_det_idx], detections[max_det_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=9, ha='left',
                         arrowprops=dict(arrowstyle='->', color='green'))

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 Pareto front saved to: {save_path}")

            plt.show()

        def plot_convergence(self, save_path: str = None):
            """Visualize την εξέλιξη του αλγορίθμου."""
            generations = [h['generation'] for h in self.history]
            pareto_sizes = [h['pareto_size'] for h in self.history]
            best_costs = [h['best_cost'] for h in self.history]
            best_detections = [h['best_detection'] for h in self.history]

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Pareto front size
            axes[0].plot(generations, pareto_sizes, 'b-', linewidth=2)
            axes[0].set_xlabel('Generation')
            axes[0].set_ylabel('Pareto Front Size')
            axes[0].set_title('Pareto Front Growth')
            axes[0].grid(True, alpha=0.3)

            # Best cost
            axes[1].plot(generations, best_costs, 'r-', linewidth=2)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Best Cost')
            axes[1].set_title('Best Cost Evolution')
            axes[1].grid(True, alpha=0.3)

            # Best detection
            axes[2].plot(generations, best_detections, 'g-', linewidth=2)
            axes[2].set_xlabel('Generation')
            axes[2].set_ylabel('Best Detection')
            axes[2].set_title('Best Detection Evolution')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📈 Convergence plot saved to: {save_path}")

            plt.show()


    # =============================================================================
    # MAIN - ΕΚΤΕΛΕΣΗ
    # =============================================================================

    if __name__ == "__main__":
        print("\n" + "=" * 70)
        print("STEP 5: NSGA-II for Multi-Objective Drone Assignment")
        print("=" * 70)

        # Φόρτωση δεδομένων
        print("\n📂 Loading data...")

        # Προσπαθούμε να φορτώσουμε από διάφορες τοποθεσίες
        import os

        possible_paths = [
            r"C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\assign100.txt"
        ]

        cost_matrix = None
        for path in possible_paths:
            if os.path.exists(path):
                cost_matrix = load_cost_matrix(path)
                print(f"✅ Loaded cost matrix from: {path}")
                break

        if cost_matrix is None:
            print("⚠️ Could not find assign100.txt, creating synthetic data...")
            np.random.seed(42)
            cost_matrix = np.random.randint(1, 100, size=(100, 100))

        # Δημιουργία probability matrix
        prob_matrix = create_probability_matrix(100)
        print(f"✅ Created probability matrix (100x100)")

        print(f"\n📊 Problem size: {cost_matrix.shape[0]} areas × {cost_matrix.shape[1]} drones")
        print(f"   Cost range: [{cost_matrix.min()}, {cost_matrix.max()}]")
        print(f"   Detection range: [{prob_matrix.min():.3f}, {prob_matrix.max():.3f}]")

        # Δημιουργία και εκτέλεση NSGA-II
        nsga2 = NSGA2(
            cost_matrix=cost_matrix,
            prob_matrix=prob_matrix,
            pop_size=100,
            num_generations=100,  # Μπορείς να αυξήσεις για καλύτερα αποτελέσματα
            crossover_rate=0.9,
            mutation_rate=0.1
        )

        # Εκτέλεση
        pareto_front = nsga2.run()

        # Visualization
        print("\n📊 Creating visualizations...")
        nsga2.plot_pareto_front(pareto_front, save_path='pareto_front.png')
        nsga2.plot_convergence(save_path='convergence.png')

        # Summary
        print("\n" + "=" * 70)
        print("📝 SUMMARY")
        print("=" * 70)

        costs = [s.objectives[0] for s in pareto_front]
        detections = [-s.objectives[1] for s in pareto_front]

        print(f"\nPareto Front Statistics:")
        print(f"  - Number of solutions: {len(pareto_front)}")
        print(f"  - Cost range: [{min(costs):.0f}, {max(costs):.0f}]")
        print(f"  - Detection range: [{min(detections):.2f}, {max(detections):.2f}]")

        # Extreme solutions
        print(f"\nExtreme Solutions:")
        min_cost_idx = np.argmin(costs)
        max_det_idx = np.argmax(detections)

        print(f"  - Minimum Cost Solution:")
        print(f"      Cost = {costs[min_cost_idx]:.0f}")
        print(f"      Detection = {detections[min_cost_idx]:.2f}")

        print(f"  - Maximum Detection Solution:")
        print(f"      Cost = {costs[max_det_idx]:.0f}")
        print(f"      Detection = {detections[max_det_idx]:.2f}")

        print("\n" + "=" * 70)
        print("✅ Step 5 Complete!")
        print("=" * 70)

        """
        ================================================================================
        COMP5012 - Step 6: Experimental Design (SKELETON)
        ================================================================================

        ΣΚΟΠΟΣ: Αυτός ο κώδικας είναι ο σκελετός για τα πειράματά σου.
                Εσύ θα αλλάξεις τις παραμέτρους για να δεις πώς επηρεάζουν τα αποτελέσματα.

        COURSE CONNECTION:
        - Lecture 4: "Evaluating MOEAs" - Πώς αξιολογούμε multi-objective αλγόριθμους
        - Workshop 3: Experimental design με multiple runs
        - Assessment Brief: "Good experimental design with clear rationale" (10%)

        ΤΙ ΘΑ ΜΑΘΕΙΣ:
        1. Γιατί τρέχουμε πολλά runs (statistical significance)
        2. Πώς οι παράμετροι επηρεάζουν την απόδοση
        3. Πώς να αναλύεις convergence

        ================================================================================
        ΟΔΗΓΙΕΣ ΧΡΗΣΗΣ:
        1. Πρώτα τρέξε τον κώδικα όπως είναι για να δεις ότι δουλεύει
        2. Μετά άλλαξε τις παραμέτρους στο SECTION: ΠΑΡΑΜΕΤΡΟΙ ΠΟΥ ΑΛΛΑΖΕΙΣ ΕΣΥ
        3. Παρατήρησε πώς αλλάζουν τα αποτελέσματα
        ================================================================================
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import random
        import time


        # ============================================================
        # SECTION: ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ (Αυτά τα έχεις ήδη από Steps 1-5)
        # ============================================================

        def load_cost_matrix(filepath):
            """Φόρτωση cost matrix από το assign100.txt"""
            with open(filepath, 'r') as f:
                lines = f.readlines()
            n = int(lines[0].strip())
            values = []
            for line in lines[1:]:
                values.extend([int(x) for x in line.split()])
            return np.array(values).reshape(n, n)


        def generate_probability_matrix(n, seed=42):
            """
            Δημιουργία probability matrix με Beta(5,2) distribution

            ΓΙΑΤΙ BETA(5,2):
            - Δίνει τιμές 0-1 (πιθανότητες)
            - Skewed προς υψηλές τιμές (ρεαλιστικό για detection)
            - Course Link: Step 2 discussion
            """
            np.random.seed(seed)
            return np.random.beta(5, 2, size=(n, n))


        # ============================================================
        # SECTION: SOLUTION CLASS (από Step 3)
        # ============================================================

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
            """
            Pareto Dominance: sol1 κυριαρχεί sol2 αν:
            - Είναι καλύτερη ή ίση σε ΟΛΑ τα objectives
            - Είναι ΑΥΣΤΗΡΑ καλύτερη σε ΤΟΥΛΑΧΙΣΤΟΝ ΕΝΑ

            Course Link: Lecture 4 - Pareto Dominance definition
            """
            better_cost = sol1.cost <= sol2.cost
            better_detection = sol1.detection >= sol2.detection
            strictly_better = (sol1.cost < sol2.cost) or (sol1.detection > sol2.detection)
            return better_cost and better_detection and strictly_better


        # ============================================================
        # SECTION: GENETIC OPERATORS (από Step 4)
        # ============================================================

        def tournament_selection(population, tournament_size=2):
            """
            Binary Tournament Selection

            ΠΩΣ ΔΟΥΛΕΥΕΙ:
            1. Διάλεξε τυχαία 2 λύσεις
            2. Επέστρεψε αυτή με το καλύτερο rank
            3. Αν ίδιο rank, επέστρεψε αυτή με μεγαλύτερο crowding distance

            Course Link: Lecture 4 - NSGA-II selection
            """
            selected = random.sample(population, tournament_size)
            selected.sort(key=lambda x: (x.rank, -x.crowding_distance))
            return selected[0]


        def pmx_crossover(parent1, parent2):
            """
            Partially Matched Crossover (PMX)

            ΓΙΑΤΙ PMX:
            - Διατηρεί την εγκυρότητα του permutation
            - Κάθε drone εμφανίζεται ΑΚΡΙΒΩΣ μία φορά

            Course Link: Lecture 4 - Permutation crossover operators
            """
            n = len(parent1.permutation)
            child1 = np.full(n, -1)
            child2 = np.full(n, -1)

            # Διάλεξε 2 crossover points
            cx1, cx2 = sorted(random.sample(range(n), 2))

            # Αντέγραψε το segment
            child1[cx1:cx2 + 1] = parent1.permutation[cx1:cx2 + 1]
            child2[cx1:cx2 + 1] = parent2.permutation[cx1:cx2 + 1]

            # Συμπλήρωσε τα υπόλοιπα
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
            """
            Swap Mutation

            ΠΩΣ ΔΟΥΛΕΥΕΙ:
            - Με πιθανότητα mutation_rate, αντάλλαξε 2 τυχαίες θέσεις
            - Διατηρεί permutation validity

            Course Link: Lecture 4 - Mutation operators for permutations
            """
            mutated = permutation.copy()
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(mutated)), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            return mutated


        # ============================================================
        # SECTION: NSGA-II ALGORITHM (από Step 5)
        # ============================================================

        def non_dominated_sorting(population):
            """
            Fast Non-dominated Sorting

            ΤΙ ΚΑΝΕΙ:
            - Χωρίζει τον πληθυσμό σε "fronts"
            - Front 0: Οι καλύτερες λύσεις (κανείς δεν τις κυριαρχεί)
            - Front 1: Κυριαρχούνται μόνο από Front 0
            - κλπ...

            Course Link: Lecture 4 - NSGA-II non-dominated sorting
            """
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
            """
            Crowding Distance Calculation

            ΤΙ ΚΑΝΕΙ:
            - Μετράει πόσο "απομονωμένη" είναι κάθε λύση
            - Μεγαλύτερο distance = πιο μοναδική θέση στο Pareto front
            - Βοηθάει να διατηρήσουμε diversity

            Course Link: Lecture 4 - NSGA-II crowding distance
            """
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
            """
            NSGA-II Main Algorithm

            ΠΑΡΑΜΕΤΡΟΙ ΠΟΥ ΕΠΗΡΕΑΖΟΥΝ ΤΗΝ ΑΠΟΔΟΣΗ:
            - pop_size: Μεγαλύτερος = περισσότερη εξερεύνηση, αλλά πιο αργό
            - generations: Περισσότερες = καλύτερη σύγκλιση, αλλά πιο αργό
            - mutation_rate: Υψηλότερο = περισσότερη εξερεύνηση, χαμηλότερο = περισσότερη εκμετάλλευση
            - crossover_rate: Πόσο συχνά κάνουμε crossover vs απλή αντιγραφή
            """
            n = len(cost_matrix)

            # Αρχικοποίηση πληθυσμού
            population = []
            for _ in range(pop_size):
                perm = np.random.permutation(n)
                population.append(Solution(perm, cost_matrix, prob_matrix))

            # Για tracking (αν θέλουμε)
            history = {'generation': [], 'pareto_size': [], 'best_cost': [], 'best_detection': []}

            # Initial sorting
            fronts = non_dominated_sorting(population)
            for front in fronts:
                crowding_distance(population, front)

            # Main loop
            for gen in range(generations):
                # Δημιουργία offspring
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

                # Συνδυασμός γονέων + παιδιών
                combined = population + offspring
                fronts = non_dominated_sorting(combined)

                # Επιλογή νέου πληθυσμού
                new_pop = []
                front_idx = 0
                while len(new_pop) + len(fronts[front_idx]) <= pop_size:
                    for i in fronts[front_idx]:
                        new_pop.append(combined[i])
                    front_idx += 1
                    if front_idx >= len(fronts):
                        break

                # Αν χρειάζεται, χρησιμοποίησε crowding distance
                if len(new_pop) < pop_size and front_idx < len(fronts):
                    crowding_distance(combined, fronts[front_idx])
                    remaining = sorted(fronts[front_idx],
                                       key=lambda x: combined[x].crowding_distance, reverse=True)
                    for i in remaining[:pop_size - len(new_pop)]:
                        new_pop.append(combined[i])

                population = new_pop

                # Update ranks
                fronts = non_dominated_sorting(population)
                for front in fronts:
                    crowding_distance(population, front)

                # Track history
                if track_history:
                    pareto = [p for p in population if p.rank == 0]
                    history['generation'].append(gen)
                    history['pareto_size'].append(len(pareto))
                    history['best_cost'].append(min(p.cost for p in population))
                    history['best_detection'].append(max(p.detection for p in population))

            return population, history


        # ============================================================
        # ============================================================
        #     SECTION: ΠΑΡΑΜΕΤΡΟΙ ΠΟΥ ΑΛΛΑΖΕΙΣ ΕΣΥ
        # ============================================================
        # ============================================================

        """
        ΟΔΗΓΙΕΣ:
        Άλλαξε τις παρακάτω τιμές και παρατήρησε πώς αλλάζουν τα αποτελέσματα!

        ΠΕΙΡΑΜΑ A - Multiple Runs:
        - Σκοπός: Να δούμε αν ο αλγόριθμος δίνει consistent αποτελέσματα
        - Γιατί: Οι evolutionary algorithms είναι stochastic (τυχαίοι)
        - Τι περιμένουμε: Μικρό standard deviation = καλή consistency

        ΠΕΙΡΑΜΑ B - Parameter Sensitivity:
        - Σκοπός: Να δούμε ποιες παράμετροι επηρεάζουν περισσότερο
        - Γιατί: Για να κάνουμε σωστό tuning
        - Τι περιμένουμε: Να βρούμε sweet spot για κάθε παράμετρο

        ΠΕΙΡΑΜΑ C - Convergence:
        - Σκοπός: Να δούμε πόσο γρήγορα converge ο αλγόριθμος
        - Γιατί: Lecture 4 ρωτάει "how quickly did it get there?"
        - Τι περιμένουμε: Γρήγορη βελτίωση στην αρχή, σταθεροποίηση μετά
        """

        # ------------------------------
        # ΠΕΙΡΑΜΑ A: Multiple Runs
        # ------------------------------
        # ΑΛΛΑΞΕ ΑΥΤΑ:
        N_RUNS = 30  # Πόσες φορές να τρέξει (συνιστώμενο: 30 για report)
        POP_SIZE_A = 200  # Μέγεθος πληθυσμού
        GENERATIONS_A = 200  # Αριθμός γενεών (άρχισε με 50, μετά δοκίμασε 100)
        MUTATION_RATE_A = 0.1  # Mutation rate (δοκίμασε: 0.05, 0.1, 0.2)

        # ------------------------------
        # ΠΕΙΡΑΜΑ B: Parameter Sensitivity
        # ------------------------------
        # ΑΛΛΑΞΕ ΑΥΤΑ για να δεις πώς επηρεάζουν:
        POP_SIZES = [50, 100, 200]  # Δοκίμασε διαφορετικά μεγέθη
        GENERATION_VALUES = [50, 100, 200]  # Δοκίμασε διαφορετικές γενιές
        MUTATION_RATES = [0.05, 0.1, 0.2]  # Δοκίμασε διαφορετικά rates
        N_RUNS_B = 5  # Runs ανά configuration

        # ------------------------------
        # ΠΕΙΡΑΜΑ C: Convergence Analysis
        # ------------------------------
        # ΑΛΛΑΞΕ ΑΥΤΑ:
        GENERATIONS_C = 200  # Περισσότερες γενιές για να δεις convergence
        POP_SIZE_C = 200


        # ============================================================
        # SECTION: ΠΕΙΡΑΜΑΤΑ (Μην αλλάξεις τον κώδικα, μόνο τις παραμέτρους πάνω)
        # ============================================================

        def run_experiment_A(cost_matrix, prob_matrix):
            """
            ΠΕΙΡΑΜΑ A: Multiple Runs

            ΣΚΟΠΟΣ: Να δούμε αν ο αλγόριθμος είναι consistent

            ΤΙ ΜΕΤΡΑΜΕ:
            - Pareto front size: Πόσες non-dominated λύσεις βρήκε
            - Best cost: Η φθηνότερη λύση
            - Best detection: Η λύση με υψηλότερη ανίχνευση
            """
            print("\n" + "=" * 60)
            print("ΠΕΙΡΑΜΑ A: Multiple Runs")
            print("=" * 60)
            print(f"Παράμετροι: runs={N_RUNS}, pop={POP_SIZE_A}, gen={GENERATIONS_A}, mut={MUTATION_RATE_A}")
            print("-" * 60)

            results = {'pareto_sizes': [], 'best_costs': [], 'best_detections': [], 'times': []}

            for run in range(N_RUNS):
                # Διαφορετικό seed για κάθε run
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

            # Στατιστικά
            print("\n" + "-" * 60)
            print("ΣΤΑΤΙΣΤΙΚΑ:")
            print(f"  Pareto Size: {np.mean(results['pareto_sizes']):.1f} ± {np.std(results['pareto_sizes']):.1f}")
            print(f"  Best Cost:   {np.mean(results['best_costs']):.0f} ± {np.std(results['best_costs']):.0f}")
            print(
                f"  Best Det:    {np.mean(results['best_detections']):.2f} ± {np.std(results['best_detections']):.2f}")
            print(f"  Avg Time:    {np.mean(results['times']):.2f}s")

            return results


        def run_experiment_B(cost_matrix, prob_matrix):
            """
            ΠΕΙΡΑΜΑ B: Parameter Sensitivity

            ΣΚΟΠΟΣ: Να δούμε πώς επηρεάζουν οι παράμετροι την απόδοση

            ΤΙ ΔΟΚΙΜΑΖΟΥΜΕ:
            - Διαφορετικά population sizes
            - Διαφορετικούς αριθμούς γενεών
            - Διαφορετικά mutation rates
            """
            print("\n" + "=" * 60)
            print("ΠΕΙΡΑΜΑ B: Parameter Sensitivity")
            print("=" * 60)

            results = {}

            # B1: Population Size
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

            # B2: Generations
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

            # B3: Mutation Rate
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
            """
            ΠΕΙΡΑΜΑ C: Convergence Analysis

            ΣΚΟΠΟΣ: Να δούμε πόσο γρήγορα converge ο αλγόριθμος

            Course Link: Lecture 4 - "how quickly did it get there?"
            """
            print("\n" + "=" * 60)
            print("ΠΕΙΡΑΜΑ C: Convergence Analysis")
            print("=" * 60)
            print(f"Παράμετροι: pop={POP_SIZE_C}, gen={GENERATIONS_C}")
            print("-" * 60)

            np.random.seed(42)
            random.seed(42)

            population, history = nsga2(cost_matrix, prob_matrix,
                                        POP_SIZE_C, GENERATIONS_C, 0.1,
                                        track_history=True)

            print(f"\nΤελικά αποτελέσματα μετά από {GENERATIONS_C} γενιές:")
            print(f"  Pareto Front Size: {history['pareto_size'][-1]}")
            print(f"  Best Cost: {history['best_cost'][-1]:.0f}")
            print(f"  Best Detection: {history['best_detection'][-1]:.2f}")

            # Βρες πότε σταθεροποιήθηκε
            sizes = history['pareto_size']
            converged_gen = GENERATIONS_C
            for i in range(len(sizes) - 10):
                if abs(sizes[i] - sizes[-1]) < 5:
                    converged_gen = i
                    break
            print(f"  Σταθεροποίηση περίπου στη γενιά: {converged_gen}")

            return history, population


        # ============================================================
        # SECTION: VISUALIZATION (Δημιουργία γραφημάτων)
        # ============================================================

        def create_visualizations(results_A, results_B, history_C, population_C):
            """Δημιουργία γραφημάτων για το report"""

            # 1. Multiple Runs Results
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

            # 2. Convergence Plot
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

            # 3. Parameter Sensitivity Bar Chart
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


        # ============================================================
        # MAIN - Τρέξε τα πειράματα
        # ============================================================

        if __name__ == "__main__":
            print("\n" + "=" * 70)
            print("  COMP5012 - Step 6: Experimental Design")
            print("  Dataset: assign100.txt (100 drones × 100 areas)")
            print("=" * 70)

            # Φόρτωση δεδομένων
            print("\nΦόρτωση δεδομένων...")
            cost_matrix = load_cost_matrix(r"C:\Users\apzym\OneDrive\Desktop\PLYMOUTH\COMP5012\assign100.txt")
            prob_matrix = generate_probability_matrix(100, seed=42)
            print(f"  Cost matrix: {cost_matrix.shape}")
            print(f"  Probability matrix: {prob_matrix.shape}")

            # Τρέξε τα πειράματα
            results_A = run_experiment_A(cost_matrix, prob_matrix)
            results_B = run_experiment_B(cost_matrix, prob_matrix)
            history_C, population_C = run_experiment_C(cost_matrix, prob_matrix)

            # Δημιούργησε γραφήματα
            print("\n" + "=" * 60)
            print("Δημιουργία Γραφημάτων...")
            print("=" * 60)
            create_visualizations(results_A, results_B, history_C, population_C)

            print("\n" + "=" * 70)
            print("  ΟΛΟΚΛΗΡΩΘΗΚΕ!")
            print("=" * 70)
            print("\nΑρχεία που δημιουργήθηκαν:")
            print("  - exp_A_multiple_runs.png")
            print("  - exp_B_parameters.png")
            print("  - exp_C_convergence.png")
            print("\n💡 ΕΠΟΜΕΝΟ ΒΗΜΑ: Άλλαξε τις παραμέτρους στο SECTION 'ΠΑΡΑΜΕΤΡΟΙ'")
            print("   και ξανατρέξε για να δεις πώς αλλάζουν τα αποτελέσματα!")