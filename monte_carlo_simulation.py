"""
Monte Carlo Simulation for Robust Supply Chain Network Design

This script runs multiple demand scenarios to find the most robust
factory allocation strategy using linear programming.
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import matplotlib.pyplot as plt


def load_data():
    """Load input data from Excel files."""
    manvar_costs = pd.read_excel("data/variable costs.xlsx", index_col=0)
    freight_costs = pd.read_excel("data/freight costs.xlsx", index_col=0)
    fixed_costs = pd.read_excel("data/fixed cost.xlsx", index_col=0)
    cap = pd.read_excel("data/capacity.xlsx", index_col=0)
    demand = pd.read_excel("data/demand.xlsx", index_col=0)

    # Variable costs = freight + manufacturing
    var_cost = freight_costs / 1000 + manvar_costs

    return var_cost, fixed_costs, cap, demand


def optimization_model(fixed_costs, var_cost, demand_values, cap):
    """Build and solve the capacitated plant location model."""
    loc = ["USA", "GERMANY", "JAPAN", "BRAZIL", "INDIA"]
    size = ["LOW", "HIGH"]

    model = LpProblem("Capacitated_Plant_Location", LpMinimize)

    # Decision variables
    x = LpVariable.dicts(
        "production",
        [(i, j) for i in loc for j in loc],
        lowBound=0,
        cat="continuous",
    )
    y = LpVariable.dicts("plant", [(i, s) for s in size for i in loc], cat="Binary")

    # Objective: minimize total cost
    model += lpSum(
        [fixed_costs.loc[i, s] * y[(i, s)] * 1000 for s in size for i in loc]
    ) + lpSum([var_cost.loc[i, j] * x[(i, j)] for i in loc for j in loc])

    # Demand constraints
    for j in loc:
        model += lpSum([x[(i, j)] for i in loc]) == demand_values[j]

    # Capacity constraints
    for i in loc:
        model += lpSum([x[(i, j)] for j in loc]) <= lpSum(
            [cap.loc[i, s] * y[(i, s)] * 1000 for s in size]
        )

    model.solve()

    # Extract plant decisions
    plant_name = [(i, s) for s in size for i in loc]
    plant_bool = [y[plant_name[i]].varValue for i in range(len(plant_name))]

    return LpStatus[model.status], value(model.objective), plant_bool


def generate_demand_scenarios(base_demand, n_scenarios=50, cv=0.5):
    """Generate demand scenarios using normal distribution."""
    np.random.seed(42)

    scenarios = {}
    for market, base_value in base_demand.items():
        sigma = cv * base_value
        scenarios[market] = np.random.normal(base_value, sigma, n_scenarios)
        scenarios[market] = np.clip(scenarios[market], 0, None)

    return pd.DataFrame(scenarios)


def plot_initial_solution(df_bool, plant_names):
    """Plot initial solution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_bool.astype(int).plot.bar(ax=ax, edgecolor='black', color='tab:green',
                                  y='INITIAL', legend=False)
    plt.xlabel('Plant')
    plt.ylabel('Open/Close (Boolean)')
    plt.title('Initial Solution')
    plt.tight_layout()
    plt.savefig('initial_solution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: initial_solution.png")


def plot_demand_scenarios(df_demand, markets, n_scenarios):
    """Plot demand scenarios by market."""
    colors = ['tab:green', 'tab:red', 'black', 'tab:blue', 'tab:orange']
    fig, axes = plt.subplots(len(markets), 1, figsize=(20, 12))

    for i in range(len(markets)):
        df_demand.plot(xlim=[0, n_scenarios], x='scenario', y=markets[i],
                       ax=axes[i], grid=True, color=colors[i])
        axes[i].axhline(df_demand[markets[i]].values[0], color=colors[i], linestyle="--")

    plt.xlabel('Scenario')
    plt.ylabel('(Units)')
    plt.tight_layout()
    plt.savefig('demand_scenarios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: demand_scenarios.png")


def plot_boolean_grid(df_bool):
    """Plot boolean grid of plant configurations."""
    plt.figure(figsize=(20, 4))
    plt.pcolor(df_bool, cmap='Blues', edgecolors='k', linewidths=0.5)
    plt.xticks([i + 0.5 for i in range(df_bool.shape[1])], df_bool.columns, rotation=90, fontsize=12)
    plt.yticks([i + 0.5 for i in range(df_bool.shape[0])], df_bool.index, fontsize=12)
    plt.title('Plant Configuration Across Scenarios')
    plt.tight_layout()
    plt.savefig('boolean_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: boolean_grid.png")


def plot_demand_with_grid(df_demand, df_bool, markets, n_scenarios):
    """Plot demand scenarios with boolean grid."""
    colors = ['tab:green', 'tab:red', 'black', 'tab:blue', 'tab:orange']

    # Demand plot
    fig, axes = plt.subplots(len(markets), 1, figsize=(15, 15))
    for i in range(len(markets)):
        df_demand.plot(xlim=[1, n_scenarios], x='scenario', y=markets[i],
                       ax=axes[i], grid=True, color=colors[i])
        axes[i].axhline(df_demand[markets[i]].mean(), color=colors[i], linestyle="--")
    plt.xlabel('Scenario')
    plt.ylabel('(Units)')
    plt.tight_layout()
    plt.savefig('demand_with_mean.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: demand_with_mean.png")

    # Boolean grid with abbreviated labels
    plt.figure(figsize=(15, 5))
    plt.pcolor(df_bool, cmap='Blues', edgecolors='k', linewidths=0.5)
    plt.xticks([i + 0.5 for i in range(df_bool.shape[1])], df_bool.columns, rotation=90, fontsize=12)
    labels = [d[0:5] + '-H' * ('HIGH' in d) + '-L' * ('LOW' in d) for d in df_bool.index]
    plt.yticks([i + 0.5 for i in range(df_bool.shape[0])], labels, fontsize=12)
    plt.title('Plant Opening by Scenario')
    plt.tight_layout()
    plt.savefig('boolean_grid_compact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: boolean_grid_compact.png")


def plot_unique_combinations(df_bool):
    """Plot unique combinations grid."""
    df_unique = df_bool.T.drop_duplicates().T
    df_unique.columns = ['INITIAL'] + ['C' + str(i) for i in range(1, len(df_unique.columns))]

    plt.figure(figsize=(12, 4))
    plt.pcolor(df_unique, cmap='Blues', edgecolors='k', linewidths=0.5)
    plt.xticks([i + 0.5 for i in range(df_unique.shape[1])], df_unique.columns, rotation=90, fontsize=12)
    plt.yticks([i + 0.5 for i in range(df_unique.shape[0])], df_unique.index, fontsize=12)
    plt.title('Unique Plant Configurations')
    plt.tight_layout()
    plt.savefig('unique_combinations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: unique_combinations.png")

    return df_unique


def plot_configuration_distribution(df_bool, df_unique):
    """Plot pie chart of configuration distribution."""
    COL_NAME, COL_NUMBER = [], []
    for col1 in df_unique.columns:
        count = 0
        COL_NAME.append(col1)
        for col2 in df_bool.columns:
            if (df_bool[col2] != df_unique[col1]).sum() == 0:
                count += 1
        COL_NUMBER.append(count)

    df_comb = pd.DataFrame({'column': COL_NAME, 'count': COL_NUMBER}).set_index('column')

    fig, ax = plt.subplots(figsize=(8, 8))
    my_circle = plt.Circle((0, 0), 0.8, color='white')
    df_comb.plot.pie(ax=ax, y='count', legend=False, pctdistance=0.7,
                     autopct='%1.0f%%', labeldistance=1.05,
                     wedgeprops={'linewidth': 7, 'edgecolor': 'white'})
    plt.xlabel('')
    plt.title('Configuration Distribution Across Scenarios')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('configuration_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: configuration_distribution.png")


def run_monte_carlo(n_scenarios=50, cv=0.5):
    """Run Monte Carlo simulation."""
    print("=" * 60)
    print("MONTE CARLO SIMULATION - SUPPLY CHAIN NETWORK DESIGN")
    print("=" * 60)

    # Load data
    var_cost, fixed_costs, cap, demand = load_data()

    # Base demand
    base_demand = demand["Demand"].to_dict()
    loc = ["USA", "GERMANY", "JAPAN", "BRAZIL", "INDIA"]
    size = ["LOW", "HIGH"]
    plant_names = [f"{i}-{s}" for s in size for i in loc]

    print(f"\nBase demand: {base_demand}")
    print(f"Coefficient of Variation: {cv*100}%")
    print(f"Number of scenarios: {n_scenarios}")

    # Generate scenarios
    df_scenarios = generate_demand_scenarios(base_demand, n_scenarios, cv)

    # Add scenario column for plotting
    df_demand = df_scenarios.copy()
    df_demand.insert(0, 'scenario', range(n_scenarios))
    # Prepend initial scenario
    initial_row = pd.DataFrame({'scenario': [0], **{k: [v] for k, v in base_demand.items()}})
    df_demand = pd.concat([initial_row, df_demand], ignore_index=True)

    # Run initial scenario
    print("\nRunning initial scenario...")
    status, obj_value, plant_bool_initial = optimization_model(
        fixed_costs, var_cost, base_demand, cap
    )
    print(f"Initial solution - Status: {status}, Cost: ${obj_value:,.0f}")

    # Create initial boolean dataframe
    df_bool = pd.DataFrame({'INITIAL': plant_bool_initial}, index=plant_names)

    # Run optimization for each scenario
    results = []
    all_solutions = []

    print("\nRunning simulations...")
    for i in range(n_scenarios):
        demand_values = df_scenarios.iloc[i].to_dict()
        status, obj_value, plant_bool = optimization_model(
            fixed_costs, var_cost, demand_values, cap
        )
        results.append({"scenario": i + 1, "status": status, "cost": obj_value})
        all_solutions.append(plant_bool)
        df_bool[i + 1] = plant_bool

    # Create results DataFrame
    df_results = pd.DataFrame(results)
    df_solutions = pd.DataFrame(all_solutions, columns=plant_names)
    df_bool = df_bool.astype(int)

    # Generate visualizations
    print("\n" + "-" * 60)
    print("GENERATING VISUALIZATIONS")
    print("-" * 60)

    plot_initial_solution(df_bool, plant_names)
    plot_demand_scenarios(df_demand, loc, n_scenarios)
    plot_boolean_grid(df_bool)
    plot_demand_with_grid(df_demand, df_bool, loc, n_scenarios)
    df_unique = plot_unique_combinations(df_bool)
    plot_configuration_distribution(df_bool, df_unique)

    # Analyze solutions
    print("\n" + "-" * 60)
    print("RESULTS ANALYSIS")
    print("-" * 60)

    # Find unique solutions
    print(f"\nUnique network configurations found: {len(df_unique.columns)}")

    # Count occurrences of each solution
    solution_counts = df_solutions.apply(tuple, axis=1).value_counts()
    print("\nTop 5 most frequent configurations:")
    for i, (config, count) in enumerate(solution_counts.head().items()):
        print(f"  Configuration {i+1}: {count} occurrences ({count/n_scenarios*100:.1f}%)")
        open_plants = [plant_names[j] for j, val in enumerate(config) if val == 1]
        print(f"    Open plants: {', '.join(open_plants)}")

    # Cost statistics
    print("\n" + "-" * 60)
    print("COST STATISTICS")
    print("-" * 60)
    print(f"Mean cost: ${df_results['cost'].mean():,.0f}")
    print(f"Std deviation: ${df_results['cost'].std():,.0f}")
    print(f"Min cost: ${df_results['cost'].min():,.0f}")
    print(f"Max cost: ${df_results['cost'].max():,.0f}")

    return df_results, df_solutions


def main():
    """Main function."""
    try:
        df_results, df_solutions = run_monte_carlo(n_scenarios=50, cv=0.5)
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
    except FileNotFoundError:
        print("Data files not found in 'data/' directory.")
        print("Please ensure the Excel files are available.")


if __name__ == "__main__":
    main()
