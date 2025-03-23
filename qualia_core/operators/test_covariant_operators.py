import numpy as np
import matplotlib.pyplot as plt
import logging
from quantum.core.operators.covariant_operators import discretized_covariant_laplacian_1d_radial as discretized_covariant_laplacian_1d_radial

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the metric function
def metric_function(r):
    return 1 - (0.5 / r)  # Example metric function

# Generate a 1D radial grid
r_grid = np.linspace(0.1, 10, 100)  # Avoid singularity at r=0

# Define different input fields
fields = {
    'Sinusoidal': np.sin(r_grid),
    'Gaussian': np.exp(-0.1 * (r_grid - 5)**2),
    'Step': np.where(r_grid < 5, 1, 0)
}

# Reverting boundary conditions to valid values
boundary_conditions = ['Dirichlet', 'Neumann']  # Valid boundary conditions only

A_r = metric_function(r_grid)  # Calcular os valores da métrica

plt.figure(figsize=(15, 5 * len(boundary_conditions) * len(fields)))

for field_name, field_1d in fields.items():
    for i, boundary_condition in enumerate(boundary_conditions):
        laplacian_result = discretized_covariant_laplacian_1d_radial(field_1d, r_grid, A_r, boundary_condition=boundary_condition)

        # Log the coherence value
        coherence_value = np.mean(np.abs(laplacian_result))  # Placeholder for actual coherence calculation
        logging.info(f'Coherence for {field_name} with {boundary_condition}: {coherence_value}')

        # Visualize the results
        plt.subplot(len(fields), len(boundary_conditions) * 2, len(boundary_conditions) * 2 * list(fields.keys()).index(field_name) + 2 * i + 1)
        plt.plot(r_grid, field_1d, label='Original Field')
        plt.title(f'{field_name} Field (BC: {boundary_condition})')
        plt.xlabel('r')
        plt.ylabel('Field Value')
        plt.grid(True)
        plt.legend()

        plt.subplot(len(fields), len(boundary_conditions) * 2, len(boundary_conditions) * 2 * list(fields.keys()).index(field_name) + 2 * i + 2)
        plt.plot(r_grid, laplacian_result, label='Laplacian Result', color='orange')
        plt.title(f'Laplacian Result (BC: {boundary_condition})')
        plt.xlabel('r')
        plt.ylabel('Laplacian Value')
        plt.grid(True)
        plt.legend()

# Adding detailed analysis for the Step function
for field_name, field_1d in fields.items():
    if field_name == 'Step':  # Focus on Step function
        print(f"\n=== Detailed Analysis for {field_name} Function ===")
        for boundary_condition in boundary_conditions:
            laplacian_result = discretized_covariant_laplacian_1d_radial(
                field_1d, r_grid, A_r, boundary_condition=boundary_condition
            )
            
            # Print detailed statistics
            print(f"\nBoundary Condition: {boundary_condition}")
            print(f"  Max absolute value: {np.max(np.abs(laplacian_result)):.6e}")
            print(f"  Mean absolute value: {np.mean(np.abs(laplacian_result)):.6e}")
            
            # Print values near the step discontinuity
            step_index = len(r_grid) // 2  # Assuming step is at r=5
            print("  Values near discontinuity:")
            for i in range(max(0, step_index-2), min(len(r_grid), step_index+3)):
                print(f"    r={r_grid[i]:.2f}: {laplacian_result[i]:.6e}")

# Adicionando código para validar as Jump Conditions Covariantes
A_r_jump = 1 + 0.1 * r_grid  # Exemplo de métrica não-trivial
step_field = np.where(r_grid < 5, 1, 0)

laplacian_jump = discretized_covariant_laplacian_1d_radial(step_field, r_grid, A_r_jump, 'Jump')

plt.figure()
plt.plot(r_grid, step_field, label="Step Field")
plt.plot(r_grid, laplacian_jump, label="Laplacian Jump", linestyle="dashed")
plt.legend()
plt.show()

plt.tight_layout()
plt.show()
