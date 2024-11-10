import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import bicgstab, gmres
from typing import List, Tuple
from solve_main import ChimneyAnalysis

class ChimneyAnalysisWithTracking(ChimneyAnalysis):
    """Extended ChimneyAnalysis class with solution tracking capabilities"""
    
    def __init__(self, nx=13, ny=10, case='a', refinement_factor=1, grid_size=0.2,
                 solver_type='bicgstab', track_residuals=True, **thermal_params):
        # thermal properties
        self.T_cold = thermal_params.get('T_cold', 4.0)
        self.T_room = thermal_params.get('T_room', 24.0)
        self.T_hot = thermal_params.get('T_hot', 300.0)
        self.h_cold = thermal_params.get('h_cold', 25.0)
        self.h_room = thermal_params.get('h_room', 4.0)
        self.h_hot = thermal_params.get('h_hot', 90.0)
        self.k_inner = thermal_params.get('k_inner', 45.0)
        self.k_outer = thermal_params.get('k_outer', 15.0)
        
        # grid properties
        self.base_nx = nx
        self.base_ny = ny
        self.refinement_factor = refinement_factor
        self.nx = (nx-1) * refinement_factor + 1
        self.ny = (ny-1) * refinement_factor + 1
        self.dx = grid_size / refinement_factor
        self.case = case.lower()
        base_node_types = self.setup_base_node_types()
        self.node_types = self.create_refined_grid(base_node_types, refinement_factor)
        
        # solver properties
        self.solver_type = solver_type
        self.track_residuals = track_residuals
        self.residuals = []
        
    def solve(self) -> Tuple[np.ndarray, List[float]]:
        """Solve system with residual tracking"""
        A, b = self.create_system()
        
        if self.track_residuals:
            def callback(xk):
                residual = np.linalg.norm(A.dot(xk) - b)
                self.residuals.append(residual)
                return False
        else:
            callback = None
            
        # solver option selection
        if self.solver_type == 'bicgstab':
            T, info = bicgstab(A, b, callback=callback, maxiter=1000)
        elif self.solver_type == 'gmres':
            T, info = gmres(A, b, callback=callback, maxiter=1000)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
            
        if info != 0:
            print(f"Warning: Solver did not converge! Info: {info}")
            
        return T.reshape((self.ny, self.nx)), self.residuals
    

class ChimneySensitivityAnalysis:
    def __init__(self, base_solver):
        self.base_solver = base_solver
        self.results = {}
        
    def run_parameter_study(self, param_name, param_values):
        """runs sensitivity analysis for a given parameter"""
        results = []
        
        for value in param_values:
            # creating a new solver instance with modified parameter
            solver = ChimneyAnalysisWithTracking(
                nx=self.base_solver.base_nx,
                ny=self.base_solver.base_ny,
                case=self.base_solver.case,
                refinement_factor=self.base_solver.refinement_factor,
                grid_size=self.base_solver.dx * self.base_solver.refinement_factor
            )
            # setting the parameter being studied
            setattr(solver, param_name, value)
            T, residuals = solver.solve()
            results.append({
                'param_value': value,
                'solution': T,
                'residuals': residuals
            })
            
        self.results[param_name] = results
        return results
    
    def plot_residual_convergence(self, param_name):
        """Plot residual convergence for all parameter values"""
        plt.figure(figsize=(10, 6))
        
        for result in self.results[param_name]:
            value = result['param_value']
            residuals = result['residuals']
            iterations = range(1, len(residuals) + 1)
            
            plt.semilogy(iterations, residuals, 
                        label=f'{param_name}={value:.1f}')
        
        plt.xlabel('Iteration Count')
        plt.ylabel('Log(Residual Norm)')
        plt.title(f'Convergence History - {param_name} Sensitivity')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'residual_convergence_{param_name}_case-{self.base_solver.case}.png')
        plt.close()
    
    def plot_solution_vectors(self, param_name):
        """Plot solution vectors for all parameter values"""
        results = self.results[param_name]
        n_values = len(results)
        
        fig, axes = plt.subplots(1, n_values, 
                               figsize=(5*n_values, 5))
        if n_values == 1:
            axes = [axes]
        
        all_temps = [r['solution'] for r in results]
        vmin = min(T.min() for T in all_temps)
        vmax = max(T.max() for T in all_temps)
        
        for i, result in enumerate(results):
            value = result['param_value']
            T = result['solution']
            
            im = axes[i].imshow(T, origin='lower', 
                              vmin=vmin, vmax=vmax,
                              cmap='coolwarm')
            axes[i].set_title(f'{param_name}={value:.1f}')
            fig.colorbar(im, ax=axes[i], label='Temperature (°C)')
        
        plt.suptitle(f'Temperature Distribution - {param_name} Sensitivity')
        plt.tight_layout()
        plt.savefig(f'solution_vectors_{param_name}_case-{self.base_solver.case}.png')
        plt.close()


def run_sensitivity_analysis(case='a'):
    """given base solver, runs sensitivity analysis for different input parameters"""
    base_solver = ChimneyAnalysisWithTracking(
        nx=13, ny=10,
        case=case,
        refinement_factor=3,
        grid_size=0.2
    )
    
    sensitivity_analyzer = ChimneySensitivityAnalysis(base_solver)
    
    # running analyses for different parameters
    # Hot fluid convection coefficient
    h_hot_values = [50, 90, 150, 250]
    sensitivity_analyzer.run_parameter_study('h_hot', h_hot_values)
    sensitivity_analyzer.plot_residual_convergence('h_hot')
    sensitivity_analyzer.plot_solution_vectors('h_hot')
    
    # inner material conductivity
    k_inner_values = [25, 45, 65, 85]
    sensitivity_analyzer.run_parameter_study('k_inner', k_inner_values)
    sensitivity_analyzer.plot_residual_convergence('k_inner')
    sensitivity_analyzer.plot_solution_vectors('k_inner')
    
    # outer material conductivity
    k_outer_values = [10, 15, 20, 25]
    sensitivity_analyzer.run_parameter_study('k_outer', k_outer_values)
    sensitivity_analyzer.plot_residual_convergence('k_outer')
    sensitivity_analyzer.plot_solution_vectors('k_outer')
    return sensitivity_analyzer

def save_solution_vectors(sensitivity_a, sensitivity_b):
    """Save solution vectors with complete parameter information in 2D array format 
    for each parameter studied (h_hot, k_inner, k_outer)
    for case-A and case-B
    """
    output = []
    
    # For each parameter studied (h_hot, k_inner, k_outer)
    for param_name in sensitivity_a.results.keys():
        output.append(f"\n=== Parameter Study: {param_name} ===")
        
        # Case A results
        output.append("\nCASE A:")
        for result in sensitivity_a.results[param_name]:
            value = result['param_value']
            solution = result['solution']
            
            output.append("\nParameters:")
            output.append(f"T_cold = {sensitivity_a.base_solver.T_cold:.1f}")
            output.append(f"T_room = {sensitivity_a.base_solver.T_room:.1f}")
            output.append(f"T_hot = {sensitivity_a.base_solver.T_hot:.1f}")
            output.append(f"h_cold = {sensitivity_a.base_solver.h_cold:.1f}")
            output.append(f"h_room = {sensitivity_a.base_solver.h_room:.1f}")
            
            all_params = {
                'h_hot': sensitivity_a.base_solver.h_hot,
                'k_inner': sensitivity_a.base_solver.k_inner,
                'k_outer': sensitivity_a.base_solver.k_outer
            }
            
            for p_name, p_value in all_params.items():
                if p_name == param_name:
                    output.append(f"{p_name} = {value:.1f} (STUDIED PARAMETER)")
                else:
                    output.append(f"{p_name} = {p_value:.1f}")
            
            output.append("\nGrid parameters:")
            output.append(f"nx = {sensitivity_a.base_solver.nx}")
            output.append(f"ny = {sensitivity_a.base_solver.ny}")
            output.append(f"grid_size = {sensitivity_a.base_solver.dx * sensitivity_a.base_solver.refinement_factor:.3f}")
            output.append(f"refinement_factor = {sensitivity_a.base_solver.refinement_factor}")
            output.append(f"refined_grid_size = {sensitivity_a.base_solver.dx:.3f}")
            
            output.append("\nSolution array:")
            for row in solution:
                row_str = ", ".join(f"{x:.6f}" for x in row)
                output.append(f"[{row_str}]")
        
        # Case B results
        output.append("\nCASE B:")
        for result in sensitivity_b.results[param_name]:
            value = result['param_value']
            solution = result['solution']
            
            output.append("\nParameters:")
            output.append(f"T_cold = {sensitivity_b.base_solver.T_cold:.1f}")
            output.append(f"T_room = {sensitivity_b.base_solver.T_room:.1f}")
            output.append(f"T_hot = {sensitivity_b.base_solver.T_hot:.1f}")
            output.append(f"h_cold = {sensitivity_b.base_solver.h_cold:.1f}")
            output.append(f"h_room = {sensitivity_b.base_solver.h_room:.1f}")
            
            all_params = {
                'h_hot': sensitivity_b.base_solver.h_hot,
                'k_inner': sensitivity_b.base_solver.k_inner,
                'k_outer': sensitivity_b.base_solver.k_outer
            }
            
            for p_name, p_value in all_params.items():
                if p_name == param_name:
                    output.append(f"{p_name} = {value:.1f} (STUDIED PARAMETER)")
                else:
                    output.append(f"{p_name} = {p_value:.1f}")
            
            output.append("\nGrid parameters:")
            output.append(f"nx = {sensitivity_b.base_solver.nx}")
            output.append(f"ny = {sensitivity_b.base_solver.ny}")
            output.append(f"grid_size = {sensitivity_b.base_solver.dx * sensitivity_b.base_solver.refinement_factor:.3f}")
            output.append(f"refinement_factor = {sensitivity_b.base_solver.refinement_factor}")
            output.append(f"refined_grid_size = {sensitivity_b.base_solver.dx:.3f}")
            
            output.append("\nSolution array:")
            # format as 2D array with each row on a new line
            for row in solution:
                row_str = ", ".join(f"{x:.6f}" for x in row)
                output.append(f"[{row_str}]")
    
    with open('solution_vectors.txt', 'w') as f:
        f.write("\n".join(output))
        
    print("Solution vectors have been saved to 'solution_vectors.txt'")


if __name__ == "__main__":
    # running analysis for both cases
    print("Running Case A analysis...")
    sensitivity_a = run_sensitivity_analysis('a')
    print("\nRunning Case B analysis...")
    sensitivity_b = run_sensitivity_analysis('b')

    save_solution_vectors(sensitivity_a, sensitivity_b)
