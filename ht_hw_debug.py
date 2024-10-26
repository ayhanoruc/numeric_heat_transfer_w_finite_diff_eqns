import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple, Set
from matplotlib.colors import LinearSegmentedColormap

class NodeType(Enum):
    HOT_FLUID = -1
    EXTERNAL_CORNER = 1
    COLD_SURFACE = 2
    ROOM_SURFACE = 3
    OUTER_INTERIOR = 4
    INTERFACE_CORNER = 5
    INTERFACE_SURFACE = 6
    INNER_INTERIOR = 7
    INNER_CORNER = 8
    INNER_SURFACE = 9



class ChimneyAnalysis:
    def __init__(self, nx=13, ny=10, case='a', refinement_factor=1, grid_size=0.2):
        # Thermal properties
        self.T_cold = 4.0
        self.T_room = 24.0
        self.T_hot = 300.0
        self.h_cold = 25.0
        self.h_room = 4.0
        self.h_hot = 90.0
        self.k_inner = 45.0
        self.k_outer = 15.0
        
        # Grid properties
        self.base_nx = nx
        self.base_ny = ny
        self.refinement_factor = refinement_factor
        self.nx = (nx-1) * refinement_factor + 1
        self.ny = (ny-1) * refinement_factor + 1
        self.dx = grid_size / refinement_factor
        self.case = case.lower()
        
        # initialize grids
        base_node_types = self.setup_base_node_types()
        self.node_types = self.create_refined_grid(base_node_types, refinement_factor)
    
    def create_refined_grid(self, coarse_grid, refinement_factor=2):
        """Create a refined grid by first placing corners, then filling surfaces between them then filling the regions"""
        ny, nx = coarse_grid.shape
        refined_ny = (ny-1) * refinement_factor + 1
        refined_nx = (nx-1) * refinement_factor + 1
        refined_grid = np.zeros((refined_ny, refined_nx), dtype=int)
        
        # Step 1: Map corner points to their new positions in refined grid
        corner_mapping = {}  # Store original corner positions and their new positions
        for i in range(ny):
            for j in range(nx):
                if coarse_grid[i, j] in [NodeType.EXTERNAL_CORNER.value, 
                                    NodeType.INTERFACE_CORNER.value,
                                    NodeType.INNER_CORNER.value]:
                    # Calculate new position in refined grid
                    refined_i = i * refinement_factor
                    refined_j = j * refinement_factor
                    refined_grid[refined_i, refined_j] = coarse_grid[i, j]
                    corner_mapping[(i, j)] = (refined_i, refined_j)
        
        # Step 2: Fill surfaces between corners
        def fill_between(pos1, pos2, node_type):
            y1, x1 = pos1
            y2, x2 = pos2
            # If horizontal line
            if y1 == y2:
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    refined_grid[y1, x] = node_type
            # If vertical line
            elif x1 == x2:
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    refined_grid[y, x1] = node_type
        
        # Fill external surfaces
        # Get corners for each type
        external_corners = sorted([(i, j) for (i, j), type_val in corner_mapping.items() 
                                if coarse_grid[i, j] == NodeType.EXTERNAL_CORNER.value])
        interface_corners = sorted([(i, j) for (i, j), type_val in corner_mapping.items() 
                                if coarse_grid[i, j] == NodeType.INTERFACE_CORNER.value])
        inner_corners = sorted([(i, j) for (i, j), type_val in corner_mapping.items() 
                            if coarse_grid[i, j] == NodeType.INNER_CORNER.value])
        
        # Connect corners with surfaces
        for i in range(len(external_corners)):
            curr_corner = external_corners[i]
            for next_corner in external_corners[i+1:]:
                if curr_corner[0] == next_corner[0] or curr_corner[1] == next_corner[1]:
                    fill_between(corner_mapping[curr_corner], corner_mapping[next_corner], 
                            NodeType.COLD_SURFACE.value)
        
        for i in range(len(interface_corners)):
            curr_corner = interface_corners[i]
            for next_corner in interface_corners[i+1:]:
                if curr_corner[0] == next_corner[0] or curr_corner[1] == next_corner[1]:
                    fill_between(corner_mapping[curr_corner], corner_mapping[next_corner], 
                            NodeType.INTERFACE_SURFACE.value)
        
        for i in range(len(inner_corners)):
            curr_corner = inner_corners[i]
            for next_corner in inner_corners[i+1:]:
                if curr_corner[0] == next_corner[0] or curr_corner[1] == next_corner[1]:
                    fill_between(corner_mapping[curr_corner], corner_mapping[next_corner], 
                            NodeType.INNER_SURFACE.value)
        
        # Step 3: Fill regions
        # Get boundary coordinates for each region
        inner_y_min = min(corner_mapping[c][0] for c in inner_corners)
        inner_y_max = max(corner_mapping[c][0] for c in inner_corners)
        inner_x_min = min(corner_mapping[c][1] for c in inner_corners)
        inner_x_max = max(corner_mapping[c][1] for c in inner_corners)
        
        interface_y_min = min(corner_mapping[c][0] for c in interface_corners)
        interface_y_max = max(corner_mapping[c][0] for c in interface_corners)
        interface_x_min = min(corner_mapping[c][1] for c in interface_corners)
        interface_x_max = max(corner_mapping[c][1] for c in interface_corners)
        
        # Fill HOT_FLUID region (inside inner corners)
        for y in range(inner_y_min + 1, inner_y_max):
            for x in range(inner_x_min + 1, inner_x_max):
                if refined_grid[y, x] == 0:
                    refined_grid[y, x] = NodeType.HOT_FLUID.value
        
        # Fill INNER_INTERIOR region (between interface corners but not in HOT_FLUID)
        for y in range(interface_y_min + 1, interface_y_max):
            for x in range(interface_x_min + 1, interface_x_max):
                if refined_grid[y, x] == 0:  # Only fill zeros
                    refined_grid[y, x] = NodeType.INNER_INTERIOR.value
        
        # Fill OUTER_INTERIOR (any remaining zeros)
        for y in range(refined_ny):
            for x in range(refined_nx):
                if refined_grid[y, x] == 0:
                    refined_grid[y, x] = NodeType.OUTER_INTERIOR.value
        
        return refined_grid

    def setup_base_node_types(self):
        """Setup grid with just corner positions defined"""
        # Initialize empty grid
        base = np.zeros((self.base_ny, self.base_nx), dtype=int)
        
        external_corners = [
            (0, 0),
            (0, -1),
            (-1, 0),
            (-1, -1)
        ]
        
        interface_corners = [
            (1, 1),
            (1, -2),
            (-2, 1),
            (-2, -2)
        ]
        
        inner_corners = [
            (3, 3),
            (3, -4),
            (-4, 3),
            (-4, -4)
        ]
        
        for x, y in external_corners:
            base[y, x] = NodeType.EXTERNAL_CORNER.value
            
        for x, y in interface_corners:
            base[y, x] = NodeType.INTERFACE_CORNER.value
            
        for x, y in inner_corners:
            base[y, x] = NodeType.INNER_CORNER.value
        
        return base

    def get_node_number(self, x: int, y: int) -> int:
        """Convert x,y coordinates to node number.
        x: horizontal position (0 to nx-1)
        y: vertical position (0 to ny-1)"""
        return y * self.nx + x

    def get_convection_params(self, x: int, y: int) -> Tuple[float, float]:
        """Get convection parameters (h, T_inf) based on position and case.
        Parameters based on physical position (bottom-left origin)"""
        if self.case == 'b' and y == self.ny-1:  # Top surface for case B
            return self.h_room, self.T_room
        return self.h_cold, self.T_cold

    def get_corner_type(self, x: int, y: int) -> str:
        """Determine corner type based on position"""
        node_type = self.node_types[self.ny-1-y, x]  # Convert to array indices
        if node_type != NodeType.EXTERNAL_CORNER.value:
            return None

        is_left = (x == 0)
        is_right = (x == self.nx-1)
        is_bottom = (y == 0)
        is_top = (y == self.ny-1)
        
        if is_top and is_right: return 'tr'
        if is_top and is_left: return 'tl'
        if is_bottom and is_right: return 'br'
        if is_bottom and is_left: return 'bl'
        return None
    
    def get_node_type(self, x: int, y: int) -> int:
        """Get node type at physical coordinates (x,y)"""
        # Convert from physical coordinates to array indices
        array_y = self.ny - 1 - y
        return self.node_types[array_y, x]
    
    def get_neighbors(self, x: int, y: int) -> dict:
        """Get valid neighbor coordinates for a given position"""
        # Define all possible neighbors
        neighbors = {
            'n': (x, y+1),    # north: same x, y+1
            's': (x, y-1),    # south: same x, y-1
            'e': (x+1, y),    # east:  x+1, same y
            'w': (x-1, y)     # west:  x-1, same y
        }
        
        # Filter valid neighbors
        return {
            dir: pos for dir, pos in neighbors.items() 
            if 0 <= pos[0] < self.nx and 0 <= pos[1] < self.ny
            and self.get_node_type(pos[0], pos[1]) != NodeType.HOT_FLUID.value
        }

    def has_hot_fluid_neighbor(self, x: int, y: int) -> bool:
        """Check if any neighboring position has hot fluid"""
        neighbor_positions = [
            (x-1, y), (x+1, y),  # left, right
            (x, y-1), (x, y+1)   # below, above
        ]
        return (any(
            0 <= nx < self.nx and 0 <= ny < self.ny and 
            self.get_node_type(nx, ny) == NodeType.HOT_FLUID.value
            for nx, ny in neighbor_positions
        )or self.get_node_type(x,y) == NodeType.INNER_CORNER.value) 

    def set_node_equations(self, A, b, x, y):
        """Set equations for node at position (x,y)"""
        node_type = self.node_types[self.ny-1-y, x]  # Convert to array indices
        node = self.get_node_number(x, y)
        dx = self.dx
        
        print(f"Processing node at ({x},{y}) - Type: {NodeType(node_type).name}")
        
        if node_type == NodeType.HOT_FLUID.value:
            A[node, node] = 1
            b[node] = self.T_hot
            return

        # Get valid neighbors for this node
        neighbors = self.get_neighbors(x, y)
        print(f"  Valid neighbors: {neighbors}")

        # Handle different node types
        if node_type == NodeType.EXTERNAL_CORNER.value:
            k = self.k_outer
            h, T_inf = self.get_convection_params(x, y)
            corner_type = self.get_corner_type(x, y)
            
            # Define appropriate neighbors based on corner type
            if corner_type == 'tr':    valid_dirs = ['w', 's']
            elif corner_type == 'tl':   valid_dirs = ['e', 's']
            elif corner_type == 'br':   valid_dirs = ['w', 'n']
            elif corner_type == 'bl':   valid_dirs = ['e', 'n']
            else: raise ValueError(f"Invalid corner position at ({x},{y})")
            
            # Apply coefficients
            for dir, pos in neighbors.items():
                if dir in valid_dirs:
                    neighbor_node = self.get_node_number(*pos)
                    A[node, neighbor_node] = 1
                    
            A[node, node] = -2 * (((h * dx) / k) + 1)
            b[node] = (-2 * h * dx * T_inf )/ k # RHS term
            
        elif node_type == NodeType.INNER_CORNER.value:
            k = self.k_inner
            h = self.h_hot
            
            # Check which neighbors are hot fluid to determine corner orientation
            has_hot_fluid = {
                    dir: self.has_hot_fluid_neighbor(*pos)
                    for dir, pos in neighbors.items()
                }
            print(f"  Hot fluid neighbors: {has_hot_fluid}")
            
            # Based on equation 4.41: 2(T_{m-1,n} + T_{m,n+1}) + (T_{m+1,n} + T_{m,n-1})
            # For each possible orientation:
            corner_coeffs = {}
            if has_hot_fluid.get('n') and has_hot_fluid.get('e'):  # Hot fluid at north-east
                corner_coeffs = {'s': 2, 'w': 2, 'n': 1, 'e': 1}  # South and west get 2, others 1
            elif has_hot_fluid.get('n') and has_hot_fluid.get('w'):  # Hot fluid at north-west
                corner_coeffs = {'s': 2, 'e': 2, 'n': 1, 'w': 1}
            elif has_hot_fluid.get('s') and has_hot_fluid.get('e'):  # Hot fluid at south-east
                corner_coeffs = {'n': 2, 'w': 2, 's': 1, 'e': 1}
            elif has_hot_fluid.get('s') and has_hot_fluid.get('w'):  # Hot fluid at south-west
                corner_coeffs = {'n': 2, 'e': 2, 's': 1, 'w': 1}
            else:
                raise ValueError(f"Inner corner at ({x},{y}) not properly adjacent to hot fluid")
                
            print(f"  Corner coefficients: {corner_coeffs}")
            
            # Apply coefficients to neighbors
            for dir, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                if dir in corner_coeffs:
                    A[node, neighbor_node] = corner_coeffs[dir]
                    
            # Central node coefficient: -2(3 + h∆x/k)
            A[node, node] = -2 * (3 + ((h * dx) / k))
            
            # Right hand side term: 2(h∆x/k)T_∞
            b[node] = (-2 * h * dx * self.T_hot) / k
            
        elif node_type == NodeType.INNER_SURFACE.value:
            k = self.k_inner
            h = self.h_hot
            
            # All possible directions
            all_dirs = {'n', 's', 'e', 'w'}
            # Get missing direction (this will be where hot fluid is)
            hot_dir = next(dir for dir in all_dirs if dir not in neighbors.keys())
            print(f"  Hot fluid direction (missing from neighbors): {hot_dir}")
            
            # Opposite direction to hot fluid gets double coefficient
            double_coef_dir = {
                'n': 's',
                's': 'n',
                'e': 'w',
                'w': 'e'
            }[hot_dir]
            
            # Parallel directions (perpendicular to hot fluid direction)
            single_coef_dirs = ['n', 's'] if hot_dir in ['e', 'w'] else ['e', 'w']
            
            print(f"  Double coef dir: {double_coef_dir}, Single coef dirs: {single_coef_dirs}")
            
            # Apply coefficients to existing neighbors
            for dir, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                if dir == double_coef_dir:
                    A[node, neighbor_node] = 2
                elif dir in single_coef_dirs:
                    A[node, neighbor_node] = 1
                        
            A[node, node] = -2 * (((h * dx) / k) + 2)
            b[node] = -2 * h * dx * self.T_hot / k  # RHS term
            
        elif node_type in [NodeType.COLD_SURFACE.value, NodeType.ROOM_SURFACE.value]:
            k = self.k_outer
            h, T_inf = self.get_convection_params(x, y)
            
            # First check if node is actually on boundary
            if not (x == 0 or x == self.nx-1 or y == 0 or y == self.ny-1):
                # Treat as interface or interior node
                A[node, node] = -4  # Center node coefficient
                for _, pos in neighbors.items():
                    neighbor_node = self.get_node_number(*pos)
                    A[node, neighbor_node] = 1
                b[node] = 0
                return
            
            # All possible directions
            all_dirs = {'n', 's', 'e', 'w'}
    
            # Determine which direction the surface faces based on position
            if x == 0:           # Left surface
                surface_dir = 'w'
            elif x == self.nx-1: # Right surface
                surface_dir = 'e'
            elif y == 0:         # Bottom surface
                surface_dir = 's'
            elif y == self.ny-1: # Top surface
                surface_dir = 'n'
            else:
                raise ValueError(f"Surface node at ({x},{y}) not on boundary")
            double_coef_dir = {
                    'n': 's',
                    's': 'n',
                    'e': 'w',
                    'w': 'e'
                }[surface_dir]
            single_coef_dirs = ['n', 's'] if surface_dir in ['e', 'w'] else ['e', 'w']
            for dir, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                if dir == double_coef_dir:
                    A[node, neighbor_node] = 2
                elif dir in single_coef_dirs:
                    A[node, neighbor_node] = 1
                    
            A[node, node] = -2 * (((h * dx) / k) + 2)
            b[node] = -2 * h * dx * T_inf / k
            
        elif node_type == NodeType.INNER_INTERIOR.value:
            # Standard 5-point stencil for interior nodes (Equation 4.29)
            # T_{m,n+1} + T_{m,n-1} + T_{m+1,n} + T_{m-1,n} - 4T_{m,n} = 0
            A[node, node] = -4  # Center node coefficient
            for _, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                A[node, neighbor_node] = 1  # Neighbor coefficients
            b[node] = 0  # RHS term

        elif node_type == NodeType.INTERFACE_SURFACE.value:
            # Get material properties on either side of interface
            k1 = self.k_inner
            k2 = self.k_outer
            
            # Determine interface orientation (vertical or horizontal)
            # All possible directions
            all_dirs = {'n', 's', 'e', 'w'}
            # Find which neighbors are inner vs outer material
            inner_neighbors = set()
            outer_neighbors = set()
            for dir, pos in neighbors.items():
                neighbor_type = self.get_node_type(*pos)
                if neighbor_type in [NodeType.INNER_INTERIOR.value, NodeType.INNER_SURFACE.value, NodeType.INNER_CORNER.value]:
                    inner_neighbors.add(dir)
                else:
                    outer_neighbors.add(dir)
            
            # For vertical interface (east-west neighbors are different materials)
            if 'e' in inner_neighbors and 'w' in outer_neighbors or 'w' in inner_neighbors and 'e' in outer_neighbors:
                # Apply harmonic mean for east-west conduction
                k_harmonic = 2 * k1 * k2 / (k1 + k2)
                # Regular averaging for north-south conduction
                k_avg = (k1 + k2) / 2
                
                for dir, pos in neighbors.items():
                    neighbor_node = self.get_node_number(*pos)
                    if dir in ['e', 'w']:
                        A[node, neighbor_node] = 1/k_harmonic * (1 / (dx * dx))
                    else:  # 'n', 's'
                        A[node, neighbor_node] = 1/k_avg * (1 / (dx * dx))
                A[node, node] = -2 * (1/k_harmonic + 1/k_avg) * (1 / (dx * dx))
                
            # For horizontal interface (north-south neighbors are different materials)
            else:
                # Apply harmonic mean for north-south conduction
                k_harmonic = 2 * k1 * k2 / (k1 + k2)
                # Regular averaging for east-west conduction
                k_avg = (k1 + k2) / 2
                
                for dir, pos in neighbors.items():
                    neighbor_node = self.get_node_number(*pos)
                    if dir in ['n', 's']:
                        A[node, neighbor_node] = 1/k_harmonic * (1 / (dx * dx))
                    else:  # 'e', 'w'
                        A[node, neighbor_node] = 1/k_avg * (1 / (dx * dx))
                A[node, node] = -2 * (1/k_harmonic + 1/k_avg) * (1 / (dx * dx))
            
            b[node] = 0  # No heat generation at interface

        elif node_type == NodeType.INTERFACE_CORNER.value:
            k1 = self.k_inner
            k2 = self.k_outer
            k_harmonic = 2 * k1 * k2 / (k1 + k2)
            
            # Identify corner type based on neighboring materials
            inner_neighbors = set()
            outer_neighbors = set()
            for dir, pos in neighbors.items():
                neighbor_type = self.get_node_type(*pos)
                if neighbor_type in [NodeType.INNER_INTERIOR.value, NodeType.INNER_SURFACE.value, NodeType.INNER_CORNER.value]:
                    inner_neighbors.add(dir)
                else:
                    outer_neighbors.add(dir)
            
            # Apply appropriate coefficients based on corner configuration
            for dir, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                if dir in inner_neighbors:
                    A[node, neighbor_node] = 1/k1 * (1 / (dx * dx))
                else:
                    A[node, neighbor_node] = 1/k2 * (1 / (dx * dx))
            
            # Central coefficient combines both materials' effects
            A[node, node] = -1 * (len(inner_neighbors)/k1 + len(outer_neighbors)/k2) * (1 / (dx * dx))
            b[node] = 0  # No heat generation at interface corner

        elif node_type == NodeType.OUTER_INTERIOR.value:
            # Standard 5-point stencil for interior nodes but with k_outer
            k = self.k_outer
            A[node, node] = -4  # Center node coefficient
            for _, pos in neighbors.items():
                neighbor_node = self.get_node_number(*pos)
                A[node, neighbor_node] = 1  # Neighbor coefficients
            b[node] = 0  # RHS term


    def create_system(self):
        """Create the system matrix and RHS vector."""
        n = self.nx * self.ny
        A = lil_matrix((n, n))
        b = np.zeros(n)
        
        # Loop starting from top-left, moving right and then down
        for y in range(self.ny-1, -1, -1):  # Start from ny-1 down to 0
            for x in range(self.nx):         # Left to right
                self.set_node_equations(A, b, x, y)
        
        return csc_matrix(A), b

    def solve(self):
        """Solve the system for temperature distribution."""
        A, b = self.create_system()
        print("A:\n",A)
        print("b:\n",b)
        T = spsolve(A, b)
        return T.reshape((self.ny, self.nx))

    def plot_solution(self, T):
        """Plot the temperature distribution."""
        plt.figure(figsize=(15, 12))
        
        # Create custom colormap
        colors = ['darkblue', 'blue', 'lightskyblue', 'yellow', 'orange', 'red', 'darkred']
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        
        # Plot temperature distribution
        im = plt.imshow(T, cmap=custom_cmap, origin='lower')
        plt.colorbar(im, label='Temperature (°C)')
        plt.title(f'Temperature Distribution - Case {self.case.upper()}')
        
        # Add temperature values
        for y in range(self.ny):
            for x in range(self.nx):
                temp_value = T[y, x]
                temp_norm = (temp_value - T.min()) / (T.max() - T.min())
                text_color = 'black' if 0.25 < temp_norm < 0.65 else 'white'
                
                plt.text(x, y, f'{temp_value:.1f}', 
                        ha='center', va='center',
                        color=text_color, fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.5, 
                                edgecolor='none', pad=0.5))
        
        # Add node type legend
        node_types = {
            'Hot Fluid': 0,
            'External Corner': 1,
            'Cold Surface': 2,
            'Room Surface': 3,
            'Interface Corner': 4,
            'Interface Surface': 5,
            'Inner Interior': 6,
            'Inner Corner': 7,
            'Inner Surface': 9
        }
        
        legend_elements = []
        for name, value in node_types.items():
            mask = self.node_types == value
            if np.any(mask):
                avg_temp = np.mean(T[mask])
                legend_elements.append(f'{name}: Avg {avg_temp:.1f}°C')
        
        plt.figtext(1.02, 0.5, '\n'.join(legend_elements), 
                   fontsize=8, va='center')
        
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.tight_layout()
        plt.show()

def main():
    print("\nSolving Case A...")
    solver_a = ChimneyAnalysis(case='a')
    T_a = solver_a.solve()
    solver_a.plot_solution(T_a)

    print("\nSolving Case B...")
    solver_b = ChimneyAnalysis(case='b')
    T_b = solver_b.solve()
    solver_b.plot_solution(T_b)

if __name__ == "__main__":
    main()
