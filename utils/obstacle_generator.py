import trimesh
import numpy as np
import pybullet as p

class ObstacleGenerator:
    def __init__(self):
        self.complexity_range = (0.2, 0.8)
        self.size_range = (0.1, 0.5)

    def generate_batch(self, n_obstacles):
        obstacles = []
        for _ in range(n_obstacles):
            complexity = np.random.uniform(*self.complexity_range)
            size = np.random.uniform(*self.size_range)

            mesh = generate_perturbed_primitive(
                complexity=complexity
            )
            mesh.apply_scale(size)
            obstacles.append(mesh)

        return obstacles


def generate_perturbed_primitive(base_type='cube', complexity=1.0):
    """
    Generate a procedural mesh by perturbing primitive shapes

    Args:
        base_type: Starting primitive ('cube', 'sphere', 'cylinder')
        complexity: Controls amount of perturbation (0-1)
    """
    # Create base mesh
    if base_type == 'cube':
        mesh = trimesh.creation.box()
    elif base_type == 'sphere':
        mesh = trimesh.creation.icosphere()

    # Add vertex noise
    vertices = mesh.vertices
    noise_amplitude = complexity * 0.3
    noise = np.random.normal(0, noise_amplitude, size=vertices.shape)

    # Apply noise while preserving volume
    new_vertices = vertices + noise
    new_mesh = trimesh.Trimesh(vertices=new_vertices,
                               faces=mesh.faces)

    # Ensure mesh remains manifold
    new_mesh.fix_normals()

    return new_mesh