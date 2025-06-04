import open3d as o3d
import numpy as np
import heapq

class Vertex:
    def __init__(self, position, normal=None, tex_coords=None, color=None, material=None):
        self.pos = np.array(position)
        self.normal = normal
        self.tex_coords = tex_coords or []
        self.color = color
        self.material = material
        self.Q = np.zeros((4, 4))  # Quadric matrix
        self.is_boundary = False
        self.curvature = 0
        self.is_critical = False

class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.cost = None
        self.opt_pos = None

def compute_plane_equation(face):
    # face: list of 3 vertices
    v0, v1, v2 = face
    normal = np.cross(v1.pos - v0.pos, v2.pos - v0.pos)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, v0.pos)
    return np.append(normal, d)

def compute_quadric(plane_eqn):
    a, b, c, d = plane_eqn
    p = np.array([[a], [b], [c], [d]])
    return p @ p.T

def update_vertex_quadric(vertex, incident_faces):
    for face in incident_faces:
        plane = compute_plane_equation(face)
        vertex.Q += compute_quadric(plane)

def compute_edge_cost(edge):
    Q = edge.v1.Q + edge.v2.Q
    try:
        v_bar = np.linalg.solve(Q[:3, :3], -Q[:3, 3])
        v_bar = np.append(v_bar, 1)
    except np.linalg.LinAlgError:
        # fallback to midpoint
        v_bar = np.append((edge.v1.pos + edge.v2.pos) / 2, 1)
    cost = v_bar.T @ Q @ v_bar
    edge.opt_pos = v_bar[:3]
    edge.cost = cost

def simplify_mesh(mesh, target_face_count):
    mesh = mesh.remove_duplicated_vertices().remove_degenerate_triangles()
    heap = []

    for edge in edges:
        compute_edge_cost(edge)
        heap.append(edge)
    heap.sort(key=lambda e: e.cost)

    while len(vertices) > target_face_count:
        edge = heap.pop(0)
        # Collapse edge.v1 and edge.v2 to edge.opt_pos
        # Transfer surface properties if needed
        # Update affected edges and costs
        # Omitted for brevity
        pass

    return vertices

def main():
    mesh = o3d.io.read_triangle_mesh("Axle_shaft.ply")
    print("Original triangle count:", len(mesh.triangles))
    simplified = simplify_mesh(mesh, target_face_count=int(len(mesh.triangles) * 0.1))
    print("Simplified triangle count:", len(simplified.triangles))

    o3d.visualization.draw_geometries([simplified])

if __name__ == "__main__":
    main()