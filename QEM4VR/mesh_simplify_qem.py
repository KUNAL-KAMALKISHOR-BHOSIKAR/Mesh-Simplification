import open3d as o3d
import numpy as np
import heapq
from collections import defaultdict
from numba import njit
import time
import itertools

class EdgePriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-edge>'
        self.counter = itertools.count()

    def add_or_update(self, edge, cost, v_bar):
        if edge in self.entry_finder:
            self.remove(edge)
        count = next(self.counter)
        entry = [cost, count, edge, v_bar]
        self.entry_finder[edge] = entry
        heapq.heappush(self.heap, entry)

    def remove(self, edge):
        entry = self.entry_finder.pop(edge)
        entry[2] = self.REMOVED

    def pop(self):
        while self.heap:
            cost, count, edge, v_bar = heapq.heappop(self.heap)
            if edge != self.REMOVED:
                self.entry_finder.pop(edge, None)
                return cost, edge, v_bar
        raise KeyError("Pop from empty priority queue")

    def empty(self):
        return not self.entry_finder
    
@njit
def add_plane_to_quadric(Q, plane):
    p = plane.reshape((4, 1))
    return Q + p @ p.T

class Quadric:
    def __init__(self):
        self.matrix = np.zeros((4, 4))

    def add_plane(self, plane):
        self.matrix = add_plane_to_quadric(self.matrix, plane)

    def __add__(self, other):
        result = Quadric()
        result.matrix = self.matrix + other.matrix
        return result

@njit
def compute_plane(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.zeros(4)
    n /= norm
    d = -np.dot(n, v0)
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)

def get_boundary_edges(mesh):
    triangles = np.asarray(mesh.triangles)
    edge_count = {}

    for tri in triangles:
        for i in range(3):
            u, v = sorted((tri[i], tri[(i + 1) % 3]))
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    return boundary_edges

@njit
def estimate_curvature(p0, p1, p2):
    return np.linalg.norm(p0 - 2 * p1 + p2)

def compute_boundary_curvature(vertices, boundary_edges):
    neighbors = defaultdict(list)
    for u, v in boundary_edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    curvature = {}
    for v in neighbors:
        if len(neighbors[v]) < 2:
            curvature[v] = 0.0
            continue
        v1, v2 = neighbors[v][:2]
        p0, p1, p2 = vertices[v1], vertices[v], vertices[v2]
        curvature[v] = estimate_curvature(p0, p1, p2)
    return curvature

def get_vertex_quadrics_qem4vr(mesh, vertex_attrs, Wb=5.0, Wt=1000.0):
    quadrics = [Quadric() for _ in range(len(mesh.vertices))]
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    boundary_edges = get_boundary_edges(mesh)
    curvature = compute_boundary_curvature(vertices, boundary_edges)

    boundary_map = {}
    for u, v in boundary_edges:
        boundary_map[u] = True
        boundary_map[v] = True

    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        plane = compute_plane(v0, v1, v2)
        Q = compute_quadric(plane)
        for idx in tri:
            quadrics[idx].add_plane(plane)

    # Add curvature-weighted boudary constraint quadrics
    for u, v in boundary_edges:
        edge = vertices[v] - vertices[u]
        if np.linalg.norm(edge) == 0:
            continue
        edge = edge / np.linalg.norm(edge)
        # Plane perpendicular to edge
        normal = np.cross(edge, [1, 0, 0])  # Arbitrary cross
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(edge, [0, 1, 0])
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, vertices[u])
        plane = np.append(normal, d)
        Qb = compute_quadric(plane)

        k_u = curvature.get(u, 0)
        k_v = curvature.get(v, 0)
        quadrics[u].matrix += Wb * k_u * Qb
        quadrics[v].matrix += Wb * k_v * Qb

    material_boundary_vertices, material_weights = detect_material_boundaries(mesh, vertex_attrs, Wt)

    # Apply material boundary weighting
    for idx, weight in material_weights.items():
        quadrics[idx].matrix *= weight  # Alternatively, we could add a penalty matrix instead of scaling

    return quadrics, set(boundary_map.keys()), material_boundary_vertices

def detect_material_boundaries(mesh, vertex_attrs, Wt=1000.0):
    material_boundary_vertices = set()
    material_weight_map = {}

    for vidx, attr in vertex_attrs.items():
        uv_set = {tuple(uv) for uv in attr["uvs"]}
        if len(uv_set) > 1:
            material_boundary_vertices.add(vidx)
            material_weight_map[vidx] = Wt

    return material_boundary_vertices, material_weight_map

@njit
def compute_quadric(plane):
    p = plane.reshape((4, 1))
    return p @ p.T

def simplify_mesh(mesh, target_face_count, Wb=5.0, Wt=1000.0):
    mesh = mesh.remove_duplicated_vertices().remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    uvs = np.asarray(mesh.triangle_uvs) if mesh.has_triangle_uvs() else None
    vertex_normals = np.asarray(mesh.vertex_normals)

    # Store per-vertex UVs and normals
    vertex_attrs = {
        idx: {
            "normal": vertex_normals[idx] if len(vertex_normals) > 0 else np.zeros(3),
            "uvs": [],
        }
        for idx in range(len(vertices))
    }

    if uvs is not None:
        # Map triangle uvs to vertices
        for tidx, tri in enumerate(triangles):
            for i in range(3):
                vertex_attrs[tri[i]]["uvs"].append(uvs[tidx * 3 + i])

    quadrics, boundary_vertices, material_boundary_vertices = get_vertex_quadrics_qem4vr(mesh, vertex_attrs, Wb, Wt)
    edge_map = defaultdict(int)

    for tri in triangles:
        for i in range(3):
            u, v = sorted([tri[i], tri[(i + 1) % 3]])
            edge_map[(u, v)] += 1

    edge_queue = EdgePriorityQueue()

    for (u, v) in edge_map.keys():
        Q = quadrics[u].matrix + quadrics[v].matrix
        try:
            A = Q[:3, :3]
            b = -Q[:3, 3]
            v_bar = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v_bar = (vertices[u] + vertices[v]) / 2

        v_bar_h = np.append(v_bar, 1)
        error = v_bar_h @ Q @ v_bar_h
        edge_queue.add_or_update((u, v), error, v_bar)

    new_vertices = list(vertices)
    collapsed = set()
    while len(triangles) > target_face_count and not edge_queue.empty():
        error, (u, v), v_bar = edge_queue.pop()
        if (u in collapsed or v in collapsed or 
            u in boundary_vertices or v in boundary_vertices or
            u in material_boundary_vertices or v in material_boundary_vertices):
            continue

        new_vertices[u] = v_bar
        collapsed.add(v)

        # Average normals
        n1 = vertex_attrs[u]["normal"]
        n2 = vertex_attrs[v]["normal"]
        vertex_attrs[u]["normal"] = (n1 + n2) / 2

        # Merge UVs
        uvs_u = vertex_attrs[u]["uvs"]
        uvs_v = vertex_attrs[v]["uvs"]
        vertex_attrs[u]["uvs"] = uvs_u + uvs_v

        # Remove faces with v; replace v with u elsewhere
        new_faces = []
        for tri in triangles:
            tri = list(tri)
            if v in tri:
                tri = [u if x == v else x for x in tri]
            if tri[0] != tri[1] and tri[1] != tri[2] and tri[2] != tri[0]:
                new_faces.append(tri)
        triangles = new_faces
        
        neighbors = set()
        for tri in triangles:
            if u in tri:
                for i in range(3):
                    neighbors.update([tri[i], tri[(i + 1) % 3]])
        neighbors.discard(v)
        for n in neighbors:
            i, j = sorted([u, n])
            Q = quadrics[i].matrix + quadrics[j].matrix
            try:
                v_bar = np.linalg.solve(Q[:3, :3], -Q[:3, 3])
            except np.linalg.LinAlgError:
                v_bar = (new_vertices[i] + new_vertices[j]) / 2
            v_bar_h = np.append(v_bar, 1)
            error = v_bar_h @ Q @ v_bar_h
            edge_queue.add_or_update((i, j), error, v_bar)

    # Update final attributes
    final_normals = np.zeros((len(new_vertices), 3))

    final_normals = np.array([attrs["normal"] / np.linalg.norm(attrs["normal"]) if np.linalg.norm(attrs["normal"]) > 1e-6 else [0, 0, 1] for attrs in vertex_attrs.values()])

    # Collect all per-vertex UV lists into one array with NaNs where missing
    max_idx = max(vertex_attrs.keys()) + 1
    uv_lengths = [len(vertex_attrs[i]["uvs"]) for i in range(max_idx)]

    # Preallocate
    final_uvs = np.zeros((max_idx, 2))
    has_uv = np.zeros(max_idx, dtype=bool)

    for i in range(max_idx):
        uvs = np.array(vertex_attrs[i]["uvs"])
        if uvs.size > 0:
            final_uvs[i] = uvs.mean(axis=0)
            has_uv[i] = True

    # Only keep UVs for vertices that have them (optional)
    final_uvs = final_uvs[has_uv]

    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_normals = o3d.utility.Vector3dVector(final_normals)

    if len(final_uvs) == len(mesh.triangles) * 3:
        mesh.triangle_uvs = o3d.utility.Vector2dVector(final_uvs)

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    return mesh

def main():
    mesh = o3d.io.read_triangle_mesh("rocker-arm.ply")
    print("Original triangle count:", len(mesh.triangles))
    start = time.time()
    simplified = simplify_mesh(mesh, target_face_count=int(len(mesh.triangles) * 0.1), Wb=5.0, Wt=1000.0)
    print("Simplified triangle count:", len(simplified.triangles))
    print(f"Simplification took {time.time() - start:.2f} seconds")

    o3d.io.write_triangle_mesh('rocker-arm-small.ply', simplified)
    
if __name__ == "__main__":
    main()