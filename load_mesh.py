import trimesh


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    Load a mesh from disk and return it as a trimesh.Trimesh object.
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single mesh, but got type: {type(mesh)}")

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("Mesh is empty or invalid.")

    return mesh