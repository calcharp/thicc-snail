import bpy
import bmesh
from dataclasses import dataclass

@dataclass
class VertexGroupData:
    label: str
    vertex_indices: list[int]
    uv_coords: list[tuple[float, float]] = None
    displacement_map_path: str = None
  
class Shell:
    def __init__(self, label, vertices, vertex_groups_data):
        """
        Initialize a new shell instance.

        Parameters:
        label (str): A label for the shell instance.
        vertices (list): A list of vertex coordinates (tuples of x, y, z).
        vertex_groups_data (list): A list of VertexGroupData instances, each containing information about a vertex group.
        """
        self.label: str = label
        self.vertices: list[tuple[float, float, float]] = vertices
        self.vertex_groups: list[VertexGroupData] = vertex_groups_data

        self.mesh: bpy.types.Mesh = None
        self.object: bpy.types.Object = None
        self.create_mesh()
        self.add_vertices()

        if self.vertex_groups:
            self.add_vertex_groups()
            for group in self.vertex_groups:
                if group.uv_coords is not None:
                    self.add_UV_map(group.label)
                if group.displacement_map_path is not None:
                    self.add_displacement_map(group.label)
                if group.uv_coords is not None:
                    self.add_texture(group.label, group.displacement_map_path)

    def create_mesh(self):
        # Create a new mesh and object in Blender with the name self.label
        self.mesh = bpy.data.meshes.new(self.label)
        self.object = bpy.data.objects.new(self.label, self.mesh)
        bpy.context.collection.objects.link(self.object)

    def add_vertices(self):
        pass

    def add_vertex_groups(self):
        pass

    def add_UV_map(self, group_label: str):
        pass

    def add_displacement_map(self, group_label: str):
        pass

    def add_texture(self, group_label: str, texture_path: str):
        pass
