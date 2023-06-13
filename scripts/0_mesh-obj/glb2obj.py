# -*- coding=utf-8 -*-
import bpy
import argparse
import sys
import os

if __name__ == '__main__':

    input_file = sys.argv[-1]
    print("input_file", input_file)

    # Set input and output file paths
    output_file = input_file.replace(".glb", ".obj")

    # Clear existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Import glTF 2.0 (GLB) file
    bpy.ops.import_scene.gltf(filepath=input_file)

    # Export to OBJ format
    bpy.ops.export_scene.obj(filepath=output_file, use_materials=True, use_selection=True)



