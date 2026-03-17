#!/usr/bin/env python
"""Minimal test script to debug segfault in eval mode."""

import os
os.environ['MUJOCO_GL'] = 'glfw'

import sys
print("Starting test...", file=sys.stderr)

import mujoco as mj
print(f"MuJoCo version: {mj.__version__}", file=sys.stderr)

# Try to create a simple GL context
print("Creating GL context...", file=sys.stderr)
gl_context = mj.gl_context.GLContext(640, 480)
print("GL context created successfully", file=sys.stderr)

# Make context current
print("Making context current...", file=sys.stderr)
gl_context.make_current()
print("Context made current", file=sys.stderr)

# Create a simple model
print("Loading model...", file=sys.stderr)
xml = """
<mujoco>
  <worldbody>
    <light name="light" pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <geom type="sphere" size="0.1"/>
      <camera name="cam" mode="fixed"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)
print("Model loaded", file=sys.stderr)

# Create MjrContext
print("Creating MjrContext...", file=sys.stderr)
mjr_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
print("MjrContext created", file=sys.stderr)

# Create scene
print("Creating scene...", file=sys.stderr)
scene = mj.MjvScene(model, maxgeom=1000)
print("Scene created", file=sys.stderr)

# Create camera and option
print("Creating camera and option...", file=sys.stderr)
camera = mj.MjvCamera()
option = mj.MjvOption()
print("Camera and option created", file=sys.stderr)

# Update scene
print("Updating scene...", file=sys.stderr)
mj.mjv_updateScene(model, data, option, None, camera, mj.mjtCatBit.mjCAT_ALL, scene)
print("Scene updated", file=sys.stderr)

# Render
print("Rendering...", file=sys.stderr)
viewport = mj.MjrRect(0, 0, 640, 480)
mj.mjr_render(viewport, scene, mjr_context)
print("Rendered successfully", file=sys.stderr)

# Clean up
print("Cleaning up...", file=sys.stderr)
gl_context.free()
print("Test completed successfully!", file=sys.stderr)
