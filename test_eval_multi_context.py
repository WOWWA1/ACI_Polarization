#!/usr/bin/env python
"""Test multiple GL contexts to debug segfault."""

import os
os.environ['MUJOCO_GL'] = 'glfw'

import sys
print("Starting multi-context test...", file=sys.stderr)

import mujoco as mj
print(f"MuJoCo version: {mj.__version__}", file=sys.stderr)

# Load a model that's similar to the compass scene
xml = """
<mujoco>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.5 0.7 1" rgb2="0.1 0.2 0.4" width="512" height="512"/>
  </asset>
  <worldbody>
    <light name="sun" pos="20 0 20" dir="-1 0 -1" diffuse="1 1 1"/>
    <light name="fill" pos="0 0 10" dir="0 0 -1" diffuse="0.3 0.3 0.3"/>
    <geom type="plane" size="10 10 0.1"/>
    <body name="agent" pos="0 0 0.2">
      <geom type="sphere" size="0.2"/>
      <camera name="eye" mode="fixed" pos="0.2 0 0"/>
    </body>
    <body name="goal" pos="5 0 0.5">
      <geom type="sphere" size="0.3" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mj.MjModel.from_xml_string(xml)
data = mj.MjData(model)
print("Model loaded", file=sys.stderr)

# Test 1: Create first context (for environment renderer)
print("\n=== Creating first GL context (env renderer) ===", file=sys.stderr)
gl_context1 = mj.gl_context.GLContext(640, 480)
gl_context1.make_current()
mjr_context1 = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
print("First context created", file=sys.stderr)

# Test 2: Create second context (for eye renderer) - this is what might cause issues
print("\n=== Creating second GL context (eye renderer) ===", file=sys.stderr)
gl_context2 = mj.gl_context.GLContext(16, 16)
gl_context2.make_current()
mjr_context2 = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
print("Second context created", file=sys.stderr)

# Test 3: Try to render with both contexts
print("\n=== Rendering with first context ===", file=sys.stderr)
gl_context1.make_current()
scene1 = mj.MjvScene(model, maxgeom=1000)
camera1 = mj.MjvCamera()
option1 = mj.MjvOption()
mj.mjv_updateScene(model, data, option1, None, camera1, mj.mjtCatBit.mjCAT_ALL, scene1)
viewport1 = mj.MjrRect(0, 0, 640, 480)
mj.mjr_render(viewport1, scene1, mjr_context1)
print("First render successful", file=sys.stderr)

print("\n=== Rendering with second context ===", file=sys.stderr)
gl_context2.make_current()
scene2 = mj.MjvScene(model, maxgeom=1000)
camera2 = mj.MjvCamera()
option2 = mj.MjvOption()
mj.mjv_updateScene(model, data, option2, None, camera2, mj.mjtCatBit.mjCAT_ALL, scene2)
viewport2 = mj.MjrRect(0, 0, 16, 16)
mj.mjr_render(viewport2, scene2, mjr_context2)
print("Second render successful", file=sys.stderr)

# Test 4: Switch back and render again
print("\n=== Switching back to first context and rendering ===", file=sys.stderr)
gl_context1.make_current()
mj.mjv_updateScene(model, data, option1, None, camera1, mj.mjtCatBit.mjCAT_ALL, scene1)
mj.mjr_render(viewport1, scene1, mjr_context1)
print("Third render successful", file=sys.stderr)

# Clean up
print("\n=== Cleaning up ===", file=sys.stderr)
gl_context1.free()
gl_context2.free()
print("Multi-context test completed successfully!", file=sys.stderr)
