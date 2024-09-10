import pybullet as p
import pybullet_data

# Start the physics engine
p.connect(p.GUI)

# Set up the environment (floor)
# get project path
import os
path = os.getcwd()
# get parent
path = os.path.dirname(path)

p.setAdditionalSearchPath(f'{path}\\panda_gym\\assets\\scenarios')
p.loadURDF("\\warehouse\\urdf\\warehouse.urdf")

# Run the simulation
while p.isConnected():
    p.stepSimulation()