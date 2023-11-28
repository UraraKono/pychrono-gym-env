"""
Author: Urara Kono
"""

# import pychrono as chrono
# import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import json
import matplotlib.pyplot as plt
import yaml
import time
from argparse import Namespace
from EGP.regulators.pure_pursuit import *
from chrono_env.environment import ChronoEnv
from chrono_env.utils import *

# --------------
step_size = 2e-3
throttle_value = 0.3 # This shouldn't be set zero; otherwise it doesn't start
map_name = 'SaoPaulo'  
constant_friction = 1.0
SAVE_MODEL = True
t_end = 300
num_laps = 1  # Number of laps
# --------------

# Load map config file
with open('EGP/configs/config_%s.yaml' % 'SaoPaulo') as file:  # map_name -- SaoPaulo
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
if not map_name == 'custom_track':

    conf.wpt_path = './EGP' + conf.wpt_path

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

 
# friction = [0.4 + i/waypoints.shape[0] for i in range(waypoints.shape[0])]
friction = [constant_friction for i in range(waypoints.shape[0])]

# Define the patch coordinates
patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in waypoints]

Kp = 5
Ki = 0.01
Kd = 0

env = ChronoEnv().make(timestep=step_size, control_period=0.1, waypoints=waypoints,
        friction=friction, speedPID_Gain=[Kp, Ki, Kd],
        x0=waypoints[0,1], y0=waypoints[0,2], w0=waypoints[0,3]+np.pi/2)

# Init Pure-Pursuit regulator
work = {'mass': 2573.14, 'lf': 1.8496278, 'tlad': 10.6461887897713965, 'vgain': 1.0} # tlad: look ahead distance   
planner_pp = PurePursuitPlanner(conf, env.vehicle_params.WB)
planner_pp.waypoints = waypoints.copy()
ballT = env.vis.GetSceneManager().addSphereSceneNode(0.1)
ballT.getMaterial(0).EmissiveColor = chronoirr.SColor(0, 0, 255, 0)
plt.figure()
plt.plot(planner_pp.waypoints[:,1], planner_pp.waypoints[:,2], label="waypoints")
plt.show()

lap_counter = 0
speed    = 0
steering = 0
control_list = []
state_list = []
execution_time_start = time.time()

observation = {'poses_x': env.my_hmmwv.state[0],'poses_y': env.my_hmmwv.state[1],
                'vx':env.my_hmmwv.state[2], 'poses_theta': env.my_hmmwv.state[3],
                'vy':env.my_hmmwv.state[4],'yaw_rate':env.my_hmmwv.state[5],'steering':env.my_hmmwv.state[6]}

while env.lap_counter < num_laps:
    # Render scene
    env.render()

    if (env.step_number % (env.control_step) == 0):

        planner_pp.waypoints = waypoints.copy()
        speed, steering = planner_pp.plan(observation['poses_x'], observation['poses_y'], observation['poses_theta'],
                                        work['tlad'], work['vgain'])
        print("pure pursuit input speed", speed, "steering angle ratio [-1,1]", steering/env.vehicle_params.MAX_STEER)  
        u = np.array([speed, steering])
        # Visualize the lookahead point of pure-pursuit
        if planner_pp.lookahead_point is not None:
            pT = chrono.ChVectorD(planner_pp.lookahead_point[0], planner_pp.lookahead_point[1], 0.0)
            ballT.setPosition(chronoirr.vector3df(pT.x, pT.y, pT.z))
        else:
            print("No lookahead point found!")

        control_list.append(u) # saving acceleration and steering speed
        state_list.append(env.my_hmmwv.state)
    
    observation, reward, done, info = env.step(steering, speed)

    if env.time > t_end:
        print("env.time",env.time)
        break

execution_time_end = time.time()
print("execution time: ", execution_time_end - execution_time_start)

control_list = np.vstack(control_list)
state_list = np.vstack(state_list)

np.save("control.npy", control_list)
np.save("state.npy", state_list)

plt.figure()
plt.plot(env.t_stepsize, env.speed)
plt.title("longitudinal speed")
plt.xlabel("time [s]")
plt.ylabel("longitudinal speed [m/s]")
plt.savefig("longitudinal_speed.png")

plt.figure()
color = [i for i in range(len(env.x_trajectory))]
plt.scatter(env.x_trajectory, env.y_trajectory, c=color,s=1, label="trajectory")
plt.scatter(waypoints[0,1],waypoints[0,2], c='r',s=5, label="start")
plt.title("trajectory")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.savefig("trajectory.png")

plt.show()
