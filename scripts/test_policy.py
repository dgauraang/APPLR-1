######################################################################################################
# (1) The script loads and deploys a policy and test the performance locally on all the 50 test worlds
# (2) Run roscore in the background first 
######################################################################################################
import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from os.path import join, dirname, abspath

import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from jackal_navi_envs.jackal_env_wrapper import *
import rospy

import gym
import numpy
import torch
from torch import nn

import argparse
from datetime import datetime
import time
import os
import json

from continuous.policy import TD3Policy
from sac.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.data import Batch

from jackal_navi_envs.APPLX import APPLD_policy, APPLE_policy, APPLI_policy
APPLD_policy = APPLD_policy()
APPLE_policy = APPLE_policy()
APPLI_policy = APPLI_policy()
APPLX = {
    "appld": lambda obs: APPLD_policy.forward(obs), 
    "appli": lambda obs: APPLI_policy.forward(obs),
    "apple": lambda obs: APPLE_policy.forward(obs),
    "dwa": lambda obs: np.array([0.5, 1.57, 6, 20, 0.1, 0.75, 1, 0.3])
}

parser = argparse.ArgumentParser(description = 'Jackal navigation simulation')
parser.add_argument('--model', dest = 'model', type = str, default = 'results/DQN_testbed_2020_08_30_10_58', help = 'path to the saved model and configuration')
parser.add_argument('--policy', dest = 'policy', type = str, default = 'policy_26.pth')
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--APPLX', dest='applx', type = str, default = "", help = "policy from other algorithms. Can be dwa, appld, appli, apple")
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--seed', dest='seed', type = int, default = 43)
parser.add_argument('--avg', dest='avg', type = int, default = 2)
parser.add_argument('--noise', dest='noise', action='store_true')
parser.add_argument('--save', dest='save', type = str, default = "test_result.txt")

args = parser.parse_args()
model_path = args.model
record = args.record
gui = 'true' if args.gui else 'false'
seed = args.seed
avg = args.avg
applx = args.applx
policy = args.policy
noise = args.noise
save = args.save

config_path = model_path + '/config.json'
model_path = join(model_path, policy)

with open(config_path, 'rb') as f:
    config = json.load(f)

def write_file(s):
    outf = open(save, "a")
    outf.write(s)
    outf.close()

print("log to %s" %(save))
write_file("Start logging the test with model %s\n" %(model_path))
if applx:
    write_file("Using %s parameter\n" %(applx)) 
env_config = config['env_config']
env_config['gui'] = gui
wrapper_config = config['wrapper_config']
training_config = config['training_config']
if record:
    env_config['world_name'] = env_config['world_name'].split('.')[0] + '_camera' + '.world'

worlds = [54, 94, 156, 68, 52, 101, 40, 135, 51, 42, 75, 67, 18, 53, 87, 36, 28, 61, 233, 25, 35, 20, 34, 79, 108, 46, 65, 90, 6, 73, 70, 10, 29, 167, 15, 31, 77, 116, 241, 155, 194, 99, 56, 149, 38, 261, 239, 234, 60, 173, 247, 178, 291, 16, 9, 21, 169, 257, 148, 296, 151, 259, 102, 145, 130, 205, 121, 105, 43, 242, 213, 171, 62, 202, 293, 224, 225, 152, 111, 55, 125, 200, 161, 1, 136, 106, 286, 139, 244, 230, 222, 238, 170, 267, 26, 132, 124, 23, 59, 3, 97, 119, 89, 12, 164, 39, 236, 263, 81, 188, 84, 11, 268, 192, 122, 22, 253, 219, 216, 137, 85, 195, 206, 212, 4, 274, 91, 248, 44, 131, 203, 63, 80, 37, 110, 50, 74, 120, 128, 249, 30, 14, 103, 49, 154, 82, 2, 143, 158, 147, 235, 83, 157, 142, 187, 185, 288, 45, 140, 271, 160, 146, 109, 223, 126, 98, 252, 134, 272, 115, 71, 117, 255, 141, 174, 33, 245, 92, 295, 281, 186, 260, 7, 166, 196, 66, 113, 153, 227, 107, 199, 298, 278, 114, 72, 165, 228, 176, 24, 162, 198, 180, 285, 232, 243, 207, 190, 262, 275, 172, 179, 269, 127, 86, 183, 273, 287, 215, 266, 95, 5, 299, 279, 13, 250, 96, 197, 177, 58, 289, 211, 220, 182, 282, 210, 280, 251, 283, 217, 276, 292, 221, 204, 191, 181, 209, 297, 264, 231, 254]

worlds = [w for w in range(300) if w not in worlds]
env_config['world_name'] = 'Benchmarking/test/world_%d.world' %(worlds[0])

rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
rospy.set_param('/use_sim_time', True)
env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous_reset-v1', **env_config), **wrapper_config['wrapper_args'])
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
print(state_shape, action_shape)
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(training_config['num_layers'], state_shape, device=device, hidden_layer_size=training_config['hidden_size'])

if config['section'] == 'SAC':
    actor = ActorProb(
        net, action_shape,
        1, device, hidden_layer_size=training_config['hidden_size']
    ).to(device)
else:
    actor = Actor(
        net, action_shape,
        1, device, hidden_layer_size=training_config['hidden_size']
    ).to(device)

actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config['actor_lr'])
net = Net(training_config['num_layers'], state_shape,
          action_shape, concat=True, device=device, hidden_layer_size=training_config['hidden_size'])
critic1 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=training_config['critic_lr'])
critic2 = Critic(net, device, hidden_layer_size=training_config['hidden_size']).to(device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=training_config['critic_lr'])
if config['section'] == 'SAC':
    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        action_range=[env.action_space.low, env.action_space.high],
        tau=training_config['tau'], gamma=training_config['gamma'],
        reward_normalization=training_config['rew_norm'],
        ignore_done=training_config['ignore_done'],
        alpha=training_config['sac_alpha'],
        exploration_noise=None,
        estimation_step=training_config['n_step'])
else:
    policy = TD3Policy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        action_range=[env.action_space.low, env.action_space.high],
        tau=training_config['tau'], gamma=training_config['gamma'],
        exploration_noise=GaussianNoise(sigma=training_config['exploration_noise']),
        policy_noise=training_config['policy_noise'],
        update_actor_freq=training_config['update_actor_freq'],
        noise_clip=training_config['noise_clip'],
        reward_normalization=training_config['rew_norm'],
        ignore_done=training_config['ignore_done'],
        estimation_step=training_config['n_step'])
print(training_config['hidden_size'])

state_dict = torch.load(model_path)
policy.load_state_dict(state_dict)
# model_path = "actor.pth"
# state_dict = torch.load(model_path, map_location=torch.device('cpu'))
# policy.actor.load_state_dict(state_dict)

if not noise:
    policy._noise = None
print(env.action_space.low, env.action_space.high)

for w in worlds:
    if w != worlds[0]:
        env.close()
        env_config['world_name'] = 'Benchmarking/test/world_%d.world' %(w)
        # env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous-v1', **env_config), **wrapper_config['wrapper_args'])
        env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous_reset-v1', **env_config), **wrapper_config['wrapper_args'])
    rs = []
    cs = []
    pms = np.array(env_config['param_init'])
    pms = np.expand_dims(pms, -1)
    succeed = 0
    for i in range(avg):
        print(">>>>>>>>>>>>>>>>>>>>>> Running world_%d: %d/%d <<<<<<<<<<<<<<<<<<<<<<<<<<" %(w, i+1, avg))
        r = 0
        f = False
        count = 0
        obs = env.reset()
        gp = env.gp
        scan = env.scan
        done = False
        while not done:
            obs_batch = Batch(obs=[obs], info={})
            obs_x = [scan, gp]
            if not applx:
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
                #print(policy.actor(obs_batch['obs']))
            else:
                actions = APPLX[applx](obs_x)
            obs_new, rew, done, info = env.step(actions)
            gp = info.pop("gp")
            scan = info.pop("scan")
            obs = obs_new
            # plt.plot(obs)
            # plt.show()

            print('current step: %d, X position: %f, Y position: %f, rew: %f' %(count, info['X'], info['Y'] , rew))
            print(info['params'])
            # params = np.array(info['params'])
            params = np.array(info['params'])
            pms = np.append(pms, np.expand_dims(params, -1), -1)
            r += rew
            count += 1
        if count != env_config['max_step'] and rew > -100:
            f = True
            succeed += 1
            rs.append(r)
            cs.append(count)
        write_file("%d %d %f %d\n" %(w, count*env_config['time_step'], r, f))
    try:
        print("succeed: %d/%d \t episode reward: %.2f \t steps: %d" %(succeed, avg, sum(rs)/float((len(rs))), sum(cs)/float((len(cs)))))
    except:
        pass
write_file("Finshed!\n")
env.close()

######## About recording ###########
# Add the camera model to the world you used to train
# Check onetwofive_meter_split_camera.world
# The frames will be save to folder /tmp/camera_save_tutorial
# run
# ffmpeg -r 30 -pattern_type glob -i '/tmp/camera_save_tutorial/default_camera_link_my_camera*.jpg' -c:v libx264 my_camera.mp4
# to generate the video
