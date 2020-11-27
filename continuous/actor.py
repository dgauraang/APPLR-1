import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import jackal_navi_envs
from torch import nn
import torch
import gym
import numpy as np
import random
import time
from policy import TD3Policy
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Batch

random.seed(43)
<<<<<<< HEAD
benchmarking_train = [54, 94, 156, 68, 52, 101, 40, 135, 51, 42, 75, 67, 18, 53, 87, 36, 28, 61, 233, 25, 35, 20, 34, 79, 108, 46, 65, 90, 6, 73, 70, 10, 29, 167, 15, 31, 77, 116, 241, 155, 194, 99, 56, 149, 38, 261, 239, 234, 60, 173, 247, 178, 291, 16, 9, 21, 169, 257, 148, 296, 151, 259, 102, 145, 130, 205, 121, 105, 43, 242, 213, 171, 62, 202, 293, 224, 225, 152, 111, 55, 125, 200, 161, 1, 136, 106, 286, 139, 244, 230, 222, 238, 170, 267, 26, 132, 124, 23, 59, 3, 97, 119, 89, 12, 164, 39, 236, 263, 81, 188, 84, 11, 268, 192, 122, 22, 253, 219, 216, 137, 85, 195, 206, 212, 4, 274, 91, 248, 44, 131, 203, 63, 80, 37, 110, 50, 74, 120, 128, 249, 30, 14, 103, 49, 154, 82, 2, 143, 158, 147, 235, 83, 157, 142, 187, 185, 288, 45, 140, 271, 160, 146, 109, 223, 126, 98, 252, 134, 272, 115, 71, 117, 255, 141, 174, 33, 245, 92, 295, 281, 186, 260, 7, 166, 196, 66, 113, 153, 227, 107, 199, 298, 278, 114, 72, 165, 228, 176, 24, 162, 198, 180, 285, 232, 243, 207, 190, 262, 275, 172, 179, 269, 127, 86, 183, 273, 287, 215, 266, 95, 5, 299, 279, 13, 250, 96, 197, 177, 58, 289, 211, 220, 182, 282, 210, 280, 251, 283, 217, 276, 292, 221, 204, 191, 181, 209, 297, 264, 231, 254]
benchmarking_train = 1*benchmarking_train[20:50] + 2*benchmarking_train[50:150] + 4*benchmarking_train[150:200] + 2*benchmarking_train[200:240]
benchmarking_train = benchmarking_train*3
# benchmarking_train = benchmarking_train[240:]*300
=======
benchmarking_train = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 130, 131, 132, 134, 135, 136, 137, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 190, 191, 192, 194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 205, 206, 207, 209, 210, 211, 212, 213, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 228, 230, 231, 232, 233, 234, 235, 236, 238, 239, 241, 242, 243, 244, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 289, 291, 292, 293, 295, 296, 297, 298, 299]
benchmarking_train = benchmarking_train[:150] + 2*benchmarking_train[150:200] + 4*benchmarking_train[200:]
>>>>>>> 6d70b141e86f4679b4f0f61162189ad35b91aa59
random.shuffle(benchmarking_train)

BASE_PATH = '/u/zifan/buffer'
#BASE_PATH = '/home/gauraang/buffer'

def init_actor(id):
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))
    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    while f is None:
        try:
            f = open(join(BASE_PATH, 'config.json'), 'rb')
        except:
            time.sleep(2)

    config = json.load(f)

    num_env = config['training_config']['num_actor']
    def count_actor():
        files = os.listdir(BASE_PATH)
        num_actor = sum([f.startswith("actor_") for f in files])
        return num_actor
    # wait until most of actors successfully initialized, tolerance 50 envs
    #while num_env-count_actor() > 50:
    #    time.sleep(10)

    return config

def load_model(model):
    model_path = join(BASE_PATH, 'policy.pth')
    state_dict = {}
    state_dict_raw = None
    while state_dict_raw is None:
        try:
            state_dict_raw = torch.load(model_path)
        except:
            time.sleep(0.1)
            pass

    model.load_state_dict(state_dict_raw)
    model = model.float()
    # exploration noise std
    with open(join(BASE_PATH, 'eps.txt'), 'r') as f:
        eps = float(f.readlines()[0])

    return model, eps

def write_buffer(traj, ep, id):
    with open(join(BASE_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        pickle.dump(traj, f)

def main(id):

    config = init_actor(id)
    env_config = config['env_config']
    if env_config['world_name'] != "sequential_applr_testbed.world":
        env_config['world_name'] = 'Benchmarking/train/world_%d.world' %(benchmarking_train[id])
        assert os.path.exists('/jackal_ws/src/jackal_helper/worlds/Benchmarking/train/world_%d.world' %(benchmarking_train[id]))
    wrapper_config = config['wrapper_config']
    training_config = config['training_config']
    wrapper_dict = jackal_navi_envs.jackal_env_wrapper.wrapper_dict
    if config['env'] == 'jackal':
        env = wrapper_dict[wrapper_config['wrapper']](gym.make('jackal_continuous-v0', **env_config), **wrapper_config['wrapper_args'])
    else:
        env = gym.make('Pendulum-v0')
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(training_config['num_layers'], state_shape, device=device, hidden_layer_size=training_config['hidden_size'])
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
    print(env.action_space.low, env.action_space.high)
    ep = 0
    while True:
        obs = env.reset()
        obs_batch = Batch(obs=[obs], info={})
        ep += 1
        traj = []
        done = False
        count = 0
        policy, eps = load_model(policy)
        policy.set_exp_noise(GaussianNoise(sigma=eps))
        while not done:
            time.sleep(0.01)
            p = random.random()
            obs = torch.tensor([obs]).float()
            actions = policy(obs_batch).act.cpu().detach().numpy()
            #actions = np.array([0.5, 1.57, 6, 20, 0.3])
            obs_new, rew, done, info = env.step(actions.reshape(-1))
            count += 1
            # print('current step: %d, X position: %f, Y position: %f, rew: %f, succeed: %d' %(count, info['X'], info['Y'], rew, info['succeed']), end = '\r')
            traj.append([obs, actions, rew, done, info])
            obs_batch = Batch(obs=[obs_new], info={})
            obs = obs_new
        # print('count: %d, rew: %f' %(count, rew))
        write_buffer(traj, ep, id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)
