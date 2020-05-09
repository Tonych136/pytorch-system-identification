from policy_transfer.utils.mlp import *
from policy_transfer.utils.optimizer import *
from policy_transfer.uposi.osi_env_wrapper import *
import numpy as np
import gym, joblib, tensorflow as tf
import policy_transfer.envs
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import *
from baselines import logger
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.running_mean_std import RunningMeanStd
from mpi4py import MPI

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
from numpy import linalg as LA
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

def osi_train_callback(model, name, iter):
    params = model.get_variable_dict()
    joblib.dump(params, logger.get_dir()+'/osi_params.pkl', compress=True)

class policy_normalization_remove:
    def __init__(self, policy, ob_rms, device, env):
        self.policy = policy
        self.ob_rms = ob_rms
        self.eval_recurrent_hidden_states = torch.zeros(
            1, self.policy.recurrent_hidden_state_size, device=device)
        self.eval_masks = torch.zeros(1, 1, device=device)
        self.device = device
        self.env = env
        self.obs = self.env.reset()
    def act(self, random = False):
        ob = torch.tensor([np.clip((self.obs - self.ob_rms.mean) /
                      np.sqrt(self.ob_rms.var + 1e-08),
                      -10.0, 10.0)],
                      dtype=torch.float32,
                      device=self.device)
        with torch.no_grad():
            _, action, _, self.eval_recurrent_hidden_states = actor_critic.act(
                ob,
                self.eval_recurrent_hidden_states,
                self.eval_masks,
                deterministic=True)
        if random:
            self.obs, rew, done, infos = self.env.step(action[0].cpu().numpy() + np.random.normal(0, args.action_noise, len(action[0].cpu().numpy())))
        else:
            self.obs, rew, done, infos = self.env.step(action[0].cpu().numpy())
        self.eval_masks = torch.tensor(
            [[0.0] if done else [1.0]],
            dtype=torch.float32,
            device=self.device)
        return self.obs, rew, done, infos, action[0].cpu().numpy()
    def reset(self):
        o = self.env.reset()
        self.obs = o
        self.eval_recurrent_hidden_states = torch.zeros(
            1, self.policy.recurrent_hidden_state_size, device=device)
        self.eval_masks = torch.zeros(1, 1, device=device)
        return o

class CEM:
    # CEM optimizer, it's used for minimization instead of maximization!!!
    # the eval_function should evaluate the costs!!!
    def __init__(self, eval_function, iter_num, num_mutation, num_elite, mean, std=0.2):
        self.eval_function = eval_function

        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.std = std # initial std
        self.mean = mean

    def find_min(self):
        weights_pop = [self.mean + self.std*np.random.randn(len(self.mean)) for i_weight in range(self.num_mutation)]

        for i in range(self.iter_num):
            rewards = [self.eval_function.reward(weights) for weights in weights_pop]
            elite_idxs = np.argsort(np.array(rewards))[0:self.num_elite]
            elite_weights = [weights_pop[idx] for idx in elite_idxs]
            mean = np.array(elite_weights).mean(axis = 0)
            std = np.array(elite_weights).std(axis = 0)
            weights_pop = [mean + std*np.random.randn(len(self.mean)) for i_weight in range(self.num_mutation)]
            print(i)
            print(weights_pop)
            print(mean)
            print(std)
        elite_idxs = np.argsort(np.array(rewards))[0:self.num_elite]
        return weights_pop[elite_idxs[0]]

class reward_env:
    def __init__(self, action_traj, state_traj, env):
        self.action_traj = action_traj
        self.state_traj = state_traj
        self.env = env

    def reward(self, params):
        for i in range(len(params)):
            if params[i] < 0:
                return np.Infinity
        self.env.env.param_manager.set_simulator_parameters(params)
        self.env.reset()
        total_diff = 0
        for i in range(len(self.action_traj)):
            self.env.set_state_vector(self.state_traj[i][0])
            states = [self.state_traj[i][0]]
            for action in self.action_traj[i]:
                obs, rew, done, infos = self.env.step(action)
                states.append(obs)
            total_diff = total_diff + self._traj_diff(states, self.state_traj[i])
        return total_diff
    
    def _traj_diff(self, traj_state1, traj_state2):
        total_distance = 0
        for i in range(min(len(traj_state1), len(traj_state2))):
            total_distance = total_distance + LA.norm(traj_state1[i] - traj_state2[i])
        for i in range(min(len(traj_state1), len(traj_state2)), 
                        max(len(traj_state1), len(traj_state2))):
            if len(traj_state1) <= len(traj_state2):
                total_distance = total_distance + LA.norm(traj_state2[i])
            else:
                total_distance = total_distance + LA.norm(traj_state1[i])
        return total_distance


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopperPT-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--OSI_hist', help='history step size', type=int, default=10)
    parser.add_argument('--policy_path', help='path to policy', type=str, default="")
    parser.add_argument('--dyn_params', action='append', type=int)

    parser.add_argument('--osi_iteration', help='number of iterations', type=int, default=6)
    parser.add_argument('--training_sample_num', help='number of training samples per iteration', type=int, default=20000)
    parser.add_argument('--action_noise', help='noise added to action', type=float, default=0.0)

    args = parser.parse_args()

    # extract information from args
    name = args.name
    seed = args.seed
    OSI_hist = args.OSI_hist
    policy_path = args.policy_path
    osi_iteration = args.osi_iteration
    training_sample_num = args.training_sample_num
    dyn_params = args.dyn_params


    env_real = gym.make(args.env)
    env_real.env.param_manager.activated_param = dyn_params
    env_real.env.param_manager.controllable_param = dyn_params

    result = torch.load("/home/tonyyang/Desktop/policy_transfer/policy_transfer/ppo/pytorch_ppo/trained_copy/ppo/UniversalPolicy.pt")
    actor_critic = result[0]
    device = torch.device("cuda:1")
    actor_critic.to(device)
    ob_rms = result[1]

    env_real.seed(seed)

    env_real.reset()
    print(env_real.env.param_manager.get_simulator_parameters())

    input_data = []
    output_data = []

    env = gym.make(args.env)
    env.seed(seed)
    env.env.param_manager.activated_param = dyn_params
    env.env.param_manager.controllable_param = dyn_params
    action_record = []
    state_record = []

    for i in range(100):
        done = False
        ob = env_real.reset()
        traj_action = []
        traj_state = []
        traj_state.append(env_real.state_vector())
        while not done:
            action = env_real.action_space.sample()
            ob, reward, done, _ = env_real.step(action)
            traj_action.append(action)
            traj_state.append(ob)
        action_record.append(traj_action)
        state_record.append(traj_state)
    action_record = np.asarray(action_record)
    state_record = np.asarray(state_record)

    reward_fun = reward_env(action_record, state_record, env)
    cem = CEM(reward_fun, iter_num = 30, num_mutation = 50, num_elite = 20, mean = [0.5, 0.5], std=0.2)
    print(cem.find_min())









     


   

