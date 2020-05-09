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
from mpi4py import MPI
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

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
        return self.obs, rew, done, infos
    def reset(self):
        o = self.env.reset()
        self.obs = o
        self.eval_recurrent_hidden_states = torch.zeros(
            1, self.policy.recurrent_hidden_state_size, device=device)
        self.eval_masks = torch.zeros(1, 1, device=device)
        return o


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHopperPT-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--name', help='name of experiments', type=str, default="")
    parser.add_argument('--OSI_hist', help='history step size', type=int, default=10)
    parser.add_argument('--policy_path', help='path to policy', type=str, default="")
    parser.add_argument('--dyn_params', action='append', type=int)

    parser.add_argument('--osi_iteration', help='number of iterations', type=int, default=100)
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


    # setup the environments
    # if use minitaur environment, set up differently
    if args.env == 'Minitaur':
        from pybullet_envs.minitaur.envs import minitaur_reactive_env
        from gym.wrappers import time_limit

        env_hist = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                                                                             accurate_motor_model_enabled=True,
                                                                             urdf_version='rainbow_dash_v0',
                                                                             include_obs_history=OSI_hist,
                                                                             include_act_history=0,
                                                                             train_UP=False),
                                                                             max_episode_steps=1000)
        env_up = time_limit.TimeLimit(minitaur_reactive_env.MinitaurReactiveEnv(render=False,
                                                                             accurate_motor_model_enabled=True,
                                                                             urdf_version='rainbow_dash_v0',
                                                                             include_obs_history=1,
                                                                             include_act_history=0,
                                                                             train_UP=True),
                                                                             max_episode_steps=1000)
    else:
        env_hist = gym.make(args.env)

        if env_hist.env.include_obs_history == 1 and env_hist.env.include_act_history == 0:
            from gym import spaces

            # modify observation space
            env_hist.env.include_obs_history = OSI_hist
            env_hist.env.include_act_history = OSI_hist
            obs_dim_base = env_hist.env.obs_dim
            env_hist.env.obs_dim = env_hist.env.include_obs_history * obs_dim_base
            env_hist.env.obs_dim += len(env_hist.env.control_bounds[0]) * env_hist.env.include_act_history

            high = np.inf * np.ones(env_hist.env.obs_dim)
            low = -high
            env_hist.env.observation_space = spaces.Box(low, high)
            env_hist.observation_space = spaces.Box(low, high)

        env_hist.env.param_manager.activated_param = dyn_params
        env_hist.env.param_manager.controllable_param = dyn_params

        env_up = gym.make(args.env)
        env_up.env.train_UP = True
        env_up.env.param_manager.activated_param = dyn_params
        env_up.env.param_manager.controllable_param = dyn_params
        env_up.env.obs_dim += len(dyn_params)
        env_up.env.UP_noise_level = False
        env_up.env.noisy_input = False
        env_up.env.resample_MP = True

        env_up.env.param_manager.resample_parameters()
        #print(env_up.env.)
        #exit(0)

        high = np.inf * np.ones(env_up.env.obs_dim)
        low = -high
        env_up.env.observation_space = spaces.Box(low, high)
        env_up.observation_space = spaces.Box(low, high)

        if hasattr(env_up.env, 'obs_perm'):
            obpermapp = np.arange(len(env_up.env.obs_perm), len(env_up.env.obs_perm) + len(dyn_params))
            env_up.env.obs_perm = np.concatenate([env_up.env.obs_perm, obpermapp])

    def policy_fn(name, ob_space, ac_space):
        hid_size = 64
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hid_size, num_hid_layers=3)

    def policy_mirror_fn(name, ob_space, ac_space):
        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                            hid_size=64, num_hid_layers=3, observation_permutation=env_up.env.obs_perm,
                            action_permutation=env_up.env.act_perm, soft_mirror=False)

    result = torch.load("/home/tonyyang/Desktop/policy_transfer/policy_transfer/ppo/pytorch_ppo/trained_copy/ppo/UniversalPolicy.pt")
    actor_critic = result[0]
    device = torch.device("cuda:0")
    actor_critic.to(device)
    env = env_up

    '''
    total = 0

    eval_recurrent_hidden_states = torch.zeros(1,
        actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1,1, device=device)

    for iter in range(100):
        obs = env.reset()
        cur_ep_len = 0
        cur_ep_ret = 0
        print('------------- Iter ', iter, ' ----------------')
        step = 0
        while True:
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                torch.tensor( [obs],dtype=torch.float32, device=device ),
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

            obs, reward, done, infos = env.step(action[0].cpu().numpy())
            print(reward)
            
            eval_masks = torch.tensor(
            [[0.0] if done else [1.0]],
            dtype=torch.float32,
            device=device)

            cur_ep_ret += reward
            cur_ep_len += 1
            step+=1
            if done:
                break
        print(cur_ep_ret)
        print(cur_ep_len)
        total = total + cur_ep_ret
    print(total/100)
    '''
    seed = 1000
    num_processes = 1
    eval_log_dir = ""
    eval_envs = make_vec_envs(env, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)
    eval_envs.eval()

    #ob_rms = utils.get_vec_normalize(eval_envs).ob_rms
    ob_rms = result[1]

    #vec_norm = utils.get_vec_normalize(eval_envs)
    #if vec_norm is not None:
    #    vec_norm.eval()
    #    vec_norm.ob_rms = ob_rms
    policy = policy_normalization_remove(actor_critic, ob_rms, device, env)

    #obs = eval_envs.reset()


    t =0
    reward = 0
    for i in range(100):
        done = False
        #obs = eval_envs.reset()
        #eval_recurrent_hidden_states = torch.zeros(
        #    num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        #eval_masks = torch.zeros(num_processes, 1, device=device)
        policy.reset()

        while done == False:
            '''
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            '''
            obs, rew, done, infos = policy.act()
            t = t +1
            reward = rew+reward

        print(reward/(i+1))

    eval_envs.close()
    print(reward/100)
    #print(" Evaluation using {} steps: reward {:.5f}\n".format(
    #    t, reward.cpu().numpy()))