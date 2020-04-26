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
import torch
from pytorch_ppo.a2c_ppo_acktr import Policy

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

        high = np.inf * np.ones(env_up.env.obs_dim)
        low = -high
        env_up.env.observation_space = spaces.Box(low, high)
        env_up.observation_space = spaces.Box(low, high)

        if hasattr(env_up.env, 'obs_perm'):
            obpermapp = np.arange(len(env_up.env.obs_perm), len(env_up.env.obs_perm) + len(dyn_params))
            env_up.env.obs_perm = np.concatenate([env_up.env.obs_perm, obpermapp])
    up_policy = torch.load('policy_transfer/ppo/trained_models/ppo/UniversalPolicy.pt')

    env_up.env.resample_MP = True

    env_up.seed(seed + MPI.COMM_WORLD.Get_rank())

    env_up.reset()

    input_data = []
    output_data = []

    pi = up_policy
    env = env_up
    horizon = 2000
    total = 0

    for iter in range(osi_iteration):
        ob = env.reset()
        cur_ep_len = 0
        cur_ep_ret = 0
        print('------------- Iter ', iter, ' ----------------')
        while True:
            ac = pi.step(ob)
            ob, rew, done, envinfo = env.step(ac)
            if done:
                break

            cur_ep_ret += rew
            cur_ep_len += 1
        print(cur_ep_ret)
        print(cur_ep_len)
        total = total + cur_ep_ret
    print(total/100)



