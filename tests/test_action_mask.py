import pytest
import gym
import numpy as np
import os
import warnings

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import  SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C, ACER, ACKTR
from stable_baselines.common.action_mask_env import DiscreteEnv, MultiDiscreteEnv1, MultiDiscreteEnv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

MODEL_LIST = [PPO2, A2C, ACER, ACKTR]
POLICY_LIST = [MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy]
ENV_LIST = [DiscreteEnv, MultiDiscreteEnv1, MultiDiscreteEnv2]

@pytest.mark.slow
@pytest.mark.parametrize('model_class', MODEL_LIST)
@pytest.mark.parametrize('policy', POLICY_LIST)
@pytest.mark.parametrize('env_class', ENV_LIST)
def test_action_mask_learn_SubprocVecEnv(model_class, policy, env_class):
    env = SubprocVecEnv([lambda: env_class() for i in range(2)])

    model = PPO2(policy, env, verbose=0, nminibatches=2)
    model.learn(total_timesteps = 500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('model_class', MODEL_LIST)
@pytest.mark.parametrize('policy', POLICY_LIST)
@pytest.mark.parametrize('env_class', ENV_LIST)
def test_action_mask_learn_DummyVecEnv(model_class, policy, env_class):
    env = DummyVecEnv([lambda: env_class()])

    model = PPO2(policy, env, verbose=0, nminibatches=1)
    model.learn(total_timesteps = 500)
    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('model_class', MODEL_LIST)
@pytest.mark.parametrize('policy', POLICY_LIST)
@pytest.mark.parametrize('env_class', ENV_LIST)
def test_action_mask_run_SubprocVecEnv(model_class, policy, env_class):
    env = SubprocVecEnv([lambda: env_class() for i in range(2)])

    model = PPO2(policy, env, verbose=0, nminibatches=2)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, reward, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()

@pytest.mark.slow
@pytest.mark.parametrize('model_class', MODEL_LIST)
@pytest.mark.parametrize('policy', POLICY_LIST)
@pytest.mark.parametrize('env_class', ENV_LIST)
def test_action_mask_run_DummyVecEnv(model_class, policy, env_class):
    env = DummyVecEnv([lambda: env_class()])

    model = PPO2(policy, env, verbose=0, nminibatches=1)

    obs, done, action_masks = env.reset(), [False], []
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, reward, done, infos = env.step(action)

        action_masks.clear()
        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks.append(env_action_mask)

    env.close()
