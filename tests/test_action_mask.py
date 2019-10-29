import pytest
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, ACKTR
from stable_baselines.common.action_mask_env import DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv,\
    MultiDiscreteUnbalancedActionMaskEnv

VEC_ENVS = [DummyVecEnv, SubprocVecEnv]
POLICIES = ["MlpPolicy", "MlpLstmPolicy"]
ENVS = [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv, MultiDiscreteUnbalancedActionMaskEnv]


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', ENVS)
def test_action_mask_learn_ppo2(vec_env, policy, env_class):
    env = vec_env([env_class]*4)

    model = PPO2(policy, env, verbose=0, nminibatches=2)
    model.learn(total_timesteps=1000)
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', ENVS)
def test_action_mask_run_ppo2(vec_env, policy, env_class):
    env = vec_env([env_class])

    model = PPO2(policy, env, verbose=0, nminibatches=1)

    obs, done, action_masks = env.reset(), [False], None
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks = np.expand_dims(np.asarray(env_action_mask), axis=0)

    env.close()


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', ENVS)
def test_action_mask_learn_a2c(vec_env, policy, env_class):
    env = vec_env([env_class] * 4)

    model = A2C(policy, env, verbose=0)
    model.learn(total_timesteps=1000)
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', ENVS)
def test_action_mask_run_a2c(vec_env, policy, env_class):
    env = vec_env([env_class])

    model = A2C(policy, env, verbose=0)

    obs, done, action_masks = env.reset(), [False], None
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks = np.expand_dims(np.asarray(env_action_mask), axis=0)

    env.close()


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv])
def test_action_mask_learn_acktr(vec_env, policy, env_class):
    env = vec_env([env_class] * 4)

    model = ACKTR(policy, env, verbose=0)
    model.learn(total_timesteps=1000)
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize('vec_env', VEC_ENVS)
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('env_class', [DiscreteActionMaskEnv, MultiDiscreteActionMaskEnv])
def test_action_mask_run_acktr(vec_env, policy, env_class):
    env = vec_env([env_class])

    model = ACKTR(policy, env, verbose=0)

    obs, done, action_masks = env.reset(), [False], None
    while not done[0]:
        action, _states = model.predict(obs, action_mask=action_masks)
        obs, _, done, infos = env.step(action)

        for info in infos:
            env_action_mask = info.get('action_mask')
            action_masks = np.expand_dims(np.asarray(env_action_mask), axis=0)

    env.close()
