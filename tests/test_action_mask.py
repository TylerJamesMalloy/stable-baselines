import pytest

from stable_baselines import A2C, ACER, ACKTR, PPO1, PPO2, TRPO
from stable_baselines.common.action_mask_env import DummyActionMaskEnvDiscrete, DummyActionMaskEnvMutliDiscrete
from stable_baselines.common.vec_env import DummyVecEnv

MODEL_LIST = [A2C, ACER, ACKTR, PPO1, PPO2, TRPO]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_action_mask_discrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: DummyActionMaskEnvDiscrete()])

    model = model_class("MlpPolicy", env)

    model.learn(total_timesteps=1000, seed=0)


MODEL_LIST = [A2C, PPO2]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_action_mask_multi_discrete(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: DummyActionMaskEnvMutliDiscrete()])

    model = model_class("MlpPolicy", env)

    model.learn(total_timesteps=1000, seed=0)

