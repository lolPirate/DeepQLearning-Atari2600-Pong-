import gym
from RepeatActionAndMaxFrame import RepeatActionAndMaxFrame
from PreprocessFrame import PreprocessFrame
from StackFrames import StackFrames


def make_env(
    env_name,
    shape=(84, 84, 1),
    repeat=4,
    clip_rewards=False,
    no_ops=False,
    fire_first=False,
):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat)
    return env


#env = make_env(env_name="Pong-v0")

