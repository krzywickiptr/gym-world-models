import statistics
from functools import reduce
from math import cos, sin
import random

import gym
import numpy
import torch.nn
import tqdm as tqdm

import matplotlib.pyplot as plt

import cv2
from utils import mlp


class RandomAgent:
    def __init__(self, action_space, stickiness):
        self.stickiness = stickiness
        self.action_space = action_space
        self.recent_action = None

    def act(self, obs):
        if self.recent_action and random.random() < self.stickiness:
            return self.recent_action

        self.recent_action = self.action_space.sample()
        return self.recent_action


CARTPOLE = 'CartPole-v0'
ACROBOT = 'Acrobot-v1'
MOUNTAIN_CAR = 'MountainCar-v0'
PENDULUM = 'Pendulum-v0'


env_name = CARTPOLE
num_episodes = 8000
batch_size = 32
training_epochs = 4
demonstration_epochs = 128
cum_eval_epochs = 4096

# env_name = ACROBOT
# num_episodes = 1024
# batch_size = 32
# training_epochs = 1
# demonstration_epochs = 16
# cum_eval_epochs = 512

# env_name = MOUNTAIN_CAR
# num_episodes = 4000
# batch_size = 32
# training_epochs = 2
# demonstration_epochs = 64
# cum_eval_epochs = 2048

frameSize = (600, 400) if env_name == CARTPOLE else\
            (500, 500) if env_name == ACROBOT else\
            (600, 400) if env_name == MOUNTAIN_CAR else\
            (500, 500) if env_name == PENDULUM else None

fourcc = cv2.VideoWriter_fourcc(*'MP42')

out1 = cv2.VideoWriter(f'{env_name}_1.avi', fourcc, 24, frameSize)
out2 = cv2.VideoWriter(f'{env_name}_2.avi', fourcc, 24, frameSize)
out3 = cv2.VideoWriter(f'{env_name}_3.avi', fourcc, 24, frameSize)

# https://stackoverflow.com/questions/47664112/is-there-a-way-to-disable-video-rendering-in-openai-gym-while-still-recording-it
def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


if __name__ == '__main__':
    disable_view_window()

    eval_env1 = gym.make(env_name)
    eval_env2 = gym.make(env_name)
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    obs_size = reduce(lambda x, y: x * y, obs_dim)

    step_model = mlp([obs_size + 1, 64, 64, obs_size], torch.nn.ReLU)
    step_model_optimizer = torch.optim.Adam(step_model.parameters())

    agent = RandomAgent(env.action_space, stickiness=0.)

    replay_buffer = []
    losses = []
    diffs = []

    for i in tqdm.tqdm(range(num_episodes)):
        o = env.reset()
        while True:
            action = agent.act(o)
            oprim, reward, done, _ = env.step(action)

            replay_buffer.append([o.copy(), oprim.copy(), action, reward, done])

            if done:
                break
            o = oprim

        if i == num_episodes - 1:
            for _ in tqdm.tqdm(range(training_epochs)):
                random.shuffle(replay_buffer)
                o = torch.stack([torch.tensor(x[0]) for x in replay_buffer])
                oprim = torch.stack([torch.tensor(x[1]) for x in replay_buffer])
                action = torch.stack([torch.tensor(x[2]) for x in replay_buffer])
                for (o, oprim, action) in tqdm.tqdm(zip(
                    torch.split(o, batch_size),
                    torch.split(oprim, batch_size),
                    torch.split(action, batch_size)
                )):
                    with torch.no_grad():
                        o = o.flatten(start_dim=1).float()
                        oprim = oprim.flatten(start_dim=1).float()
                        if env_name != PENDULUM:
                            obs_action = torch.cat([o, action.unsqueeze(dim=1)], dim=1).float()
                        else:
                            obs_action = torch.cat([o, action], dim=1).float()

                    step_model_optimizer.zero_grad()
                    state_reconstruction_loss = torch.nn.MSELoss()(o + step_model(obs_action), oprim)
                    state_reconstruction_loss.backward()
                    step_model_optimizer.step()

            with torch.no_grad():
                for j in tqdm.tqdm(range(cum_eval_epochs)):
                    losses.append([])
                    diffs.append([])
                    eval_env1.done = eval_env2.done = True
                    o2 = eval_env2.reset()
                    eval_env1.reset()
                    o1 = eval_env1.state = o2.copy()

                    while True:
                        action = agent.act(o1)
                        o1 = torch.tensor(o1, dtype=torch.float32).flatten()

                        if env_name != PENDULUM:
                            obs_action = torch.cat([o1, torch.tensor([action])])
                        else:
                            obs_action = torch.cat([o1, torch.tensor(action)])

                        oprim = o1 + step_model(obs_action)
                        state = oprim.clone().detach().numpy()
                        _, _, done1, _ = eval_env1.step(action)
                        eval_env1.state = state
                        o2prim, _, done2, _ = eval_env2.step(action)

                        s = eval_env2.state
                        if env_name == ACROBOT:
                            s = numpy.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

                        if env_name == PENDULUM:
                            theta, thetadot = s
                            s = numpy.array([numpy.cos(theta), numpy.sin(theta), thetadot])

                        losses[-1].append(numpy.abs(state - s).mean())
                        diffs[-1].append(numpy.abs(o2 - o2prim).mean())

                        if done2:
                            break

                        if j < demonstration_epochs:
                            img1 = eval_env1.render(mode='rgb_array')
                            img2 = eval_env2.render(mode='rgb_array')

                            out1.write(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                            out2.write(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                            out3.write(cv2.cvtColor(img1 - img2, cv2.COLOR_RGB2BGR))

                        o1 = oprim.clone().detach().numpy()
                        o2 = o2prim

    def pointwise_mean_std(trajs, cutoff=0.8):
        max_traj_length = int(max([len(traj) for traj in trajs]) * cutoff)
        mean = [0] * max_traj_length
        stds = [0] * max_traj_length
        counts = [0] * max_traj_length

        for traj in trajs:
            for i, loss in enumerate(traj[:max_traj_length]):
                mean[i] += loss
                stds[i] += loss ** 2
                counts[i] += 1

        for i in range(max_traj_length):
            mean[i] /= counts[i]
            stds[i] /= counts[i]
            stds[i] -= mean[i] ** 2

        return mean, stds

    mean, stds = pointwise_mean_std(losses)
    diff_mean, diff_stds = pointwise_mean_std(diffs)

    common = min(len(mean), len(diff_mean))

    plt.errorbar(numpy.arange(0, common), numpy.array(mean), yerr=numpy.array(stds), label='model error')
    plt.errorbar(numpy.arange(0, common), numpy.array(diff_mean), yerr=numpy.array(diff_stds), label='env state diff')
    plt.suptitle(f'{env_name} – model error vs number of steps')
    plt.legend()
    plt.savefig(f'{env_name}.png')

    env.close()
    out1.release()
    out2.release()
    out3.release()
    eval_env1.close()
    eval_env2.close()
