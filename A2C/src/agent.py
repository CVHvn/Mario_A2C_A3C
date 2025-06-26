from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.memory import *
from src.model import *

class Agent():
    def __init__(self, world, stage, action_type, envs, num_envs, state_dim, action_dim, save_dir, save_model_step,
                 save_figure_step, learn_step, total_step_or_episode, total_step, total_episode, model,
                 gamma, learning_rate, entropy_coef, V_coef, max_grad_norm, device):
        self.world = world
        self.stage = stage
        self.action_type = action_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.envs = envs
        self.model = model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.V_coef = V_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.max_grad_norm = max_grad_norm

        self.memory = Memory(self.num_envs)

        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9

        # I just log 1000 lastest update and print it to log.
        self.V_loss = np.zeros((1000,)).reshape(-1)
        self.P_loss = np.zeros((1000,)).reshape(-1)
        self.E_loss = np.zeros((1000,)).reshape(-1)
        self.total_loss = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

    def save_figure(self, is_training=False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, True)
        state = self.env.reset()
        done = False

        images = []
        total_reward = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        # create h, c as zeros
        h = torch.zeros((1, 512), dtype=torch.float, device = self.device)
        c = torch.zeros((1, 512), dtype=torch.float, device = self.device)

        episode_time = datetime.now()

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                logit, V, h, c = self.model(torch.tensor(state, dtype = torch.float, device = self.device).unsqueeze(0), h, c)
            action = logit.argmax(-1).item()
            next_state, reward, done, trunc, info = self.env.step(action)
            state = next_state
            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_reward += reward
            total_step += 1

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_reward} episode_step: {total_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_reward > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_reward
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")

        # if model can complete this game, stop training by set self.is_completed to True
        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = f"model_{self.current_step}.pth"
        self.model.load_state_dict(torch.load(model_path))

    def select_action(self, states, h, c):
        # select action when training, we need use Categorical distribution to make action base on probability from model
        states = torch.tensor(np.array(states), device = self.device)

        logits, V, h, c = self.model(states, h, c)
        with torch.no_grad():
            policy = F.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(policy)
            actions = distribution.sample().cpu().numpy().tolist()
        return actions, logits, V, h, c

    def update_loss_statis(self, loss_p, loss_v, loss_e, loss):
        # update loss for logging, just save 1000 latest updates.
        self.V_loss[self.loss_index] = loss_v
        self.P_loss[self.loss_index] = loss_p
        self.E_loss[self.loss_index] = loss_e
        self.total_loss[self.loss_index] = loss
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def learn(self, h, c):
        # reset optimizer
        self.optimizer.zero_grad()

        # get all data
        states, actions, next_states, rewards, dones, logits, values = self.memory.get_data()

        # calculate target (td lambda target) and gae advantages
        targets = []
        with torch.no_grad():
            _, next_value, h, c = self.model(torch.tensor(np.array(next_states[-1]), device = self.device), h, c)
        target = next_value
        advantage = 0

        for state, next_state, reward, done, V in zip(states[::-1], next_states[::-1], rewards[::-1], dones[::-1], values[::-1]):
            done = torch.tensor(done, device = self.device, dtype = torch.float).reshape(-1, 1)
            reward = torch.tensor(reward, device = self.device).reshape(-1, 1)

            target = next_value * self.gamma * (1-done) + reward
            advantage = target + self.gamma * advantage * (1-done)
            targets.append(advantage)
            advantage = advantage - V.detach()
            next_value = V.detach()
        targets = targets[::-1]

        # convert all data to tensor
        values = torch.cat(values, 0)
        targets = torch.cat(targets, 0).view(-1, 1)
        logits = torch.cat(logits, 0)
        probs = torch.softmax(logits, -1)
        advantages = (targets - values).reshape(-1)

        # calculate loss
        entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()
        loss_V = F.smooth_l1_loss(values, targets)

        index = torch.arange(0, len(probs), device = self.device)
        actions = torch.flatten(torch.tensor(actions, device = self.device, dtype = torch.int64))
        loss_P = -((probs[index, actions] + 1e-9).log() * advantages.detach()).mean()

        loss = - entropy * self.entropy_coef + loss_V * self.V_coef + loss_P
        loss.backward()

        # norm gradient and update agent
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        self.update_loss_statis(loss_P.item(), loss_V.item(), entropy.item(), loss.item())

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        # reset envs
        states = self.envs.reset()

        # create h, c as zeros
        h = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device)
        c = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device)

        while True:
            # finish training if agent reach total_step or total_episode based on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            actions, logit, V, h, c = self.select_action(states, h, c)

            next_states, rewards, dones, truncs, infos = self.envs.step(actions)

            # save to memory
            self.memory.save(states, actions, rewards, next_states, dones, logit, V)

            episode_reward = [x + reward for x, reward in zip(episode_reward, rewards)]
            episode_step = [x+1 for x in episode_step]

            # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} complete: {infos[i]["flag_get"]==True} mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} loss: {(self.total_loss.sum()/self.len_loss):.4f} episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

            # reset h and c to zeros for enviroments that just ending episode, just multiply h and c with (1-dones)
            h = h * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            c = c * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

            # training agent every learn_step
            if self.current_step % self.learn_step == 0:
                self.learn(h, c)
                self.memory.reset()

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0 and self.save_figure_step != -1:
                self.save_figure(is_training=True)
                if self.is_completed:
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    return

            states = list(next_states)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
        f_out.close()