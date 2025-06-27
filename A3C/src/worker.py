from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import time
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

def sample_action(model, states, h, c, device):
    states = torch.tensor(np.array(states), device = device).unsqueeze(0)
    logits, V, h, c = model(states, h, c)

    with torch.no_grad():
        policy = F.softmax(logits, dim=1)
        distribution = torch.distributions.Categorical(policy)
        action = distribution.sample().cpu().numpy()[0]
    return action, logits, V, h, c

def update_loss_statistic(loss_p, loss_v, loss_e, loss,
                          V_loss, P_loss, E_loss, total_loss, loss_index, len_loss):
    # update loss for logging, just save 1000 latest updates.
    V_loss[loss_index.value] = loss_v
    P_loss[loss_index.value] = loss_p
    E_loss[loss_index.value] = loss_e
    total_loss[loss_index.value] = loss
    loss_index.value = (loss_index.value + 1)%1000
    len_loss.value = min(len_loss.value+1, 1000)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        else:
            shared_param._grad = param.grad.cpu()

def train(model, shared_model, memory, optimizer, h, c, gamma, entropy_coef, V_coef, max_grad_norm, device,
          V_loss, P_loss, E_loss, total_loss, loss_index, len_loss):

    # get all data
    states, actions, next_states, rewards, dones, logits, values = memory.get_data()

    # calculate target (td lambda target) and gae advantages
    targets = []
    with torch.no_grad():
        _, next_value, h, c = model(torch.tensor(np.array(next_states[-1]), device = device).unsqueeze(0), h, c)
    target = next_value
    advantage = 0

    for reward, done, V in zip(rewards[::-1], dones[::-1], values[::-1]):
        done = torch.tensor(done, device = device, dtype = torch.float).reshape(-1)
        reward = torch.tensor(reward, device = device).reshape(-1)

        target = next_value * gamma * (1-done) + reward
        advantage = target + gamma * advantage * (1-done)
        targets.append(advantage)
        advantage = advantage - V.detach()
        next_value = V.detach()
    targets = targets[::-1]

    # convert all data to tensor
    values = torch.cat(values, 0)
    targets = torch.cat(targets, 0).view(-1, 1)
    advantages = (targets - values).reshape(-1)
    logits = torch.cat(logits, 0)
    probs = torch.softmax(logits, -1)

    # calculate loss
    entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()
    loss_V = F.smooth_l1_loss(values, targets)

    index = torch.arange(0, len(probs), device = device)
    actions = torch.flatten(torch.tensor(actions, device = device, dtype = torch.int64))
    loss_P = -((probs[index, actions] + 1e-9).log() * advantages.detach()).mean()

    loss = - entropy * entropy_coef + loss_V * V_coef + loss_P
    model.zero_grad()
    loss.backward()
    optimizer.zero_grad()
    ensure_shared_grads(model, shared_model)
    torch.nn.utils.clip_grad_norm_(shared_model.parameters(), max_grad_norm)
    optimizer.step()

    update_loss_statistic(loss_P.item(), loss_V.item(), entropy.item(), loss.item(),
                          V_loss, P_loss, E_loss, total_loss, loss_index, len_loss)

def worker(worker_id, world, stage, action_type, additional_bonus_state_8_4_option,
                 shared_model, optimizer, shared_current_step, max_episode_reward, max_episode_step,
                 current_episode, state_dim, num_action, device, total_steps, save_frequency, learn_step,
                 V_loss, P_loss, E_loss, total_loss, loss_index, len_loss, shared_file_lock,
                 episode_rewards, episode_steps, episode_2_env_step, max_grad_norm, 
                 gamma, entropy_coef, V_coef, is_completed):
        model = Model(state_dim, num_action)
        model.to(device)
        model.train()
        with torch.no_grad():
            model.load_state_dict(shared_model.state_dict())

        env = create_env(world, stage, action_type, additional_bonus_state_8_4_option, test=False)

        memory = Memory()

        current_step = 0
        episode_reward, episode_step = 0, 0
        state = env.reset()
        start_time = datetime.now()
        episode_time = datetime.now()

        h = torch.zeros((1, 512), dtype=torch.float, device = device)
        c = torch.zeros((1, 512), dtype=torch.float, device = device)

        while shared_current_step.value < total_steps and is_completed.value == False:
            h, c = h.detach(), c.detach()

            with torch.no_grad():
                model.load_state_dict(shared_model.state_dict())

            if shared_current_step.value % save_frequency == 0:
                torch.save(shared_model.state_dict(), f"{shared_current_step.value}.pth")

            for _ in range(learn_step):
                current_step += 1
                with shared_current_step.get_lock():
                    shared_current_step.value = max(current_step, shared_current_step.value)
                action, logit, V, h, c = sample_action(model, state, h, c, device)
                next_state, reward, done, trunc, info = env.step(action)
                episode_reward += reward
                episode_step += 1

                memory.save(state, action, reward, next_state, done, logit, V)

                if done:
                    next_state = env.reset()
                    h = torch.zeros((1, 512), dtype=torch.float, device = device)
                    c = torch.zeros((1, 512), dtype=torch.float, device = device)

                    episode_rewards.append(episode_reward)
                    episode_steps.append(episode_step)
                    episode_2_env_step.append(shared_current_step.value)

                    with current_episode.get_lock():
                        current_episode.value += 1
                    with max_episode_reward.get_lock():
                        max_episode_reward.value = max(max_episode_reward.value, episode_reward)
                    with max_episode_step.get_lock():
                        max_episode_step.value = max(max_episode_step.value, episode_step)

                    with shared_file_lock:
                        f_out = open(f"logging.txt", "a")
                        f_out.write(f'episode: {current_episode.value} agent: {worker_id} rewards: {episode_reward:.4f} '
                                    f'steps: {episode_step} complete: {info["flag_get"]==True} '
                                    f'mean_rewards: {np.array(episode_rewards[-min(len(episode_rewards), 100):]).mean():.4f} '
                                    f'max_rewards: {max_episode_reward.value:.4f} max_steps: {max_episode_step.value} current_step: {shared_current_step.value} '
                                    f'loss_p: {(P_loss.sum()/len_loss.value):.4f} loss_v: {(V_loss.sum()/len_loss.value):.4f} '
                                    f'loss_e: {(E_loss.sum()/len_loss.value):.4f} loss: {(total_loss.sum()/len_loss.value):.4f} '
                                    f'episode_time: {datetime.now() - episode_time} total_time: {datetime.now() - start_time}\n')
                        f_out.close()
                    episode_reward = 0
                    episode_step = 0
                    episode_time = datetime.now()

                state = next_state

            train(model, shared_model, memory, optimizer, h, c, gamma, entropy_coef, V_coef, max_grad_norm, device,
                  V_loss, P_loss, E_loss, total_loss, loss_index, len_loss) 
            memory.reset()