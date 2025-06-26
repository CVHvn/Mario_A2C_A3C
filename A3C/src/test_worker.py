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

def test_Worker(world, stage, action_type, additional_bonus_state_8_4_option, device,
                shared_model, state_dim, num_action, shared_current_step, total_steps, is_training, is_completed,
                P_loss, V_loss, E_loss, total_loss, len_loss, max_test_score):
        env = create_env(world, stage, action_type, additional_bonus_state_8_4_option, test=True)
        model = Model(state_dim, num_action)
        model.to(device)
        last_test_step = 0.

        while shared_current_step.value < total_steps and is_completed.value == False:
            if shared_current_step.value < last_test_step + 100:
                continue
            last_test_step = shared_current_step.value
            episode_time = datetime.now()

            model.train()
            with torch.no_grad():
                model.load_state_dict(shared_model.state_dict())
            model.eval()

            images = []
            total_reward = 0
            total_step = 0
            num_repeat_action = 0
            old_action = -1

            state = env.reset()
            done = False

            # create h, c as zeros
            h = torch.zeros((1, 512), dtype=torch.float, device = device)
            c = torch.zeros((1, 512), dtype=torch.float, device = device)

            # play 1 episode, just get loop action with max probability from model until the episode end.
            while not done:
                with torch.no_grad():
                    logit, V, h, c = model(torch.tensor(state, dtype = torch.float, device = device).unsqueeze(0), h, c)
                action = logit.argmax(-1).item()
                next_state, reward, done, trunc, info = env.step(action)
                state = next_state
                img = Image.fromarray(env.current_state)
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
                f_out.write(f'episode_reward: {total_reward:.4f} episode_step: {total_step} max_test_reward: {max_test_score.value:.4f} '
                            f'current_step: {last_test_step} loss_p: {(P_loss.sum()/len_loss.value):.4f} '
                            f'loss_v: {(V_loss.sum()/len_loss.value):.4f} loss_e: {(E_loss.sum()/len_loss.value):.4f} '
                            f'loss: {(total_loss.sum()/len_loss.value):.4f} episode_time: {datetime.now() - episode_time}\n')
                f_out.close()

            if total_reward > max_test_score.value or info['flag_get']:
                imageio.mimsave('test_episode.mp4', images)
                max_test_score.value = total_reward
                if is_training:
                    torch.save(model.state_dict(), f"best_model.pth")

            # if model can complete this game, stop training by set is_completed to True
            if info['flag_get']:
                is_completed.value = True

            if not is_training:
                break