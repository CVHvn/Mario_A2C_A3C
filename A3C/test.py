import argparse
import time
import torch
import torch.multiprocessing as mp

from src.environment import *
from src.memory import *
from src.model import *
from src.optimizer import *
from src.worker import *
from src.test_worker import *

def get_args():
    parser = argparse.ArgumentParser(
        """A2C implement to playing Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environment')
    parser.add_argument('--learn_step', type=int, default=20, help='Number of steps between training model')

    parser.add_argument('--learning_rate', type=float, default=7e-5)
    parser.add_argument('--weight_decay', type=float, default = 0)
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor for rewards')
    parser.add_argument('--V_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=40, help='Max gradient norm')

    parser.add_argument('--total_step', type=int, default=int(5e6), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(1, 84, 84))
    parser.add_argument("--additional_bonus_state_8_4_option", type=str, default='no', help='if you want bonus reward for stage 8-4, set it to "right_pipe"')

    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()
    return args

def test(config):
    env = create_env(config.world, config.stage, config.action_type, config.additional_bonus_state_8_4_option, test=True)
    model = Model(config.state_dim, config.action_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.load_state_dict(torch.load(config.save_dir))
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

    imageio.mimsave('test_episode.mp4', images)

if __name__ == "__main__":
    config = get_args()
    test(config)