import argparse
import torch

from src.environment import *
from src.memory import *
from src.model import *
from src.agent import *

def get_args():
    parser = argparse.ArgumentParser(
        """A2C implement to playing Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default="best_model.pth", help = 'Pretrained model path')

    parser.add_argument('--num_envs', type=int, default=16, help='Number of environment')
    parser.add_argument('--learn_step', type=int, default=20, help='Number of steps between training model')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help=(
                                                    "epsilon in optimizer, I can complete almost stages with 1e-8 as PyTorch default eps. \n"\
                                                    "But in most RL papers, they recomment use 1e-5 or even larger.\n"\
                                                    "I find that 1e-8 make learn faster but don't sure it better for long training"
                                                    ))
    parser.add_argument('--detach_lstm_state', type=bool, default=True, help="If True, Model just use h, c as inputs. If False, LSTM will backpropagation through time.")
    parser.add_argument('--init_weights', type=bool, default=True, help = 'use _initialize_weights function or not')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--V_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')

    parser.add_argument('--total_step', type=int, default=int(5e6), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')
    parser.add_argument('--save_figure_step', type=int, default=int(1e3), help='Number of steps between testing model')
    parser.add_argument('--total_step_or_episode', type=str, default='step', help='choice stop training base on total step or total episode')
    parser.add_argument('--total_episode', type=int, default=None, help='Total episodes for training')
    

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(1, 84, 84))

    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()
    return args


def test(config):
    model = Model(config.state_dim, config.action_dim, config.detach_lstm_state, config.init_weights)
    agent = Agent(world = config.world, stage = config.stage, action_type = config.action_type, envs = None, num_envs = config.num_envs, 
              state_dim = config.state_dim, action_dim = config.action_dim, save_dir = config.save_dir,
              save_model_step = config.save_model_step, save_figure_step = config.save_figure_step, learn_step = config.learn_step,
              total_step_or_episode = config.total_step_or_episode, total_step = config.total_step, total_episode = config.total_episode,
              model = model, gamma = config.gamma, learning_rate = config.learning_rate, entropy_coef = config.entropy_coef, V_coef = config.V_coef,
              max_grad_norm = config.max_grad_norm, optimizer_eps = config.optimizer_eps,
              device = "cuda" if torch.cuda.is_available() else "cpu")
    agent.load_model(config.pretrained_model)
    agent.save_figure()

if __name__ == "__main__":
    config = get_args()
    test(config)
