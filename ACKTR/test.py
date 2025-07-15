import argparse
import torch

from src.environment import *
from src.memory import *
from src.model import *
from src.agent import *
from src.KFAC import *

def get_args():
    parser = argparse.ArgumentParser(
        """A2C implement to playing Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default="best_model.pth", help = 'Pretrained model path')

    parser.add_argument('--num_envs', type=int, default=32, help='Number of environment')
    parser.add_argument('--learn_step', type=int, default=20, help='Number of steps between training model')
    parser.add_argument('--init_weights', type=bool, default=True, help = 'use _initialize_weights function or not')

    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--V_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy loss coefficient')

    parser.add_argument('--total_step', type=int, default=int(5e6), help='Total step for training')
    parser.add_argument('--save_model_step', type=int, default=int(1e5), help='Number of steps between saving model')
    parser.add_argument('--save_figure_step', type=int, default=int(1e3), help='Number of steps between testing model')
    parser.add_argument('--total_step_or_episode', type=str, default='step', help='choice stop training base on total step or total episode')
    parser.add_argument('--total_episode', type=int, default=None, help='Total episodes for training')

    parser.add_argument("--action_dim", type=int, default=12, help='12 if set action_type to complex else 7')
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--state_dim", type=tuple, default=(4, 84, 84))

    parser.add_argument('--kfac_momentum', type=float, default=0.9, help='momentum in KFAC optimizer')
    parser.add_argument('--stat_decay', type=float, default=0.99, help='stat_decay in KFAC optimizer')
    parser.add_argument('--kfac_Ts', type=int, default=1)
    parser.add_argument('--kfac_Tf', type=int, default=10)
    parser.add_argument('--kfac_kl_clip', type=float, default=0.001)
    parser.add_argument('--kfac_damping', type=float, default=1e-2)
    parser.add_argument('--kfac_fast_cnn', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--eigen_eps', type=float, default=1e-4, help=f'This parameter can distort the gradient, '
                                                                        f'but it helps the algorithm not to fail when calculating the eigenvalues. '
                                                                        f'Reducing this parameter may make the agent better (no guarantees).')

    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()
    return args


def test(config):
    model = Model(config.state_dim, config.action_dim, config.init_weights)
    agent = Agent(world = config.world, stage = config.stage, action_type = config.action_type, envs = None, num_envs = config.num_envs,
                  state_dim = config.state_dim, action_dim = config.action_dim, save_dir = config.save_dir,
                  save_model_step = config.save_model_step, save_figure_step = config.save_figure_step, learn_step = config.learn_step,
                  total_step_or_episode = config.total_step_or_episode, total_step = config.total_step, total_episode = config.total_episode,
                  model = model, gamma = config.gamma, learning_rate = config.learning_rate, entropy_coef = config.entropy_coef, 
                  V_coef = config.V_coef, kfac_momentum = config.kfac_momentum, stat_decay = config.stat_decay, kfac_kl_clip = config.kfac_kl_clip, 
                  kfac_damping = config.kfac_damping, weight_decay = config.weight_decay, kfac_fast_cnn = config.kfac_fast_cnn, 
                  kfac_Ts = config.kfac_Ts, kfac_Tf = config.kfac_Tf, gae_lambda = config.gae_lambda,
                  eigen_eps = config.eigen_eps, device = "cuda" if torch.cuda.is_available() else "cpu")
    agent.load_model(config.pretrained_model)
    agent.save_figure()

if __name__ == "__main__":
    config = get_args()
    test(config)
