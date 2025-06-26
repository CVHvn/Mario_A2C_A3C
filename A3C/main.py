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


def learn(config):
    shared_model = Model(config.state_dim, config.action_dim)
    shared_model.train()
    shared_model.share_memory()
    optimizer = SharedRMSprop(shared_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.share_memory()

    is_completed = mp.Value("b", False)

    # I just log 1000 lastest update and print it to log.
    V_loss = torch.zeros((1000,)).reshape(-1)
    V_loss.share_memory_()
    P_loss = torch.zeros((1000,)).reshape(-1)
    P_loss.share_memory_()
    E_loss = torch.zeros((1000,)).reshape(-1)
    E_loss.share_memory_()
    total_loss = torch.zeros((1000,)).reshape(-1)
    total_loss.share_memory_()
    loss_index = mp.Value('i', 0)
    len_loss = mp.Value('i', 0)

    current_step = mp.Value('i', 1)
    current_episode = mp.Value('i', 1)
    max_episode_reward = mp.Value("f", 1)
    max_episode_step = mp.Value("f", 1)

    max_test_score = mp.Value("f", 0)

    shared_file_lock = mp.Lock()

    episode_rewards = mp.Manager().list()
    episode_steps = mp.Manager().list()
    episode_2_env_step = mp.Manager().list()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processes = []
    for rank in range(0, config.num_envs):
        p = mp.Process(
            target=worker, args=(
                rank, config.world, config.stage, config.action_type, config.additional_bonus_state_8_4_option,
                shared_model, optimizer, current_step, max_episode_reward, max_episode_step,
                current_episode, config.state_dim, config.action_dim, device, config.total_step, 
                config.save_model_step, config.learn_step,
                V_loss, P_loss, E_loss, total_loss, loss_index, len_loss, shared_file_lock,
                episode_rewards, episode_steps, episode_2_env_step, config.max_grad_norm, 
                config.gamma, config.entropy_coef, config.V_coef, is_completed))
        p.start()
        processes.append(p)
        time.sleep(0.001)

    p = mp.Process(
            target=test_Worker, args=(
                config.world, config.stage, config.action_type, config.additional_bonus_state_8_4_option, device,
                shared_model, config.state_dim, config.action_dim, current_step, config.total_step, True, is_completed,
                P_loss, V_loss, E_loss, total_loss, len_loss, max_test_score)
            )
    p.start()
    processes.append(p)
    time.sleep(0.001)

    for p in processes:
        time.sleep(0.001)
        p.join()

    print("done")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    config = get_args()
    learn(config)