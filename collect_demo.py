import os
import argparse
import torch

from envs.env_norm import make_env
from gail.algo import SACExpert
from gail.utils import collect_demo
from envs import MultimodalEnvs
from gail.utils import set_seed


# 定义run函数，用于运行算法
def run(args):
    # 创建环境
    env = make_env(MultimodalEnvs[args.env_id](num_modes=args.num_modes))

    # Set random seed
    # 设置随机种子
    set_seed(args.seed)
    env.seed(args.seed)

    # 设置权重路径
    weight_dir = args.weight_dir + args.env_id + '_' + str(args.num_modes) + '_modes/' + '{}.pth'.format(args.idx)
    # 设置缓冲区路径
    if args.labeled:
        buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_lb'
    else:
        buffer_dir = args.buffer_dir + args.env_id + '_' + str(args.num_modes) + '_modes_ulb'

    # 创建算法
    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=weight_dir
    )

    # 收集专家数据
    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
        obs_horizon=args.obs_horizon,
        idx=args.idx,
        rend_env=args.rend_env
    )
    # 保存缓冲区
    buffer.save(os.path.join(
        buffer_dir,
        'size{}_{}.pth'.format(args.buffer_size, args.idx)
    ))


if __name__ == '__main__':
    # 创建参数解析器
    p = argparse.ArgumentParser()
    # 添加参数
    p.add_argument('--idx', type=int, default=0)
    p.add_argument('--weight_dir', type=str, default='weights/')
    p.add_argument('--buffer_dir', type=str, default='buffers/')
    p.add_argument('--env_id', type=str, default='Pusher-v4')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--rend_env', type=bool, default=False)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--labeled', type=int, default=0)
    p.add_argument('--num_modes', type=int, default=6)
    p.add_argument('--obs_horizon', type=int, default=8)
    # 解析参数
    args = p.parse_args()
    # 运行算法
    run(args)