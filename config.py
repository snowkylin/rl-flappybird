import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='synthesis',
                    help='predict or synthesis')
parser.add_argument('--observe', type=int, default=10000,
                    help='timesteps to observe before training')
parser.add_argument('--explore', type=int, default=3000000,
                    help='frames over which to anneal epsilon')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='decay rate of past observations')
parser.add_argument('--actions', type=int, default=2,
                    help='number of valid actions')
parser.add_argument('--initial_epsilon', type=float, default=0.1,
                    help='starting value of epsilon')
parser.add_argument('--final_epsilon', type=float, default=0.0001,
                    help='final value of epsilon')
parser.add_argument('--frame_per_action', type=int, default=1,
                    help='')
parser.add_argument('--replay_memory', type=int, default=50000,
                    help='')
parser.add_argument('--resize_width', type=int, default=80,
                    help='')
parser.add_argument('--resize_height', type=int, default=80,
                    help='')
parser.add_argument('--frames', type=int, default=4,
                    help='')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--learning_rate', type=float, default=0.000001,
                    help='learning rate')
args = parser.parse_args()