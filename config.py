import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='res', type=str)
parser.add_argument('--test_mode', default=0, type=int, choices=list(range(10)))
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--train_dataset', default='cifar10', type=str)
parser.add_argument('--n_epoch', default=30, type=int)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--test_batch_size', default=10, type=int)
parser.add_argument('--lambd', default=0.0001, type=float)
parser.add_argument('--noise_dev', default=20.0, type=float)
parser.add_argument('--Linfinity', default=0.03, type=float)
parser.add_argument('--binary_threshold', default=0.5, type=float)
parser.add_argument('--lr_mode', default=0, type=int)
parser.add_argument('--test_interval', default=1000, type=int)
parser.add_argument('--save_model', default='res_cifar10', type=str)

# attack
parser.add_argument("--attack_method", default="PGD", choices=["PGD", "FGSM", "Momentum", "STA"])
parser.add_argument("--epsilon", type=float, default=8 / 255)

# dataset
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

# net
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--num_classes', default=10, type=int)

args = parser.parse_args()