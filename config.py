# 各种参数，方便通过命令行指定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--id', required=True, default='test')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--step', type=int, default=0)
parser.add_argument('--img', type=int, default=580695)

parser.add_argument('--vocab', default='./data/vocab.pkl')
parser.add_argument('--train', default='./data/dataset_train.json')
parser.add_argument('--val', default='./data/dataset_val.json')
parser.add_argument('--test', default='./data/dataset_test.json')
parser.add_argument('--resnet_feat_dir', default='/home/data_ti4_c/chengkz/data/resnet_feat')

parser.add_argument('--save_loss_freq', type=int, default=20)
parser.add_argument('--save_model_freq', type=int, default=5000)
parser.add_argument('--log_dir', default='/home/chenyang/checkpoints/CVAE_Caption/log/{}')

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--fixed_len', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--unk_rate', type=float, default=0.0)
parser.add_argument('--kl_rate', type=float, default=0.05)
parser.add_argument('--sl_rate', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--beam_num', type=int, default=5)
parser.add_argument('--beam_alpha', type=float, default=0.2)
parser.add_argument('--num_samples', type=int, default=20)
parser.add_argument('--test_length', type=int, default=10)

parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--image_dim', type=int, default=2048)

config = parser.parse_args()

