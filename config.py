import argparse

parser = argparse.ArgumentParser(description="神经网络参数配置")
parser.add_argument("--gpu_id", default='cpu')
args = parser.parse_args()

class TextCNNConfig(object):
    trainfile = 'data/train'
    testfile = 'data/train'
    d_model = 50
    filter_num = 300
    filter_sizes = [3]
    dropout = 0.5
    lr = 0.001
    batch_size = 50
    epoch = 50
    device = args.gpu_id


