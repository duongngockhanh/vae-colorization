import argparse

parser = argparse.ArgumentParser(description="PyTorch Diverse Colorization")
parser.add_argument("dataset_key", help="Dataset")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device id")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("-b", "--batchsize", type=int, default=32, help="Batch size")
parser.add_argument("-z", "--hiddensize", type=int, default=64, help="Latent vector dimension")
parser.add_argument("-n", "--nthreads", type=int, default=4, help="Data loader threads")
parser.add_argument("-em", "--epochs_mdn", type=int, default=1, help="Number of epochs for MDN")
parser.add_argument("-m", "--nmix", type=int, default=8, help="Number of diverse colorization (or output gmm components)")
parser.add_argument('-lg', '--logstep', type=int, default=100, help='Interval to log data')
args = parser.parse_args()