
from scipy import sparse as sp
from trans_model import trans_model as model
import argparse
import pickle as pkl

DATASET = 'citeseer'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help='learning rate for supervised loss', type=float, default=0.1)
parser.add_argument('--embedding_size', help='embedding dimensions', type=int, default=50)
parser.add_argument('--window_size', help='window size in random walk sequences', type=int, default=3)
parser.add_argument('--path_size', help='length of random walk sequences', type=int, default=10)
parser.add_argument('--batch_size', help='batch size for supervised loss', type=int, default=200)
parser.add_argument('--g_batch_size', help='batch size for graph context loss', type=int, default=200)
parser.add_argument('--g_sample_size', help='batch size for label context loss', type=int, default=100)
parser.add_argument('--neg_samp', help='negative sampling rate; zero means using softmax', type=int, default=0)
parser.add_argument('--g_learning_rate', help='learning rate for unsupervised loss', type=float, default=1e-2)
parser.add_argument('--model_file', help='filename for saving models', type=str, default='trans.model')
parser.add_argument('--use_feature', help='whether use input features', type=bool, default=True)
parser.add_argument('--update_emb', help='whether update embedding when optimizing supervised loss',
                    type=bool, default=True)
parser.add_argument('--layer_loss', help='whether incur loss on hidden layers', type=bool, default=True)
args = parser.parse_args()


def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis=1) == np.argmax(ty, axis=1)).sum() * 1.0 / tpy.shape[0]


# load the data: x, y, tx, ty, graph
NAMES = ['x', 'y', 'tx', 'ty', 'graph']
objects = {}
for name in NAMES:
    data = pkl.load(open("data/trans.{}.{}".format(DATASET, name), 'rb'), encoding='latin1')
    objects[name] = data
#     objects.append(cPickle.load(open("data/trans.{}.{}".format(DATASET, name))))
# x, y, tx, ty, graph = tuple(objects)

# initialize the model
m = model(args)
# add data
m.add_data(objects['x'], objects['y'], objects['graph'])
# build the model
m.build()
# pre-training
m.init_train(init_iter_label=2000, init_iter_graph=70)
iter_cnt, max_accu = 0, 0
for _ in range(10000):
    # perform a training step
    m.step_train(max_iter=1, iter_graph=0, iter_inst=1, iter_label=0)
    # predict the dev set
    tpy = m.predict(objects['tx'])
    # compute the accuracy on the dev set
    accu = comp_accu(tpy, objects['ty'])
    print(iter_cnt, accu, max_accu)
    iter_cnt += 1
    if accu > max_accu:
        # store the model if better result is
        m.store_params()                                                        
        max_accu = max(max_accu, accu)
