import os
import sys
import pickle
import tqdm
import numpy as np
import h5py

from argparse import ArgumentParser

import tensorflow as tf
import lib
import lib.task.seq2seq.models.transformer_head_gates as tr

hp = {
        "num_layers": 6,
        "num_heads": 8,
        "ff_size": 2048,
        "ffn_type": "conv_relu",
        "hid_size": 512,
        "emb_size": 512,
        "res_steps": "nlda",

        "rescale_emb": True,
        "inp_emb_bias": True,
        "normalize_out": True,
        "share_emb": False,
        "replace": 0,

        "relu_dropout": 0.1,
        "res_dropout": 0.1,
        "attn_dropout": 0.1,
        "label_smoothing": 0.1,

        "translator": "ingraph",
        "beam_size": 4,
        "beam_spread": 3,
        "len_alpha": 0.6,
        "attn_beta": 0,
    }

def main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--path', default='experiments/baseline_5M/model')
    parser.add_argument('--checkpoint', default='model-28672.npz')
    parser.add_argument('--read', default='test.bpe')
    parser.add_argument('--write', default='test.hdf5')
    args = parser.parse_args()

    assert args.read # and args.write

    inp_voc = pickle.load(open(os.path.join(args.path, 'src.voc'), 'rb'))
    out_voc = pickle.load(open(os.path.join(args.path, 'dst.voc'), 'rb'))
    path_to_ckpt = os.path.join(args.path, 'checkpoint', args.checkpoint)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    model = tr.Model('mod', inp_voc, out_voc, inference_mode='fast', **hp)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    lib.train.saveload.load(path_to_ckpt, var_list)

    hf = h5py.File(args.write, 'w')

    def gen_batches(data, size):
        l = len(data)
        for n in range(0, l, size):
            yield (data[n:min(n + size, l)])

    f = open(args.read, 'r').read().splitlines()
    print("Encoding {} sentences".format(len(f)))

    counter = 0
    f1 = tf.placeholder(dtype='int32')
    f2 = tf.placeholder(dtype='int32')
    graph = model.transformer.decode(f1, f2, False)
    for batch in tqdm.tqdm(gen_batches(f, args.batch_size)):
        blank = [""] * len(batch)
        batch = list(zip(batch, blank))
        feed = model.make_feed_dict(batch)
        rep = graph.eval({f1: feed['inp'], f2: feed['inp_len']})
        for n, sent in enumerate(rep):
            data = sent[:(feed['inp_len'][n] - 1)]
            hf.create_dataset(str(counter), data=data)
            counter += 1

    hf.close()

if __name__ == '__main__':
    main()