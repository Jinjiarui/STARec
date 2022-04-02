import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from SearchDataLoader import DataSet
from utils import load_model


def load_dataset(args):
    data_paths = {
        'taobao': 'TaoBao/taobao/raw_data',
        'tmall': 'TaoBao/tmall/raw_data',
        'alipay': 'TaoBao/alipay/raw_data',
    }
    dataset = os.path.join(args['data_dir'], data_paths[args['dataset']])
    return np.load(os.path.join(dataset, 'user_item.npz'))


def main(args):
    print(args)
    user_item = load_dataset(args)
    args.update({'all_features_num': np.sum(user_item['fields'])})
    model = load_model(args)
    dataset = DataSet(user_item, args['batch_size'], args['item_num'], args['hard_search'])
    print("dataset", args["dataset"])
    print("hidden dim", args["hidden_dim"])
    print("lr", args["lr"])
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_data = dataset.input_fn('train', epochs=args['num_epochs'], shuffle=False)
    input_x = tf.placeholder(tf.int32, name='x', shape=[None, None, dataset.fields_sum + 2])
    input_x_continuous = tf.placeholder(tf.int32, name='x_continuous',
                                        shape=[None, 30 - args['item_num'], dataset.fields_sum + 2])
    input_self_loc = tf.placeholder(tf.int32, name='self_loc', shape=[None])
    input_y = tf.placeholder(tf.float32, name='y', shape=[None])

    train_input = [input_x, input_x_continuous, input_self_loc, input_y]
    train_step = model(train_input)
    test_step = model(train_input, training=False)
    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    print("-----------------Training Start-----------------\n")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        average_time = 0
        for _ in tqdm(range(1000)):
            x, x_continuous, self_loc, label = sess.run(train_data)
            t1 = time.perf_counter()
            _ = sess.run(train_step, feed_dict={
                input_x: x,
                input_x_continuous: x_continuous,
                input_self_loc: self_loc,
                input_y: label
            })
            average_time += time.perf_counter() - t1
        print("Train Average time :{}".format(average_time / 1000.0))
        average_time = 0
        for _ in tqdm(range(1000)):
            x, x_continuous, self_loc, label = sess.run(train_data)
            t1 = time.perf_counter()
            _ = sess.run(test_step, feed_dict={
                input_x: x,
                input_x_continuous: x_continuous,
                input_self_loc: self_loc,
                input_y: label
            })
            average_time += time.perf_counter() - t1
        print("Test Average time :{}".format(average_time / 1000.0))


if __name__ == "__main__":
    import argparse
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="STARec")
    parser.add_argument(
        "-m", "--model", type=str, choices=['STARec', 'Sim', 'light-STARec'], default='STARec', help="Model to use"
    )
    parser.add_argument('-d', '--dataset', type=str, choices=["taobao", "tmall", 'alipay'],
                        default="tmall", help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--use_label', action="store_true", default=False)
    parser.add_argument('--hard_search', action="store_true", default=False)
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument("-c", "--cuda", type=str, default="0")
    parser.add_argument("--postfix", type=str, default="", help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    args = parser.parse_args().__dict__
    args["exp_name"] = "_".join([args["model"], args["dataset"]])
    args.update(get_exp_configure(args))
    if args["cuda"] == "none":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args["cuda"]
    main(args)
