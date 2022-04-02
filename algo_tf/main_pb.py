import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

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
    dataset = DataSet(user_item, args['batch_size'], args['item_num'])
    print("dataset", args["dataset"])
    print("hidden dim", args["hidden_dim"])
    print("lr", args["lr"])
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_data = dataset.input_fn('train', epochs=args['num_epochs'])

    input_x = tf.placeholder(tf.int32, name='x', shape=[None, None, dataset.fields_sum + 2])
    input_x_continuous = tf.placeholder(tf.int32, name='x_continuous',
                                        shape=[None, args['item_num'], dataset.fields_sum + 2])
    input_self_loc = tf.placeholder(tf.int32, name='self_loc', shape=[None])
    input_y = tf.placeholder(tf.float32, name='y', shape=[None])

    train_input = [input_x, input_x_continuous, input_self_loc, input_y]
    train_base_loss, train_loss, train_eval_metric_ops, train_y_, train_y, train_op = model(train_input)

    saver = tf.train.Saver(max_to_keep=2)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    step_up = len(dataset.user_log_id) // args['batch_size']
    print("-----------------Training Start-----------------\n")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, model_path)
        eval_ops = tf.identity(train_base_loss, name='eval')
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['eval'])
        with tf.gfile.GFile(os.path.join(model_folder, 'model.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def main2(args):
    print(args)
    user_item = load_dataset(args)
    args.update({'all_features_num': np.sum(user_item['fields'])})
    model = load_model(args)
    dataset = DataSet(user_item, args['batch_size'], args['item_num'])
    print("dataset", args["dataset"])
    print("hidden dim", args["hidden_dim"])
    print("lr", args["lr"])
    model_folder = "./saved_models/"

    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_data = dataset.input_fn('train', epochs=args['num_epochs'])
    with  tf.Session() as sess:

        with gfile.GFile(os.path.join(model_folder, 'model.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        input_x = sess.graph.get_tensor_by_name('x:0')
        input_x_continuous = sess.graph.get_tensor_by_name('x_continuous:0')
        input_self_loc = sess.graph.get_tensor_by_name('self_loc:0')
        input_y = sess.graph.get_tensor_by_name('y:0')
        y_ = sess.graph.get_tensor_by_name('eval:0')
        for i in range(10):
            x, x_continuous, self_loc, label = sess.run(train_data)
            batch_eval = sess.run(y_, feed_dict={
                input_x: x,
                input_x_continuous: x_continuous,
                input_self_loc: self_loc,
                input_y: label
            })
            print(batch_eval)


if __name__ == "__main__":
    import argparse
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="STARec")
    parser.add_argument(
        "-m", "--model", type=str, choices=['STARec'], default='STARec', help="Model to use"
    )
    parser.add_argument('-d', '--dataset', type=str, choices=["taobao", "tmall", 'alipay'],
                        default="tmall", help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--use_label', action="store_true", default=False)
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
    if args['load_model']:
        main(args)
    else:
        main2(args)
