import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from tqdm import tqdm

from SearchDataLoader import SearchDataset, SearchCollate
from utils import set_random_seed, load_model, evaluate_utils


def load_dataset(args):
    data_paths = {
        'taobao': 'TaoBao/taobao/raw_data',
        'tmall': 'TaoBao/tmall/raw_data',
        'alipay': 'TaoBao/alipay/raw_data',
        'amazon': 'Amazon/',
        'movie': 'ml-25m/'
    }
    dataset = os.path.join(args['data_dir'], data_paths[args['dataset']])
    return np.load(os.path.join(dataset, 'user_item.npz')), np.load(os.path.join(dataset, 'category_users.npy'),
                                                                    allow_pickle=True)


def main(args):
    print(args)

    if args["rand_seed"] > -1:
        set_random_seed(args["rand_seed"])
    user_item, category_users = load_dataset(args)
    args.update({'all_users_num': len(user_item['begin_len']), 'all_features_num': np.sum(user_item['fields'])})
    model = load_model(args).to(device=args['device'])
    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args['l2_reg'])
    min_lr = args['min_lr']
    scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.96, patience=1000, verbose=True, min_lr=min_lr)
    print("scheduler min_lr", min_lr)
    criterion = nn.BCELoss()
    print("dataset", args["dataset"])
    print("hidden dim", args["hidden_dim"])
    print("lr", args["lr"])
    best_val_auc = 0
    model_folder = "./saved_models/"
    model_path = model_folder + str(args["exp_name"]) + "_" + str(args["postfix"])
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    train_data = SearchDataset(user_item, category_users, args['candidate_user_num'], args['assist_user_num'],
                               mode='train')
    valid_data = SearchDataset(user_item, category_users, args['candidate_user_num'], args['assist_user_num'],
                               mode='valid')
    test_data = SearchDataset(user_item, category_users, args['candidate_user_num'], args['assist_user_num'],
                              mode='test')
    collate = SearchCollate(train_data, model.embedding_layer, device=args['device'])
    train_dataloader = data.DataLoader(train_data, batch_size=args['batch_size'], collate_fn=lambda _: _,
                                       shuffle=True, num_workers=6)
    valid_dataloader = data.DataLoader(valid_data, batch_size=args['batch_size'], collate_fn=lambda _: _,
                                       shuffle=True, num_workers=6)
    test_dataloader = data.DataLoader(test_data, batch_size=args['batch_size'], collate_fn=lambda _: _,
                                      shuffle=True, num_workers=6)

    print("-----------------Training Start-----------------\n")
    if args['load_model']:
        model.load_state_dict(torch.load(model_path))
        print('Load Model from {}'.format(model_path))
    for epoch in range(args['num_epochs']):
        i = 0
        dur = []
        train_eval = [torch.tensor([]), torch.tensor([])]
        train_data.change_mode('train')
        collate.dataset = train_data
        model.train()
        for data_ in tqdm(train_dataloader):
            # try:
            t0 = time.perf_counter()
            x, x_continuous, y, self_loc = collate.collate_fn(data_)
            x, y = x.to(args['device']), y.to(args['device'])
            x_continuous = x_continuous.to(args['device'])
            optimizer.zero_grad()
            y_ = torch.squeeze(model(x, x_continuous, self_loc))
            loss = criterion(y_, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            train_eval[0] = torch.cat([train_eval[0], y.detach().cpu()])
            train_eval[1] = torch.cat([train_eval[1], y_.detach().cpu()])
            if train_eval[0].shape[0] > 10000:
                train_eval[0] = train_eval[0][-10000:]
                train_eval[1] = train_eval[1][-10000:]
            _, acc, auc, f1 = evaluate_utils(train_eval[1], train_eval[0])
            scheduler.step(auc)
            dur.append(time.perf_counter() - t0)
            dur = dur[-100:]
            avg_time = np.mean(dur)
            i += 1
            print('#Epoch:{}, #batch:{}, avg_time:{:.4f} loss:{:.4f} acc:{:.4f} auc:{:.4f}'
                  .format(epoch, i, avg_time, loss, acc, auc))
        # except Exception as e:
        #     print(e)

        print("-----------------Validating Start-----------------\n")
        j, val_eval = 0, [torch.tensor([]), torch.tensor([])]
        collate.dataset = valid_data
        model.eval()
        with torch.no_grad():
            for data_ in tqdm(valid_dataloader):
                try:
                    x, x_continuous, y, self_loc = collate.collate_fn(data_)
                    x, x_continuous = x.to(args['device']), x_continuous.to(args['device'])
                    y_ = torch.squeeze(model(x, x_continuous, self_loc)).cpu()
                    val_eval[0] = torch.cat([val_eval[0], y])
                    val_eval[1] = torch.cat([val_eval[1], y_])
                    j += 1
                    if j >= args['valid_step']:
                        break
                except:
                    continue
            loss, acc, auc, f1 = evaluate_utils(val_eval[1], val_eval[0], criterion)
            print("Validating loss:{:.4f} acc:{:.4f} auc:{:.4f}".format(loss, acc, auc))
        if auc >= best_val_auc:
            best_val_auc = auc
            print("New best dataset Saved!")
            torch.save(model.state_dict(), model_path)
    print("-----------------Testing Start-----------------\n")
    k, test_eval = 0, [torch.tensor([]), torch.tensor([])]
    collate.dataset = test_data
    model.eval()
    with torch.no_grad():
        for data_ in tqdm(test_dataloader):
            try:
                k += 1
                # if k > 200:
                #     break
                x, x_continuous, y, self_loc = collate.collate_fn(data_)
                x, x_continuous = x.to(args['device']), x_continuous.to(args['device'])
                y_ = torch.squeeze(model(x, x_continuous, self_loc)).cpu()
                test_eval[0] = torch.cat([test_eval[0], y])
                test_eval[1] = torch.cat([test_eval[1], y_])
            except:
                continue
        print(test_eval[1])
        print(test_eval[0])
        loss, acc, auc, f1 = evaluate_utils(test_eval[1], test_eval[0], criterion)
        print("Testing loss:{:.4f} acc:{:.4f} auc:{:.4f} f1:{:.4f}".format(loss, acc, auc, f1))


if __name__ == '__main__':
    import argparse
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description="Search-based Time-Aware Recommendation")
    parser.add_argument(
        "-m", "--model", type=str, choices=["STARec"], default="STARec", help="Model to use"
    )
    parser.add_argument('-d', '--dataset', type=str, choices=["taobao", "tmall", 'alipay', 'amazon', 'movie'],
                        default="tmall", help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument('--use_label', action='store_true', default=False)
    parser.add_argument("-c", "--cuda", type=str, default="0")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument('--valid_step', type=int, default=200)
    parser.add_argument("--postfix", type=str, default="", help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    args = parser.parse_args().__dict__
    # Get experiment configuration
    args["exp_name"] = "_".join([args["model"], args["dataset"]])
    args.update(get_exp_configure(args))

    if not args["cuda"] == "none":
        args["device"] = torch.device("cuda:" + str(args["cuda"]))
    else:
        args["device"] = torch.device("cpu")
    main(args)
