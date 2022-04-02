def get_exp_configure(args):
    configure_list = {'taobao': {}, 'alipay': {}, 'tmall': {}}
    model_list = ['STARec', 'Sim', 'light-STARec', 'PNN', 'SimpleSTARec']
    for dataset in configure_list.keys():
        for model in model_list:
            configure_list[dataset][model] = {
                'input_dim': 48,
                'hidden_dim': 64,
                'item_num': 15,
                'batch_size': 100,
                'dropout_rate': 0.5,
                'decay_step': 1000,
                'min_lr': 1e-6,
                'l2_reg': 4e-5,
                'predict_hidden_width': [256, 64, 16],
                'long_time': False,
            }
    configure_list['taobao']['STARec'].update({'long_time': True})
    return configure_list[args['dataset']][args['model']]
