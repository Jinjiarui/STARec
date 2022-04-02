def get_exp_configure(args):
    configure_list = {
        'taobao': {},
        'alipay': {},
        'tmall': {},
        'amazon': {},
        'movie': {}
    }
    rnn_model = ['LSTM', 'RRN', 'DUPN', 'NARM', 'STAMP', 'TimeLSTM', 'Motivate', 'DIN', 'DIEN']
    not_rnn_model = ['ESMM', 'ESMM2', 'MMOE']
    for dataset in configure_list.keys():
        configure_list[dataset]['STARec'] = {
            'input_dim': 48,
            'hidden_dim': 64,
            'output_dim': 1,
            'item_num': 15,
            'batch_size': 100,
            'dropout_rate': 0.5,
            'decay_step': 1000,
            'min_lr': 1e-6,
            'l2_reg': 4e-5,
            'predict_hidden_width': [256, 64, 16],
            'candidate_user_num': 200,
            'assist_user_num': 5,
            'long_time': False
        }
    for model in rnn_model:
        for dataset in configure_list.keys():
            configure_list[dataset][model] = {
                'input_dim': 48,
                'hidden_dim': 64,
                'output_dim': 1,
                'batch_size': 100,
                'dropout_rate': 0.5,
                'decay_step': 1000,
                'l2_reg': 4e-5,
                'predict_hidden_width': [256, 64, 16],
                'with_time': False
            }
    for model in not_rnn_model:
        for dataset in configure_list.keys():
            configure_list[dataset][model] = {
                'input_dim': 48,
                'hidden_dim': [64, 256, 64, 16, 1],
                'batch_size': 100,
                'dropout_rate': 0.5,
                'decay_step': 1000,
                'l2_reg': 4e-5,
                'with_time': False
            }
    configure_list['taobao']['STARec'].update({'long_time': True})
    configure_list[args['dataset']]['RRN'].update({'with_time': True})
    configure_list[args['dataset']]['TimeLSTM'].update({'with_time': True})
    configure_list[args['dataset']]['DIN'].update({'hidden_dim': 96})
    configure_list[args['dataset']]['DIEN'].update({'hidden_dim': 96})
    return configure_list[args['dataset']][args['model']]
