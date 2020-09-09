class Config:
    batch_size = 256
    lr = 1e-2
    momentum = 0.9
    weights_decay = 1e-5
    class_num = 26
    eval_interval = 1
    checkpoint_interval = 1
    print_interval = 50
    checkpoints = './checkpoints/'
    pretrained = None
    start_epoch = 0
    epoches = 150
