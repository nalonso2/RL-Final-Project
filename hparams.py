

class HyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    data_dir = 'datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = 2 # actually batchsize * Seqlen
    seq_len = 32

    test_batch = 1
    n_sample = 64

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 500
    save_interval = 1000

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 10000
    n_rollout = 50
    seed = 0

    n_workers = 1

class RNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'

    extra = False
    data_dir = 'datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = 1 # actually batchsize * Seqlen
    test_batch = 1
    seq_len = 32
    n_sample = 64

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size
    rnn_hunits = 256
    log_interval = 1000
    save_interval = 1000

    max_step = 100000

    seed = 10

    n_workers = 0

class VAEHyperParams:
    vision = 'VAE'

    extra = False
    data_dir = 'datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    img_height = 96
    img_width = 96
    img_channels = 3

    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 3 # action size

    log_interval = 500
    save_interval = 2000

    max_step = 61000

    n_workers = 0
