params = dict(
    device='cuda',
    epochs=10e4,
    lr=8e-5,
    gamma=0.1,
    mp_steps=10,
    embedding_dim=64,
    dim=2,
    proc_hidden_dim=64,
    encoder_hidden_dim=64,
    decoder_hidden_dim=64,
    normalization=False,
    vel_context=5,
    batch_size=4,
    model_save_interval=100,
    decay_interval=500
)
