params = dict(
    device='cuda',
    epochs=10e4,
    lr=8e-5,
    gamma=0.1,
    mp_steps=6,
    embedding_dim=128,
    dim=2,
    proc_hidden_dim=128,
    encoder_hidden_dim=128,
    decoder_hidden_dim=128,
    normalization=True,
    vel_context=5,
    batch_size=4,
    model_save_interval=100,
    decay_interval=100
)
