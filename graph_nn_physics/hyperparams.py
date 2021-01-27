params = dict(
    device='cuda',
    epochs=5 * 10e4,
    lr=1e-4,
    gamma=0.1,
    mp_steps=10,
    embedding_dim=128,
    dim=2,
    proc_hidden_dim=128,
    encoder_hidden_dim=128,
    decoder_hidden_dim=128,
    normalization=True,
    noise_std=6.7e-4,
    vel_context=5,
    batch_size=2,
    model_save_interval=500,
    decay_interval=5 * 10e3
)
