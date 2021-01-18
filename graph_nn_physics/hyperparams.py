params = dict(
    device='cpu',
    epochs=10e4,
    lr=8e-5,
    gamma=0.7,
    mp_steps=10,
    embedding_dim=64,
    dim=2,
    proc_hidden_dim=64,
    encoder_hidden_dim=32,
    decoder_hidden_dim=32,
    relative_encoder=True,
    vel_context=5,
    batch_size=4,
    model_save_interval=500,
    decay_interval=750
)
