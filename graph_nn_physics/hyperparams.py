params = dict(
    device='cpu',
    epochs=10e4,
    lr=1e-4,
    mp_steps=4,
    embedding_dim=32,
    dim=3,
    proc_hidden_dim=64,
    encoder_hidden_dim=16,
    decoder_hidden_dim=16,
    relative_encoder=True,
    vel_context=5,
    batch_size=1,
)
