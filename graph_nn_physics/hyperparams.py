params = dict(
    device='cuda',
    epochs=10e4,
    lr=5e-4,
    mp_steps=4,
    embedding_dim=32,
    dim=2,
    proc_hidden_dim=64,
    encoder_hidden_dim=16,
    decoder_hidden_dim=16,
    relative_encoder=False,
    vel_context=5,
    batch_size=1,
)
