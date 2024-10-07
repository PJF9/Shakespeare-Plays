## Dataset Configurations
TRAIN_DATASETS_PATH = './data/train_data'
NORMALIZE_TRAIN_DATA_PATH = './data/norm_train_data'
TEST_DATASETS_PATH = './data/test_data'
NORMALIZE_TEST_DATA_PATH = './data/norm_test_data'
CONTRACTIONS_PATH = './data/contractions.json'
GENERATE_PATH = './generated'
MODELS_PATH = './checkpoints'
PLOTS_PATH = './plots'
LOGS_PATH = './logs'

## Training Configurations
BATCH_SIZE = 128
BLOCK_SIZE = 100
LSTM_CONFIGS = dict(
    embedding_dim=64,
    hidden_dim=64,
    num_layers=2,
    dropout=0.2
)
TRANSFORMER_CONFIGS = dict(
    embedding_dim=64,
    n_head=2,
    n_encoders=2,
    n_decoders=2,
    dim_feedforward=64,
    dropout=0.2
)
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.01
EPOCHS=40
GAMMA=0.98
NUM_TRAIN_DATA=4
