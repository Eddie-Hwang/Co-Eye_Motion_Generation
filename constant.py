DATA = './processed/processed_final_pca7.pickle'
D_WORD_VEC = 512
D_MODEL = 512
D_INNER = 2048
N_LAYERS = 6
N_HEADS = 8
D_K = int(D_MODEL / N_HEADS)
POSITION = 128

PAD = 0
UNK = 1
SOS = 2
EOS = 3

PAD_WORD = 'PAD'
UNK_WORD = 'UNK'
SOS_WORD = 'SOS'
EOS_WORD = 'EOS'

FRAME_DURATION = 1/10    # we assumed 10fps for processed dataset

# PRE_MOTIONS = 8
# ESTIMATION_MOTIONS = 32

PRE_MOTIONS = 10
ESTIMATION_MOTIONS = 20

# SPEECH_SPEED = 2.5 # assumed speech speed in dataset is 2.5 wps (word per seceond)
SPEECH_SPEED = 1.5
NUM_EYE_POINTS = 48