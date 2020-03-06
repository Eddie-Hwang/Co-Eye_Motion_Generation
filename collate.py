import numpy as np
import torch

from constant import *


def collate_fn(insts):
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, trg_insts = list(zip(*seq_pairs))
    # find max length of each seqeunce
    max_src_len = max(len(inst) for inst in src_insts)
    max_trg_len = max(len(inst) for inst in trg_insts)
    
    sp_duration = max_src_len / SPEECH_SPEED
    pre_duration = PRE_MOTIONS * FRAME_DURATION
    expression_duration = ESTIMATION_MOTIONS * FRAME_DURATION

    num_words_for_pre_motion = round(max_src_len * pre_duration / sp_duration)
    num_words_for_estimation = round(max_src_len * expression_duration / sp_duration)
    total_motion_frames = PRE_MOTIONS + ESTIMATION_MOTIONS

    # padding src
    padded_src_array = np.array([inst + [PAD] * (max_src_len - len(inst)) for inst in src_insts])
    # padding trg
    for inst in trg_insts:
        trg_pad = []
        if max_trg_len - len(inst) > 0:
            for _ in range(max_trg_len - len(inst)):
                trg_pad.append([0] * len(inst[0]))
            inst += trg_pad
    padded_trg_array = np.array(trg_insts)

    src_seq_list = []
    src_length_list = []
    trg_seq_list = []
    for i in range(0, padded_src_array.shape[1] - num_words_for_pre_motion, num_words_for_estimation):
        input_seq = padded_src_array[:, i:i + num_words_for_pre_motion + num_words_for_estimation]
        # add SOS and EOS token
        input_seq = np.hstack((np.zeros((input_seq.shape[0], 1)) + SOS, input_seq))
        input_seq = np.hstack((input_seq, (np.zeros((input_seq.shape[0], 1)) + EOS)))
        # count seqeunce available length for future padded function of RNN
        input_seq_length = []
        for seq in input_seq:
            length = np.count_nonzero(seq)
            input_seq_length.append(length)
        # get eye expression_motion seqeunces
        output_seq = padded_trg_array[:, i:i + total_motion_frames, 2:] # don't use 1 and 2 plane
        # convert numpy array to torch tensor and add up
        src_seq_list.append(torch.LongTensor(input_seq)) # long type due to embedding 
        src_length_list.append(input_seq_length)
        trg_seq_list.append(torch.FloatTensor(output_seq))

    return zip(src_seq_list, src_length_list, trg_seq_list)

