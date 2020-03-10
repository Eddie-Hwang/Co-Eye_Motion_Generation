# Co-Eye_Motion_Generation
This is a PyTorch implementation of the Co-Eye Motion Generation Sequence-to-Sequence model.

This model generate eye motion sequence according to speech

This project support training and generation with trained model. Note that this project is still a work in progress.
if there is any suggestion or error, feel free to fire an issue to let me know by email ([ejhwang@nlp.kaist.ac.kr](mailto:ejhwang@nlp.kaist.ac.kr)).

## Requirement
- python 3.5.4
- torch
- sklearn
- matplotlib
- pandas
- seaborn
- numpy
- cv2
- pickle
- tqdm

## Usage

### 1. Run preprocessing from Speech Eye Motion Dataset(SEMD)
SEMD repository: (https://github.com/Eddie-Hwang/Speech-Eye-Motion-Dataset)
```bash
python run_preprocessing.py -dataset ./data/processed_eye_motion_dataset_pca_7.pickle -pretrained_emb ./data/glove.6B.300d.txt -data_size -1 -processed_path ./processed
```

### 2. Run train 
```bash
python train.py -data ./processed/processed_final.pickle -rnn_type LSTM -hidden 200 -n_layers 2 -dropout 0.1 -lr 0.0001 -beta 1.0 -chkpt ./chkpt -save_mode best
```

### 3. Run inference from trained model
You can download the trained file and processed dataset from: (https://drive.google.com/drive/folders/1Clz1yNz5WqgYeV0lUOkES6Kuxys90j3a?usp=sharing) 
```bash
python infer.py -data ./processed/processed_final.pickle -chkpt ./chkpt/eye_model.chkpt -vid_save_path ./output_vid
```


