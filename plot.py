import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import seaborn as sns
import argparse

def display_loss(log_train_file, log_vaild_file):
    loss_tr = []
    loss_vf = []
    with open(log_train_file, 'r') as log_tf, open(log_vaild_file, 'r') as log_vf:
        for l in log_tf:
            line = l.rstrip()
            line = line.split(',')
            val = line[1].strip()
            try:
                loss_tr.append(float(line[1].strip()))
            except ValueError:
                pass
        for l in log_vf:
            line = l.rstrip()
            line = line.split(',')
            val = line[1].strip()
            try:
                loss_vf.append(float(line[1].strip()))
            except ValueError:
                pass

    loss_df = pd.DataFrame({
        'Epoch': [i for i in range(len(loss_tr))],
        'Train_loss': loss_tr,
        'Valid_loss': loss_vf
    })  

    plt.figure(figsize=(12, 9))
    sns.set_style('darkgrid')
    sns.lineplot(data=loss_df, x='Epoch', y='Train_loss', label='Train')
    sns.lineplot(data=loss_df, x='Epoch', y='Valid_loss', label='Test')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Test loss (K-fold cross validation)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default='./log')
    opt = parser.parse_args()

    display_loss(opt.log + '/train.log', opt.log + '/valid.log')
    plt.savefig(opt.log + '/loss.png')


if __name__ == '__main__':
    main()
