import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from os.path import join
import numpy as np
def plot_training(training_history, save_name):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))


    #1、损失相关
    ax1.plot(training_history.history['loss'])
    ax1.plot(training_history.history['out_seg_loss'])
    ax1.plot(training_history.history['out_recon_loss'])


    ax1.set_title('loss compare')
    ax1.set_ylabel('loss', fontsize=12)
    ax1.legend(['loss', 'out_seg_loss','out_recon_loss','out_seg_dice_hard'], loc='upper left')
    # ax1.set_yticks(np.arange(0, 1.05, 0.05))

    ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_dice_hard'])))

    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')


    f.savefig(save_name)
    plt.close()