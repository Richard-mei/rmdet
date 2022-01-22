from os import listdir
from tensorboard.backend.event_processing import event_accumulator
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


class tensorboard_event_vis(object):
    def __init__(self):
        pass

    def __call__(self, event_file: str, key_to_vis: str, save_to_txt: str, save_curve: str, loc='lower right'):
        self.event_file = event_file
        self.txt_file = save_to_txt
        self.curve = save_curve

        self.load_event_data(self.event_file, key_to_vis, save_to_txt)
        self.plot_event_curve(self.txt_file, self.curve, loc=loc)

    @staticmethod
    def load_event_data(event_file=None, key_to_save=None, save_file=None):
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        assert key_to_save in ea.scalars.Keys(), f'Error at tensorboard_event_vis.load_event_data: ' \
                                                 f'\nkey_to_save must in {ea.scalars.Keys()},got {key_to_save}. '
        data_container = ea.scalars.Items(key_to_save)

        with open(save_file, 'a') as f:
            for i in tqdm(data_container, f'saving {key_to_save} data...'):
                f.write(f'{i.step},{i.value}\n')

    @staticmethod
    def plot_event_curve(event_file=None, curve_plots=None, loc='lower right'):
        file_list = event_file if isinstance(event_file, list) else [event_file]

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)

        for i, fi in enumerate(file_list):
            Step = []
            Value = []
            step = 0
            with open(fi, 'r') as f:
                for d in tqdm(f.readlines(), 'plotting..'):
                    d.strip('\n')
                    _, value = d.split(',')
                    Step.append(step)
                    Value.append(float(value))
                    step += 1
            ax1.plot(Step, Value, label=fi.split('/')[-1].split('.')[0])
            # ax1.set_xlim(i)
        ax1.set_xlabel("step")
        ax1.set_ylabel("")

        plt.legend(loc=loc)
        plt.savefig(f'{curve_plots}.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_file', type=str,
                        default='/home/richard/Projects/MMM/runs/rep-bifpn-gs-gf/vocdataset/20220120_233446/events.out.tfevents.1642692886.richard-NB_20220120_233446')
    parser.add_argument('--save_file', type=str, default='Repmodel_acc.txt')
    parser.add_argument('--files_list', type=list, default=['vgg16.txt'])
    args = parser.parse_args()

    tensorboard_event_vis()(event_file=args.event_file, key_to_vis='total_loss', save_to_txt='rep-bifpn.txt',
                            save_curve='rep-bifpn', loc='upper right')

    # # load_event_data(event_file=args.event_file, save_file=args.save_file)
    # plot_datas(file_list=['vgg16_acc.txt', 'shufflenetv2_acc.txt', 'resnet18_acc.txt', 'Repmodel_acc.txt'])
    # # plot_datas(file_list=['vgg16_acc.txt', 'shufflenetv2_acc.txt', 'resnet18_acc.txt', 'Repmodel_acc.txt'])
