import os
import re
import sys
import math
from collections import deque
import matplotlib.pyplot as plt


class MovingAverage():
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return

        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()

    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())

    def __repr__(self):
        return repr(self.get_avg())

    def __len__(self):
        return len(self.window)


with open(sys.argv[1], 'r') as f:
    inp = f.read()

patterns = {
    'train': re.compile(r'iteration (?P<iteration>\S+) train_loss (?P<train_loss>\S+)'),
    'LCX': re.compile(r'LCX: precision (?P<precision>\S+) recall (?P<recall>\S+) f1-score (?P<f1>\S+)'),
    'acc': re.compile(r'train_loss (?P<train_loss>\S+) test_loss (?P<test_loss>\S+) '
                      r'train_acc (?P<train_acc>\S+) test_acc (?P<test_acc>\S+)'),
    'micro': re.compile(r'micro avg: precision (?P<precision>\S+), recall (?P<recall>\S+)'
                        r', f1-score (?P<f1>\S+)'),
    'macro': re.compile(r'macro avg: precision (?P<precision>\S+), recall (?P<recall>\S+)'
                        r', f1-score (?P<f1>\S+)'),
    'weighted': re.compile(r'weighted avg: precision (?P<precision>\S+), recall (?P<recall>\S+)'
                           r', f1-score (?P<f1>\S+)'),
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
    for key, pattern in patterns.items():
        f = pattern.search(line)

        if f is not None:
            datum = f.groupdict()
            for k, v in datum.items():
                if v is not None:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                    datum[k] = v

            if key == 'val':
                datum = (datum, data['train'][-1])
            data[key].append(datum)
            break


def smoother(y, interval=100):
    avg = MovingAverage(interval)

    for i in range(len(y)):
        avg.append(y[i])
        y[i] = avg.get_avg()

    return y


def plot_train(data_train):
    plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    loss_names = ['Train Loss']
    loss_data_filename = sys.argv[2]
    with open(loss_data_filename, 'w') as loss_dat:
        print("len of train data: ", len(data_train))
        loss_data = smoother([y['train_loss'] for y in data_train[:25000]])
        for y in loss_data:
            loss_dat.write(str(y))
            loss_dat.write('\n')

    x = [x['iteration'] for x in data_train]
    plt.plot(x, smoother([y['train_loss'] for y in data_train]))
    # plt.plot(x, smoothsddr([y['test_loss'] for y in data_train]))

    # if data[0]['s'] is not None:
    #     plt.plot(x, smoother([y['s'] for y in data_train]))
    #     loss_names.append('Segmentation Loss')

    plt.legend(loss_names)
    plt.show()
    # plt.savefig(sys.argv[2])


def plot_f1(metrics_data):
    f1 = 0
    f1_dic = None
    for metric in metrics_data:
        if metric['f1'] > f1:
            f1 = metric['f1']
            f1_dic = metric
    print(f1_dic)


def plot_acc(data_train):
    test_acc = 0
    for y in data_train:
        if y['test_acc'] > test_acc:
            test_acc = y['test_acc']
    print("test acc: ", test_acc)

    # plt.title(os.path.basename(sys.argv[1]) + ' Training Acc')
    # plt.xlabel('Iteration')
    # plt.ylabel('Acc')

    # acc_name = ['Train Acc', 'Test Acc']

    # x = [x['iteration'] for x in data_train]
    # plt.plot(x, smoother([y['train_acc'] for y in data_train]))
    # plt.plot(x, smoother([y['test_acc'] for y in data_train]))

    # plt.legend(acc_name)
    # plt.savefig(sys.argv[3])
    # plt.show()

# plt.savefig(sys.argv[2])
# plt.show()


def plot_val(data_val):
    plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')

    x = [x[1]['epoch'] for x in data_val if x[0]['type'] == 'box']
    plt.plot(x, [x[0]['all'] for x in data_val if x[0]['type'] == 'box'])
    plt.plot(x, [x[0]['all'] for x in data_val if x[0]['type'] == 'mask'])

    plt.legend(['BBox mAP', 'Mask mAP'])
    plt.savefig(sys.argv[3])


# plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
    plot_val(data['val'])
if len(sys.argv) > 2 and sys.argv[2] == 'acc':
    plot_acc(data['acc'])
    print('micro')
    plot_f1(data['micro'])
    print('macro')
    plot_f1(data['macro'])
    print('weighted')
    plot_f1(data['weighted'])
    print('LCX')
    plot_f1(data['LCX'])
else:
    plot_train(data['train'])
