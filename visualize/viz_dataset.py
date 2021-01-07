import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import h5py
import numpy as np
from os import listdir
from os.path import join, splitext
from functools import cmp_to_key
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir')
args = parser.parse_args()


def compare(a, b):
    if int(splitext(a)[0]) > int(splitext(b)[0]):
        return 1
    else:
        return -1

def get_hd5_boxbath(path):
    files = []

    for idx, i in enumerate(sorted(listdir(path), key=cmp_to_key(compare))):
        files.append(h5py.File(join(path, i), 'r'))

    tstep_pos = []

    for idx, i in enumerate(files):
        tstep_pos.append((i['positions']))

    return np.array(tstep_pos)

def animation_func(num, data, points):
    points._offsets3d = (data[num][:, 0], data[num][:, 2], data[num][:, 1])
    return points


fig = plt.figure()
ax = p3.Axes3D(fig)

data = get_hd5_boxbath(args.dir)

points = ax.scatter(data[0][:, 0], data[0][:, 2], data[0][:, 1], s=1000, alpha=0.8)
anim = animation.FuncAnimation(fig, animation_func, data.shape[0], fargs=(data, points), interval=1)
plt.show()
