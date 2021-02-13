import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
import h5py
import numpy as np
import argparse

def print_attrs(name, obj):
    for key in obj.attrs:
        print("{}".format(key))

def get_hdf5(file, path, num):
    f = h5py.File(file, 'r')

    dataset = f[path]
    positions = dataset['positions'].get(num)

    arr = np.zeros(positions.shape, dtype='double')
    positions.read_direct(arr)

    return positions

def animation_func(num, data, points):
    points.set_offsets(data[num])
    return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('path')
    parser.add_argument('rollout')
    args = parser.parse_args()

    fig, ax = plt.subplots()
    plt.axis('scaled')

    data = get_hdf5(args.dir, args.path, args.rollout)

    # print(data.shape)

    ax.axis([0, 1, 0, 1])

    points = ax.scatter(data[0][:, 0], data[0][:, 1])
    # anim = animation.FuncAnimation(fig, animation_func, int(data.shape[0]), fargs=(data, points), interval=1)
    fig.show()
    fig.canvas.draw()

    for i in range(data.shape[0]):
        print('test')
        points.set_offsets(data[i])
        fig.canvas.draw()
