import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import h5py
import numpy as np
import argparse


def print_attrs(name, obj):
    print(name)
    for key in obj.attrs:
        print("{}".format(key))

def get_hdf5(file, path, num):
    f = h5py.File(file, 'r')
    group = f[path]
    group = group['positions']

    return np.array(group['{}'.format(num)])

def animation_func(num, data, points):
    points._offsets3d = (data[num * 8][:, 0], data[num * 8][:, 2], data[num * 8][:, 1])  # , data[num][:, 1])
    return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('path')
    parser.add_argument('rollout')
    args = parser.parse_args()

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    data = get_hdf5(args.dir, args.path, args.rollout)

    ax.set_xlim3d([0, 0.9])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 0.9])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 0.9])
    ax.set_zlabel('Z')

    points = ax.scatter(data[0][:, 0], data[0][:, 2], data[0][:, 1])  # , data[0][:, 1], s=1000, alpha=0.8)
    anim = animation.FuncAnimation(fig, animation_func, int(data.shape[0] / 8), fargs=(data, points), interval=1)
    plt.show()
