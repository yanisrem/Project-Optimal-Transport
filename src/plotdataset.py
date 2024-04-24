import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(sample, offset = 0):
    fig = plt.figure(figsize=(15, 15))
    columns = 5
    rows = 5
    for i in range(1, columns*rows +1):
        img = sample[offset+i].reshape(16,16)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, vmin=0.0, vmax=1.0)
    plt.show()

def just_plot(sample, color = "violet", label="", figsize=(5, 5)):
    x_sample = np.array(sample[:,0])
    y_sample = np.array(sample[:,1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.scatter(x_sample, y_sample,  s=10, color=color, label=label)
    plt.legend()
    plt.show()