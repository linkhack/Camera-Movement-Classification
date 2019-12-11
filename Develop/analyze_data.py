import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--flist', default='annotation.flist',type=str,help='The dataset to analyze.')

if __name__ == '__main__':
    args = parser.parse_args()
    flist = args.flist


    with open(flist, 'r') as file:
        """
        0: name 1: label 2: start 3:end
        """
        content = file.read()
        lines = content.splitlines()
        lines = [tuple(line.split()) for line in lines]
    labels = [int(line[1]) for line in lines]
    durations = [int(line[3])-int(line[2]) for line in lines]
    unique, counts = np.unique(labels, return_counts=True)
    distribution = counts/len(labels)
    durations_per_class = {0:[], 1:[], 2:[]}
    for label, duration in zip(labels, durations):
        durations_per_class[label].append(duration)
    average_duation  = {key:np.mean(durations_per_class.get(key)) for key in durations_per_class}
    print(counts)
    print(distribution)
    print(average_duation)

    # Plots:
    # Histogram per class
    plt.hist(durations_per_class[0])
    plt.title("Distribution of durations of pans")
    plt.xlabel("Duration in frames")
    plt.ylabel("Count")
    plt.show()
    plt.hist(durations_per_class[1])
    plt.title("Distribution of durations of tilts")
    plt.xlabel("Duration in frames")
    plt.ylabel("Count")
    plt.show()
    plt.hist(durations_per_class[2])
    plt.title("Distribution of durations of tracking")
    plt.xlabel("Duration in frames")
    plt.ylabel("Count")
    plt.show()

    # Class distribution
    plt.bar(range(3),counts,tick_label=['Pan','Tilt','Tracking'])
    plt.title("Distribution of class labels")
    plt.xlabel("Class label")
    plt.ylabel("Count")
    plt.show()
