import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
import tabulate

parser.add_argument('--flist', default='annotation.flist', type=str, help='The dataset to analyze.')

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
    durations = [int(line[3]) - int(line[2]) for line in lines]
    unique, counts = np.unique(labels, return_counts=True)
    distribution = counts / len(labels)
    durations_per_class = {0: [], 1: []}
    for label, duration in zip(labels, durations):
        durations_per_class[label].append(duration)
    average_duration = {key: np.mean(durations_per_class.get(key)) for key in durations_per_class}
    minimum_duration = {key: np.min(durations_per_class.get(key)) for key in durations_per_class}
    maximum_duration = {key: np.max(durations_per_class.get(key)) for key in durations_per_class}
    print(counts)
    print(distribution)
    print(average_duration)
    print(minimum_duration)
    print(maximum_duration)

    average_duration = np.array(list(average_duration.values()))
    minimum_duration = np.array(list(minimum_duration.values()))
    maximum_duration = np.array(list(maximum_duration.values()))
    print(average_duration)
    array = np.stack([counts, average_duration, minimum_duration, maximum_duration])

    print(tabulate.tabulate(array, tablefmt="latex", floatfmt=".2f"))

    # # Plots:
    # # Histogram per class
    # plt.figure()
    # plt.hist(durations_per_class[0])
    # plt.title("Distribution of durations of pans")
    # plt.xlabel("Duration in frames")
    # plt.ylabel("Count")
    # plt.figure()
    # plt.hist(durations_per_class[1])
    # plt.title("Distribution of durations of tilts")
    # plt.xlabel("Duration in frames")
    # plt.ylabel("Count")
    #
    # plt.figure()
    # plt.hist(durations_per_class[2])
    # plt.title("Distribution of durations of tracking")
    # plt.xlabel("Duration in frames")
    # plt.ylabel("Count")
    #
    # # Class distribution
    # plt.figure()
    # plt.bar(range(3),counts,tick_label=['Pan','Tilt'])
    # plt.title("Distribution of class labels")
    # plt.xlabel("Class label")
    # plt.ylabel("Count")
    # plt.show()
