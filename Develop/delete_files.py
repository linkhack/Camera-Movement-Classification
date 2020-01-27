import argparse
import csv
import os
import cv2
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--annotation', default='./corrections.txt', type=str,
                    help='The annotation file.')

if __name__ == "__main__":
    args = parser.parse_args()
    line_count = 0
    folder_path = args.folder_path

    with open(args.annotation) as file:
        lines = file.readlines()

        for line in lines:
            annotation = line.split()

            file_name = annotation[0].strip().rstrip()

            orig_file_path = args.folder_path + "/" + file_name

            if os.path.isfile(orig_file_path):
                os.remove(orig_file_path)
            else:
                print(f"File not found: {orig_file_path}")

