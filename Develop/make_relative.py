import argparse
import cv2
from pathlib import Path
import os
import random
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--annotation', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--output', default='./annotation', type=str,
                    help='The folder for the annotation files.')

def write_annotation(file_list, output_path):
    annotation_str = "\n".join(map(lambda record: " ".join(map(str, record)), file_list))

    with open(output_path, "w+", newline='') as flist_file:
        flist_file.write(annotation_str)

if __name__ == '__main__':
    args = parser.parse_args()
    output_file = args.output
    annotation = args.annotation

    with open(annotation, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        lines = [tuple(line.split()) for line in lines]

        lines =  [(os.path.relpath(line[0],'.'),line[1],line[2],line[3]) for line in lines]

        write_annotation(file_list=lines,output_path=output_file)