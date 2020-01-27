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
parser.add_argument('--output_folder', default='./training_data', type=str,
                    help='The output folder.')

if __name__ == "__main__":
    args = parser.parse_args()
    line_count = 0
    folder_path = args.folder_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with open(args.annotation) as file:
        lines = file.readlines()

        for line in lines:
            annotation = line.split()

            file_name = annotation[0].strip().rstrip()
            classification = annotation[1].strip().rstrip()
            start_frame = int(annotation[2].strip().rstrip())
            end_frame = annotation[3].strip().rstrip()

            orig_file_path = args.folder_path + "/" + file_name
            new_file_name = classification+"_"+file_name
            new_file_path = args.output_folder + "/" + classification.strip() + "/" + new_file_name
            # Cut and save file
            # Parse fileinfo
            if os.path.isfile(orig_file_path):
                cap = cv2.VideoCapture(orig_file_path)
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                orig_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                start_frame = int(start_frame*fps/24)
                if end_frame == 'end':
                    end_frame = orig_frame_number
                else:
                    end_frame = int(int(end_frame)*fps/24)
                frame_number = end_frame - start_frame
                buf = np.empty((frame_number, frameHeight, frameWidth, 3),
                               np.dtype('uint8'))

                fc = 0
                output_fc = 0
                ret = True
                while (fc < end_frame and ret):
                    ret, frame = cap.read()
                    if ((fc >= start_frame) and ret):
                        buf[output_fc] = frame
                        output_fc += 1
                    fc += 1

                cap.release()
                print(new_file_path)
                out = cv2.VideoWriter(new_file_path, fourcc, fps, (frameWidth, frameHeight))
                print(out.isOpened())
                for i in range(np.size(buf, 0)):
                    out.write(buf[i])
                out.release()
            else:
                print(f"File not found: {orig_file_path}")

