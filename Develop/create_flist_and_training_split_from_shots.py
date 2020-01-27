import argparse
import cv2
from pathlib import Path
import os
import random


parser = argparse.ArgumentParser()

parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--output_folder', default='./annotation', type=str,
                    help='The folder for the annotation files.')

categories = {
    'pan':0,
    'tilt':1,
    'tracking':2
}

def write_annotation(file_list, output_path):
    annotation_str = "\n".join(map(lambda record: " ".join(map(str, record)), file_list))

    with open(output_path, "w+", newline='') as flist_file:
        flist_file.write(annotation_str)

if __name__ == '__main__':
    args = parser.parse_args()
    base_folder = args.folder_path
    annotation_dict = {}
    for classification in os.listdir(base_folder):
        print(classification)
        annotation_dict[classification]=[]
        folder = os.path.join(base_folder,classification)
        for file in os.listdir(folder):
            absolute_path = Path(os.path.join(folder,file)).absolute().as_posix()

            #get info about file
            cap = cv2.VideoCapture(absolute_path)
            frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # collect all infos needed
            annotation_dict[classification].append([absolute_path,categories.get(classification),0,frame_number])

    # Create flist (file list) file
    # shuffle
    test_files = []
    validation_files = []
    training_files = []

    for classification in annotation_dict:
        files = annotation_dict.get(classification)

        # shuffle files
        random.shuffle(files)

        #create split 80% train+val 20% test
        nr_files = len(files)
        val_files_number  = test_files_number = int(0.2*nr_files)
        train_files_number = nr_files-val_files_number-test_files_number


        test_files.extend(files[0:test_files_number])
        validation_files.extend(files[test_files_number:(test_files_number+val_files_number)])
        training_files.extend(files[(test_files_number+val_files_number):])

    random.shuffle(test_files)
    random.shuffle(validation_files)
    random.shuffle(training_files)

    # write files
    annotation_base_path = args.output_folder

    test_annotation_path = os.path.join(annotation_base_path,'test_shots.flist')
    val_annotation_path = os.path.join(annotation_base_path,'val_shots.flist')
    train_annotation_path = os.path.join(annotation_base_path,'train_shots.flist')

    write_annotation(test_files,test_annotation_path)
    write_annotation(validation_files,val_annotation_path)
    write_annotation(training_files,train_annotation_path)

