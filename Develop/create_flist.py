import argparse
import csv
from pathlib import Path
import os


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--annotation', default='./annotations.csv', type=str,
                    help='The annotation file.')

parser.add_argument('--output', default='./annotation.flist', type=str,
                    help = 'Output file')

categories = {
    'pan':0,
    'tilt':1,
    'tracking':2
}

if __name__ == "__main__":
    args = parser.parse_args()
    line_count = 0
    folder_path = args.folder_path
    with open(args.annotation) as annotation_csv:
        dialect = csv.Sniffer().sniff(annotation_csv.read(2048))
        annotation_csv.seek(0)
        annotation_reader = csv.reader(annotation_csv, dialect)

        # Column index for relevant information. befause format varies
        # gets set while parsing first row of document
        move_id_index = shot_id_index = start_frame_index = end_frame_index = classification_index = name_index = 0
        annotation_list = []
        for row in annotation_reader:
            if line_count==0:
                # find index of relevant columns
                for index, column_name in enumerate(row):
                    if column_name == 'mid' or column_name=='ID_Movies':
                        move_id_index=index
                    elif column_name == 'ID_Annotations' or column_name=='movie_shot_id':
                        shot_id_index=index
                    elif column_name == 'start_frame_idx' or column_name=='startTime':
                        start_frame_index = index
                    elif column_name == 'end_frame_idx' or column_name == 'endTime':
                        end_frame_index = index
                    elif column_name == 'camera_movement' or column_name == 'description':
                        classification_index = index
                    elif column_name == 'eF_FILM_ID' or column_name == 'movie_name':
                        name_index = index
            else:
                #read values
                movie_id = row[move_id_index]
                shot_id = row[shot_id_index]
                start_frame = int(row[start_frame_index])
                end_frame = int(row[end_frame_index])
                classification = categories.get(row[classification_index].strip())
                print(row[classification_index])
                movie_name = row[name_index]

                file_path =  Path(args.folder_path + "/" + movie_name + ".mp4").absolute().as_posix()
                #Parse fileinfo
                if os.path.isfile(file_path):
                    annotation_list.append([file_path, classification, start_frame, end_frame])

            line_count+=1

        #write flist only if could open original flist
        new_annotation_path = args.output

        annotation_str = "\n".join(map(lambda record: " ".join(map(str,record)), annotation_list))

        with open(new_annotation_path, "w+", newline='') as flist_file:
            flist_file.write(annotation_str)

