import argparse
import csv
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--annotation', default='./annotations.csv', type=str,
                    help='The annotation file.')
parser.add_argument('--output_folder', default='./training_data/shots', type=str,
                    help='The output folder.')
parser.add_argument('--create-list', default='1', type = int,
                    help="Create file list and annotation")
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled.')

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
                classification = row[classification_index]
                movie_name = row[name_index]

                orig_file_path = args.folder_path + "/" + movie_name + ".mp4"
                new_file_name = movie_id + "_" + shot_id + ".mp4"
                new_file_path = args.output_folder + "/" + new_file_name
                #Cut and save file
                #Parse fileinfo
                if os.path.isfile(orig_file_path):
                    info = ffmpeg_parse_infos(orig_file_path)
                    fps = info["video_fps"]
                    start_time = start_frame/fps
                    end_time = end_frame/fps

                    ffmpeg_extract_subclip(orig_file_path,start_time,end_time,new_file_path)

                    #Prepare own annotation format
                    annotation_list.append([new_file_name, classification])

            line_count+=1

        #write annotation only if could open original annotation
        new_annotation_path = args.output_folder + "/annotation.csv"
        with open(new_annotation_path, "w+", newline='') as csvfile:
            fieldnames = ['shot', 'classification']
            csvWriter = csv.writer(csvfile, delimiter=";")
            csvWriter.writerow(fieldnames)
            csvWriter.writerows(annotation_list)

