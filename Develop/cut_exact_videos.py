import argparse
import csv
import os
import cv2
import numpy as np
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--annotation', default='./annotations.csv', type=str,
                    help='The annotation file.')
parser.add_argument('--output_folder', default='./training_data/shots/', type=str,
                    help='The output folder.')

if __name__ == "__main__":
    args = parser.parse_args()
    line_count = 0
    folder_path = args.folder_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with open(args.annotation) as annotation_csv:
        dialect = csv.Sniffer().sniff(annotation_csv.read(2048))
        annotation_csv.seek(0)
        annotation_reader = csv.reader(annotation_csv, dialect)

        # Column index for relevant information. befause format varies
        # gets set while parsing first row of document
        move_id_index = shot_id_index = start_frame_index = end_frame_index = classification_index = name_index = 0
        annotation_list = []
        for row in tqdm.tqdm(annotation_reader):
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
                new_file_path = args.output_folder + "/" + classification.strip() + "/"+ new_file_name
                #Cut and save file
                #Parse fileinfo
                if os.path.isfile(orig_file_path):
                    cap = cv2.VideoCapture(orig_file_path)
                    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    frame_number = end_frame-start_frame
                    buf = np.empty((frame_number,frameHeight, frameWidth, 3),
                                   np.dtype('uint8'))

                    fc = 0
                    output_fc = 0
                    ret = True
                    while(fc<end_frame and ret):
                        ret, frame = cap.read()
                        if(fc>=start_frame):
                            buf[output_fc] = frame
                            output_fc+=1
                        fc+=1

                    cap.release()
                    print(new_file_path)
                    out = cv2.VideoWriter(new_file_path, fourcc, fps, (frameWidth,frameHeight))
                    print(out.isOpened())
                    for i in range(np.size(buf, 0)):
                        out.write(buf[i])
                    out.release()

                    #Prepare own annotation format

            line_count+=1
