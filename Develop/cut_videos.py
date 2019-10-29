import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path to videos.')
parser.add_argument('--annotation', default='./annotations.csv', type=str,
                    help='The annotation file.')
parser.add_argument('--output_folder', default='./training_data/shots', type=str,
                    help='The output folder.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled.')