from camclassifier.InferenceModel import InferenceModel
from camclassifier.DataLoader import DataLoader
import yaml
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='./Data/tilt_4_65457.mp4', type=str,
                    help='The file to classify.')
if __name__ == '__main__':
    args = parser.parse_args()
    file = args.file
    #open config
    stream = open('config.yml', 'r')
    config = yaml.load(stream, yaml.SafeLoader)
    inference_config = config.get('inference', dict())

    # Config DataLoading (strides preprocess etc.)
    data_args = DataLoader.get_args_from_config('config.yml').get('base')
    dataloader = DataLoader(**data_args)

    # Create Model
    inference_model = InferenceModel.build_model_from_config('config.yml')

    # save to csv
    csv = inference_config.get('csv_file', None)

    # load shot
    shot = dataloader.load_complete_shot(file)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    prediction = inference_model.predict(shot,name,'0', csv)

    print(prediction)

