from camclassifier.InferenceModel import InferenceModel
from camclassifier.DataLoader import DataLoader
import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='./Data', type=str,
                    help='The folder path to shots.')

if __name__ == '__main__':
    #open config
    args = parser.parse_args()
    folder = args.folder
    stream = open('config.yml', 'r')
    config = yaml.load(stream, yaml.SafeLoader)
    inference_config = config.get('inference', dict())

    # Config DataLoading (strides preprocess etc.)
    data_args = DataLoader.get_args_from_config('config.yml').get('base')
    dataloader = DataLoader(**data_args)

    # Create Model
    inference_model = InferenceModel.build_model_from_config('config.yml')

    # save to csv
    csv_file = inference_config.get('csv_file', None)

    # load shots
    shots = []
    for file in os.listdir(folder):
        path = os.path.join(folder,file)
        if os.path.isfile(path):
            shot = dataloader.load_complete_shot(path)
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            id = '0'
            shots.append((shot,name,id))
    # Batch predict
    prediction = inference_model.batch_predict(shots,csv_file=csv_file)

    print(prediction)