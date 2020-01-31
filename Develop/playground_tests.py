from camclassifier.DataLoader import DataLoader
from camclassifier.CameraMovementClassifier import build_model

model = build_model()
training_configs = DataLoader.get_args_from_config('config.yml')

test_set = DataLoader(**training_configs.get('test'))
test_pipeline = test_set.validation_pipeline(batch_size=16)
for shot, label in test_pipeline:
    # Do something
    pass

complete_shots_generator = test_set.py_iterator()

for shot, label, file_name in complete_shots_generator:
    pass

