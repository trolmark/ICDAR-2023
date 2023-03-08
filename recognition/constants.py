import pathlib

root_path = pathlib.Path('')
path_to_dataset = root_path / "dataset"
path_to_images = path_to_dataset / "tfrecords"
path_to_synth_images = path_to_dataset / "tfrecords" / "synth_text"

path_to_project = root_path / "recognition"
path_to_checkpoints = path_to_project / "checkpoints"
path_to_charset = path_to_project / "src" / "charset.txt"

path_to_test_dataset = root_path / "test_recogntion_input"
path_to_prediction_output = root_path / "test_recognition_output"

DELIMITER = '[!%&]'
START_TOKEN = '<START>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
SPE_TOKENS = [EOS_TOKEN, UNK_TOKEN]

INPUT_SIZE = 100
OUTPUT_STRIDE = 1
BATCH_SIZE = 64
EPOCHS = 30