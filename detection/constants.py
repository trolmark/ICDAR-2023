import pathlib

root_path = pathlib.Path('')
path_to_dataset = root_path / "dataset"
path_to_images = path_to_dataset / "tfrecords"

path_to_project = root_path / "detection"
checkpoint_path = path_to_project / "checkpoints"

path_to_test_dataset = path_to_dataset / "test_images"
path_to_prediction_output = root_path / "test_detection_output"
path_to_recognition_input = root_path / "test_recogntion_input"

INPUT_SIZE = 192
OUTPUT_STRIDE = 4
BATCH_SIZE = 32
EPOCHS = 30
HEATMAP_KERNEL_SIGMA = 2
MAX_KEYPOINTS_ON_IMAGE = 36 # 360 degrees/ 20 degress * 2