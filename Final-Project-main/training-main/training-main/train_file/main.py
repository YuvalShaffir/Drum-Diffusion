import os
import torch

data_prep_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train_file\\utils\\data_prep_utils.py"
audio_data_dir = "D:\\yuval.shaffir\\fma_small"
csv_file_name = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train_file\\annotation.csv"
save_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train_file"
graveyard = ""

unet_train_file_path = "unet_train.py"
import subprocess


if __name__ == '__main__':
    # os.system(f"python {data_prep_path} {audio_data_dir} {csv_file_name} {save_path}")
    # os.system(f"python {unet_train_file_path} {csv_file_name}")
    subprocess.run(f"python {data_prep_path} {audio_data_dir} {csv_file_name} {save_path}")
