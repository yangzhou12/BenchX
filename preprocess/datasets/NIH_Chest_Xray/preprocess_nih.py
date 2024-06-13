import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Download link: https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset

# Specify data paths and desired preprocessed image size
data_path = "/source/path/to/NIH_ChestX-ray/"
processed_datapath = "/target/path/to/NIH_ChestX-ray/"

if not os.path.exists(processed_datapath):
    os.makedirs(processed_datapath)
output_image_dir = os.path.join(processed_datapath, "images")
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

def process_data():
    desired_size = 224
    source_directory = os.path.join(data_path, "images")
    files_to_copy = os.listdir(source_directory)
    # Iterate over the files and copy them to the destination directory
    for file_name in tqdm(files_to_copy):
        source_path = os.path.join(source_directory, file_name)
        destination_path = os.path.join(output_image_dir, file_name)
        
        img = Image.open(source_path).convert("RGB")
        old_size = img.size
        new_size = (desired_size, desired_size)
        final_image = img.resize(new_size, Image.Resampling.LANCZOS)
        final_image.save(destination_path)

        # shutil.copy2(source_path, destination_path)


def read_mrm_split(split):
    if split =="train_1":
        split_path = os.path.join(data_path, "train_1.txt")
    elif split =="train_10":
        split_path = os.path.join(data_path, "train_10.txt")
    elif split =="train":
        split_path = os.path.join(data_path, "train_list.txt")
    elif split =="val":
        split_path = os.path.join(data_path, "val_list.txt")
    elif split =="test":
        split_path = os.path.join(data_path, "test_list.txt")

    train_images = []
    train_labels = []
    with open(split_path, "r") as file_descriptor:
        for line in file_descriptor:
            line_items = line.split()
            image_file = line_items[0]
            image_label = line_items[1:]
            image_label = [int(i) for i in image_label]
            train_images.append(image_file)
            train_labels.append(image_label)

    return train_images, train_labels

def save_anno(img_list, file_path, remove_suffix=False):
    if remove_suffix:
        img_list = [
            img_path.split('/')[-1] for img_path in img_list
        ]
        img_list = [
            '.'.join(img_path.split('.')[:-1]) for img_path in img_list
        ]
    with open(file_path, 'w') as file_:
        for x in list(img_list):
            file_.write(x + '\n')


def split_dataset():
    # Use MRM splits
    data_splits = (read_mrm_split(split) for split in ["train", "val", "test", "train_1", "train_10"])
    x_train, x_val, x_test, x_train_1, x_train_10 = data_splits

    save_anno(x_train[0], processed_datapath + '/train.txt')
    save_anno(x_train_1[0], processed_datapath + '/train_1.txt')
    save_anno(x_train_10[0], processed_datapath + '/train_10.txt')
    save_anno(x_val[0], processed_datapath + '/val.txt')
    save_anno(x_test[0], processed_datapath + '/test.txt')

    csvpath = os.path.join(data_path, "Data_Entry_2017_v2020.csv")
    csv = pd.read_csv(csvpath)

    train_split = pd.DataFrame({'Image Index': x_train[0], "labels": x_train[1], "split": len(x_train[0]) * ["train"]})
    val_split = pd.DataFrame({'Image Index': x_val[0], "labels": x_val[1], "split": len(x_val[0]) * ["val"]})
    test_split = pd.DataFrame({'Image Index': x_test[0], "labels": x_test[1], "split": len(x_test[0]) * ["test"]})

    train_csv = pd.merge(train_split, csv, on='Image Index')
    val_csv = pd.merge(val_split, csv, on='Image Index')
    test_csv = pd.merge(test_split, csv, on='Image Index')

    csv = pd.concat([train_csv, val_csv, test_csv], axis=0, ignore_index=True)
    csv = csv.drop('Unnamed: 11', axis=1)
    csv.to_csv(os.path.join(processed_datapath, "nih_labels.csv"), index=False)

if __name__ == "__main__":
    process_data()
    split_dataset()