import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Download link: https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

data_path = "/source/path/to/COVIDx-CXR4/"
processed_datapath = "/target/path/to/COVIDx-CXR4/"

def process_data():
    desired_size = 224
    data_split = ['train', 'val', 'test']
    for split in data_split:
        image_path = os.path.join(data_path, split)
        txtpath = os.path.join(data_path, split + ".txt")

        csv = pd.read_csv(txtpath, sep=' ')
        column_names = ['patient_id', 'filename','class', 'data_source'] 
        csv.columns = column_names

        output_image_dir = os.path.join(processed_datapath, split)
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        for i, row in tqdm(csv.iterrows(), total=len(csv)):
            filename = row.filename
            img_id = filename.split(".")[0]
            img_path = os.path.join(image_path, filename)
            img = Image.open(img_path).convert("RGB")

            new_size = (desired_size, desired_size)
            final_image = img.resize(new_size, Image.Resampling.LANCZOS)
            # Save in the same image format
            final_image.save(os.path.join(output_image_dir, img_id + ".png"), 'PNG')
            # Update the file name
            csv.loc[i].filename = img_id + ".png"

        output_csvpath = os.path.join(processed_datapath, f"{split}.csv")
        csv.to_csv(output_csvpath, index=False)

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

def split_dataset(seed):
    train_csvpath = os.path.join(processed_datapath, "train.csv")
    val_csvpath = os.path.join(processed_datapath, "val.csv")
    test_csvpath = os.path.join(processed_datapath, "test.csv")

    train_csv = pd.read_csv(train_csvpath)
    val_csv = pd.read_csv(val_csvpath)
    test_csv = pd.read_csv(test_csvpath)

    train_csv["filename"] = train_csv["filename"].apply(lambda x: "train/" + x)
    val_csv["filename"] = val_csv["filename"].apply(lambda x: "val/" + x)
    test_csv["filename"] = test_csv["filename"].apply(lambda x: "test/" + x)

    x_train, y_train = train_csv["filename"].tolist(), train_csv["class"].tolist()
    x_val, y_val = val_csv["filename"].tolist(), val_csv["class"].tolist()
    x_test, y_test = test_csv["filename"].tolist(), test_csv["class"].tolist()

    # Further split the training set into subsets with 1% and 10% of the training samples
    x_train_1, _ = train_test_split(x_train, test_size=0.99, stratify=y_train, random_state=seed)
    x_train_10, _ = train_test_split(x_train, test_size=0.90, stratify=y_train, random_state=seed)

    save_anno(x_train, processed_datapath + '/train.txt')
    save_anno(x_train_1, processed_datapath + '/train_1.txt')
    save_anno(x_train_10, processed_datapath + '/train_10.txt')
    save_anno(x_val, processed_datapath + '/val.txt')
    save_anno(x_test, processed_datapath + '/test.txt')

    csv = pd.concat([train_csv, val_csv, test_csv], ignore_index=True)
    csv.to_csv(os.path.join(processed_datapath, "covidx_labels.csv"), index=False)

if __name__ == "__main__":
    process_data()
    split_dataset(seed=42)