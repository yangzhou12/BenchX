import torch
import numpy as np
import os
from PIL import Image
import MedDataHelpers
from torch.utils.data import Dataset
from torchvision import transforms
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedDataset(Dataset):
    """Chx - Report dataset.""" #d
    def __init__(self, source = 'mimic_cxr', group='train', im_aug = 1,
                 out_heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 filters = []):
        # filepaths
        fps = {}
        fps['mimic_root_dir'] = '/home/faith/datasets/mimic_512' #NEEDED FOR TRAINING
        fps['mimic_meta_file'] = '/home/faith/datasets/mimic_512/mimic-cxr-2.0.0-metadata.csv' #NEEDED FOR TRAINING
        fps['mimic_csv_file'] = '/home/faith/datasets/mimic_512/mimic-cxr-2.0.0-split.csv' #NEEDED FOR TRAINING
        fps['mimic_chex_file'] = '/home/faith/datasets/mimic_512/mimic-cxr-2.0.0-chexpert.csv' #NEEDED FOR TRAINING
        fps['chexpert_root_dir'] = '/data/faith/datasets/' #NEEDED FOR EVAL
        fps['covid_chestxray_csv_file'] = '/n/data2/hms/XXXX-9/XXXX-4/covid19qu/infection_dat/data_list.csv' #not found
        fps['padchest_root_dir'] = '/n/data2/hms/XXXX-9/XXXX-4/padchest/' #not found
        fps['padchest_file'] = '/n/data2/hms/XXXX-9/XXXX-4/padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz' #not found
        sourceDict = {'m': 'mimic_cxr', 'ms': 'ms_cxr', 'i': 'indiana_cxr', 'c': 'chexpert', 'co': 'covid-chestxray', 'med': 'medpix', 'p':'padchest'}

        #setting attributes
        self.heads = out_heads
        self.source = (sourceDict[source] if source in sourceDict.keys() else source) #chexpert when evaluating
        self.group = group
        self.im_aug = im_aug #1
        self.get_orig = True
        self.get_findings = 'impression' not in filters #evaluates to False

        # Image processing
        self.im_preprocessing_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, ratio=(.9, 1.0)),
            transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.2, contrast=.2),
            #transforms.GaussianBlur(kernel_size=15, sigma=(.1, 3.0))
            transforms.Resize(size=(224, 224))])

        self.im_preprocessing_test = transforms.Compose([
            transforms.Resize(256),  #256
            transforms.CenterCrop(224)]) #224

        self.im_finish = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.totensor = transforms.Compose([
            transforms.ToTensor()
        ])

        self.im_list, self.root_dir = MedDataHelpers.getImList(sr = self.source, group = self.group, fps = fps,
                                                               heads = self.heads, filters = filters)
        print(self.source, group + " size= " + str(self.im_list.shape))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.source == 'mimic_cxr' or self.source == 'mscxr':
            ims = self.im_list.iloc[idx, :]
            img_name = os.path.join(self.root_dir, 'files', ims['pGroup'], ims['pName'], ims['sName'], ims['dicom_id'] + '.jpg') #edited this to files
            image = Image.open(img_name).convert("RGB")
            df = ims.loc[self.heads]
            if self.group == 'test' or self.source == 'mscxr':
                images = [self.im_preprocessing_test(image) for i in range(self.im_aug)]
            else:
                images = [self.im_preprocessing_train(image) for i in range(self.im_aug)]

            images = [self.im_finish(im) for im in images]
            sample['images'] = images
            sample['labels'] = df.to_dict()
            sample['texts'] = ims.loc['text'] #ims, df, text
            if self.get_findings:
                sample['findings'] = ims.loc['findings']
            return sample

        elif self.source == 'chextest': #originally chexpert
            df = self.im_list.iloc[idx, :]
            img_name = self.root_dir + df['Path']
            image = Image.open(img_name)
            image = image.convert("RGB")
            if self.group == 'test' or self.group == 'all':
                image = self.im_preprocessing_test(image)
            else:
                image = self.im_preprocessing_train(image)
            image = self.im_finish(image)
            df = df.loc[self.heads]
            sample['images'] = [image]
            sample['labels'] = df.to_dict()
            sample['study'] = str(re.search("(.*study\d+)", img_name).group(0))
            return sample #ims, df

        elif self.source == 'covid-chestxray':
            df = self.im_list.iloc[idx, :]
            img_name = df['im_path']
            image = Image.open(img_name)
            image = image.convert("RGB")
            image = self.im_preprocessing_test(image)
            image = self.im_finish(image)
            lung_mask = Image.open(df['lung_path'])
            inf_mask = Image.open(df['inf_path'])
            lung_mask = self.totensor(lung_mask)
            inf_mask = self.totensor(inf_mask)
            df = df.loc[['No Finding', 'Pneumonia', 'covid19']]
            sample['images'] = [image]
            sample['labels'] = df.to_dict()
            sample['inf_mask'] = inf_mask
            sample['lung_mask'] = lung_mask
            return sample
        
        elif self.source == 'padchest':
            df = self.im_list.iloc[idx, :]
            img_name = df['im_path']
            image = Image.open(img_name)
            image = Image.fromarray(np.divide(np.array(image), 2 ** 8 - 1))
            image = image.convert("RGB")
            image = self.im_preprocessing_test(image)
            image = self.im_finish(image)

            sample['images'] = [image]
            df = df.iloc[1:]
            sample['labels'] = df.to_dict() #labels and image pairs seems right


            return sample


