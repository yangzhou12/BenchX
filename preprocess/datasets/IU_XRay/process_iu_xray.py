import pandas as pd
import json

# Download link: https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg
# Source: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

ann = json.loads(open('annotation.json', 'r').read())

train_df = pd.DataFrame(ann['train'])
train_df.head()

df = pd.DataFrame(ann['train'] + ann['val'] + ann['test'])
df.to_csv('iu_xray_annotations.csv', index=False)