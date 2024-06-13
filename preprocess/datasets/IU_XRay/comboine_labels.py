import pandas as pd
import json

# Download link: https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg

ann = json.loads(open('iu_xray/annotation.json', 'r').read())

df = pd.DataFrame(ann['train'] + ann['val'] + ann['test'])
# df = df.drop(['id', 'image_path', 'split'], axis=1)
# df.head()

# csv_file = "iu_xray/iu_xray_reports.csv"
# df.to_csv(csv_file, index=False)

df_label = pd.read_csv('iu_xray/iu-xray-label.csv')
df_label.fillna(0, inplace=True)
# df_label = df_label.rename(columns={'report': 'Reports'})

df_all = pd.concat([df, df_label.iloc[:,1:]], axis=1)
dict_train = df_all[df_all['split'] == 'train'].to_dict(orient='records')
dict_val = df_all[df_all['split'] == 'val'].to_dict(orient='records')
dict_test = df_all[df_all['split'] == 'test'].to_dict(orient='records')

new_ann = {"train": dict_train, "val": dict_val, "test": dict_test}
with open('iu_xray/annotation_with_labels.json', 'w') as json_file:
    json.dump(new_ann, json_file, indent=2)