from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import os
from torch.optim.lr_scheduler import MultiStepLR



class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split

        if split == "train":
            self.transform = transform(is_train=True)
        else:
            self.transform = transform(is_train=False)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SLAKE(BaseImageDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01, imsize=None):
        super().__init__(split=split, transform=transform) 
        
        self.config = config
        self.root = './dataset/' + dataset
        # self.anno_file = 'question_' + split + '.json'
        self.image_data_path = os.path.join('./dataset/', dataset, dataset +'_image_data.pkl')
        self.text_file =  os.path.join('./dataset/', dataset, dataset + '_text_data.pkl' )
        print("Read image data from", self.image_data_path)
        self.img_id2idx, self.idx2img_id, self.img_list_in_np = cPickle.load(open(self.image_data_path, 'rb'))
        self.entries, self.ans2label, self.label2ans = cPickle.load(open(self.text_file, 'rb'))

        self.entries = self.entries[split]
        self.max_len = 23
        if dataset == 'VQA-RAD':
            self.max_len = 30

        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        img_id = self.entries[index]['img_id']
        img_idx = self.img_id2idx[img_id]
        v = self.img_list_in_np[img_idx]
        q = self.entries[index]['q_ids']
        _a = self.entries[index]['label']
        # for unanswerable questions, set to ignore_index in nn.CELoss
        if _a == None:
            a = -100
        else:
            a = _a
        question_type = self.entries[index]['answer_type']
        return v, q, a, question_type, img_id, self.entries[index]['question']

    def __len__(self):
        return len(self.entries)

    def tokenize(self, tokenizer):
        max_q_len = 0
        assert tokenizer.eos_token == '[END]', 'tokenizer.eos_token must be [END]!'
        for entry in self.entries:
            question = entry['question'] + ' [END]'
            q_tokens = tokenizer.tokenize(question)
            q_ids = tokenizer.convert_tokens_to_ids(q_tokens)
            entry['q_ids'] = q_ids
            if len(q_ids) > max_q_len:
                max_q_len = len(q_ids)
        print("max question length in dataset:", max_q_len, "setting:", self.max_len)
        for entry in self.entries:
            q_ids = entry['q_ids']
            q_ids = np.array(q_ids, dtype=np.int64)
            q_new = np.zeros(self.max_len, dtype=np.int64)
            q_new[:min(q_ids.shape[0], self.max_len)] = q_ids[:min(q_ids.shape[0], self.max_len)]
            entry['q_ids'] = q_new