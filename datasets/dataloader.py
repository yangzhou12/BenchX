from . import classification_dataset
from . import pretraining_dataset
from . import transforms

from torch.utils.data import DataLoader


DATASETS = {
    "rsna_pneumonia": classification_dataset.RSNAImageDataset,
    "nih_chest_xray": classification_dataset.NIHChestXRay14,
    "pretrain": pretraining_dataset.MultimodalPretrainingDataset
}


def get_dataloaders(args):
    if args.dataset not in DATASETS:
        raise RuntimeError(w
            "Please specify a dataset.\n" +
            "Run --help to see datasets available for downstream task."
        )
    
    dataset_class = DATASETS[args.dataset]

    train_dataset = dataset_class('train', transforms.DataTransforms, args.data_pct)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )
    
    val_dataset = dataset_class('valid', transforms.DataTransforms, args.data_pct)
    val_dataloader = DataLoader(
            val_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    
    test_dataset = dataset_class('test', transforms.DataTransforms, args.data_pct)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    
    return train_dataloader, val_dataloader, test_dataloader