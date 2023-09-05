from . import pretraining_dataset
from . import classification_dataset
from . import segmentation_dataset
from . import zeroshot_dataset
from . import transforms

import torch
from torch.utils.data import DataLoader


_DATASETS = {
    "pretrain": pretraining_dataset.MultimodalPretrainingDataset,
    "rsna_pneumonia": classification_dataset.RSNAImageDataset,
    "nih_chest_xray": classification_dataset.NIHChestXRay14,
    "siim_acr_pneumothorax": segmentation_dataset.SIIMImageDataset,
    "mimic_5x200": zeroshot_dataset.MIMIC_5x200,
    "chexpert_5x200": zeroshot_dataset.CheXpert_5x200,
}


def get_ft_dataloaders(args):
    if args.dataset not in _DATASETS:
        raise RuntimeError(
            "Please specify a dataset.\n"
            + "Run --help to see datasets available for downstream task."
        )

    dataset_class = _DATASETS[args.dataset]

    # Get image transforms
    if args.phase in ["classification", "retrieval"]:
        transform = transforms.DataTransforms
    elif args.phase == "segmentation":
        transform = transforms.SegmentationTransforms

    train_dataset = dataset_class("train", transform, args.data_pct)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
    )

    val_dataset = dataset_class("valid", transform, args.data_pct)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    test_dataset = dataset_class("test", transform, args.data_pct)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_zeroshot_dataloader(args, tokenizer=None):
    if args.dataset not in _DATASETS:
        raise RuntimeError(
            "Please specify a dataset.\n"
            + "Run --help to see datasets available for downstream task."
        )

    dataset_class = _DATASETS[args.dataset]

    zeroshot_dataset = dataset_class(transforms.DataTransforms, tokenizer)
    sampler = torch.utils.data.RandomSampler(
        zeroshot_dataset, replacement=False, num_samples=len(zeroshot_dataset)
    )

    zeroshot_dataloader = DataLoader(
        zeroshot_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=zeroshot_dataset.collate_fn
        if hasattr(zeroshot_dataset, "collate_fn")
        else None,
        drop_last=False,
    )

    return zeroshot_dataloader
