from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RSNADataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'front'),
        palette=[[120, 120, 120], [6, 230, 230]])

    def __init__(self, **kwargs):
        super(RSNADataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs
        )