import torch

import sys
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
from biovil.Modeling.CLIP_Embedding import MedCLIP

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def build_biovil_model(args, eval=False):
    biovil_model = MedCLIP(eval=eval).to(device)
    
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        biovil_model.load_state_dict(checkpoint, strict=False)

    return biovil_model
