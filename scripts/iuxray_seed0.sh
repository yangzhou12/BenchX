CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mgca_vit.yml \
"name=Our_MGCA_ViT" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/gloria.yml "trainer.optim_params.lr=2e-3" "seed=0" 

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/medclip_resnet50.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/medclip_vit.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/medklip.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mgca_resnet50.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mgca_vit.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mrm.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/refers.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/convirt.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/gloria.yml \
"name=Our_GLoRIA" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/medklip.yml \
"name=Our_MedKLIP" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mgca_resnet50.yml \
"name=Our_MGCA_R50" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mrm.yml \
"name=Our_MRM" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/refers.yml \
"name=Our_REFERS" "pretrained_source=custom" "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/mflag.yml "trainer.optim_params.lr=2e-3" "seed=0"

CUDA_VISIBLE_DEVICES=1 python bin/train.py config/report_generation/IUXray/ptunifier.yml "trainer.optim_params.lr=2e-3" "seed=0"