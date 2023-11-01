import torch


def load_gloria(args, device="cuda"):
    try:
        import gloria
    except:
        raise RuntimeError("GLoRIA not downloaded")

    ckpt = torch.load(args.pretrain_path)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = gloria.builder.build_gloria_model(cfg).to(device)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model

