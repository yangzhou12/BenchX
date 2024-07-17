import copy
from .config import ex
from .models.ptunifier_module import PTUnifierTransformerSS


@ex.main
def main(_config):
    _config = copy.deepcopy(_config)
    model = PTUnifierTransformerSS(_config)
    return model


def load_ptunifier(ckpt, **kwargs):
    exp_result = ex.run(named_configs=['clip16', 'text_roberta'],
                        config_updates={'load_path': ckpt,
                                        'tokenizer': 'allenai/biomed_roberta_base',
                                        'pseudo_vision_token_pool_size': 2048,
                                        'pseudo_langauge_token_pool_size': 2048,
                                        **kwargs})
    model = exp_result.result
    return model