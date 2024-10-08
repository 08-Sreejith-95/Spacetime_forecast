from omegaconf import OmegaConf
from os.path import join





#the elements of components of ssm's inside the encoder and decoder should be called from main environment. so this function is defined to get their configs
def init_encoder_decoder_config(config, main_config_dir):
    for ix, _config in enumerate(config['blocks']):
        # Load preprocess kernel configs
        c_path = join(main_config_dir, f"{_config['pre_config']}.yaml")
        _config['pre_config'] = OmegaConf.load(c_path)
        # Load SSM kernel configs
        c_path = join(main_config_dir, f"{_config['ssm_config']}.yaml")
        _config['ssm_config'] = OmegaConf.load(c_path)
        # Load MLP configs
        c_path = join(main_config_dir, f"{_config['mlp_config']}.yaml")
        _config['mlp_config'] = OmegaConf.load(c_path)
    return config
  
class TrainConfigs():
  def __init__(self, train_configs):
    for k, v in train_configs.items():
      setattr(self,k,v)