from omegaconf import OmegaConf

alpha_config = """
method: companion
kwargs:
  model_dim: 128
  n_kernels: 128
  kernel_dim: 64
  kernel_repeat: 1
  n_heads: 1
  head_dim: 1
  kernel_weights: null
  kernel_init: normal
  kernel_train: true
  skip_connection: true
  norm_order: 1
"""
alpha_conf = OmegaConf.create(alpha_config)
alpha_conf.k = 32
alpha_conf.n = 16
OmegaConf.save(alpha_conf, '/Users/sreejithkk/Thesis_code/spacetime/configs/model/ssm/alpha_ssm.yaml')
