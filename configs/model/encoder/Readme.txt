blocks:
- input_dim: 128
  pre_config: 'ssm/preprocess/residual'
  ssm_config: 'ssm/companion_preprocess'
  mlp_config: 'mlp/default'
  skip_connection: true
  skip_preprocess: false
- input_dim: 128
  pre_config: 'ssm/preprocess/none'
  ssm_config: 'ssm/companion' # change the ssm here for using different matrices:- default is comnpanion
  mlp_config: 'mlp/default'
  skip_connection: true
  skip_preprocess: true