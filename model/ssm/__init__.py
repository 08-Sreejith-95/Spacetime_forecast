from .companion import CompanionSSM, AlphaSSM
from .shift import ShiftSSM
from .closed_loop import ClosedLoopCompanionSSM, ClosedLoopShiftSSM , ClosedLoopAlphaSSM

#todo:- Add our custom SSM here
def init_ssm(config):
    supported_methods = ['companion', 'closed_loop_companion',
                         'shift', 'closed_loop_shift', 'alpha_ssm', 'closed_loop_alpha'] # to_do add one more ssm:- Custom SSM with alpha_A 
    if config['method'] == 'companion':
        ssm = CompanionSSM
    elif config['method'] == 'closed_loop_companion':
        ssm = ClosedLoopCompanionSSM
    elif config['method'] == 'shift':
        ssm = ShiftSSM
    elif config['method'] == 'closed_loop_shift':
        ssm = ClosedLoopShiftSSM
    elif config['method'] == 'alpha_ssm': #added Alpha_SSM 
        print("------------------------Initializing ALPHA--------------------------")
        ssm = AlphaSSM
    elif config['method'] == 'closed_loop_alpha':
        print("---------------------Initializing Closed_loop_Alpha-----------------")
        ssm = ClosedLoopAlphaSSM
    else:
        raise NotImplementedError(
            f"SSM config method {config['method']} not implemented! Please choose from {supported_methods}")
    return ssm(**config['kwargs'])