from .sam import SAM
from .samhess import SAMHESS

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperparameter={}):
    if opt_name == 'sam':
        return SAM(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'samhess':
        return SAMHESS(
            net.parameters(), 
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")