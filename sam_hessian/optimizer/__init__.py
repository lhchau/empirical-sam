from .sam import SAM
from .samdirection import SAMDIRECTION
from .sammagnitude import SAMMAGNITUDE
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
    elif opt_name == 'samdirection':
        return SAMDIRECTION(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'sammagnitude':
        return SAMMAGNITUDE(
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