from .cifar10 import get_cifar10
from .cifar100 import get_cifar100
from .tiny_imagenet import get_tiny_imagenet


# Data
def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=16,
    split=(0.9, 0.1)    
):
    print('==> Preparing data..')

    if data_name == "cifar10":
        return get_cifar10(
            batch_size,
            num_workers,
            split
        )
    elif data_name == "cifar100":
        return get_cifar100(
            batch_size,
            num_workers,
            split
        )
    elif data_name == "tiny_imagenet":
        return get_tiny_imagenet(
            batch_size,
            num_workers
        )

