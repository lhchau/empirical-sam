import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime

import os
import argparse
import yaml
import wandb
import pprint
import shutil
from torch.utils.tensorboard import SummaryWriter

from sam_hessian.models import *
from sam_hessian.utils import *
from sam_hessian.dataloader import *
from sam_hessian.scheduler import *
from sam_hessian.optimizer import *
from sam_hessian.utils.pyhessian import get_eigen_hessian_plot


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
parser.add_argument('--rho', default=None, type=float, help='SAM rho')
parser.add_argument('--wd', default=None, type=float, help='Weight decay')
parser.add_argument('--model_name', default=None, type=str, help='Model name')
parser.add_argument('--opt_name', default=None, type=str, help='Optimization name')
parser.add_argument('--adaptive', default=None, type=bool, help='ASAM')
parser.add_argument('--project_name', default=None, type=str, help='Wandb Project name')
parser.add_argument('--framework_name', default=None, type=str, help='Logging Framework')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)
    cfg = override_cfg(cfg, args)
    pprint.pprint(cfg)
seed = cfg['trainer'].get('seed', 42)
initialize(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0

EPOCHS = cfg['trainer']['epochs'] 

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']
if framework_name == 'tensorboard':
    writer = SummaryWriter(os.path.join('runs', logging_name))
elif framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)

logging_dict = {}
################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(**cfg['dataloader'])
try:
    num_classes = len(classes)
except:
    num_classes = classes

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(
    **cfg['model'],
    num_classes=num_classes
)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]["model_name"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss().to(device)
sch = cfg['trainer'].get('sch', None)
optimizer = get_optimizer(
    net, 
    **cfg['optimizer']
)
scheduler = get_scheduler(
    optimizer, 
    **cfg['scheduler']
)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    try: 
        for epoch in range(start_epoch, start_epoch+EPOCHS):
            print('\nEpoch: %d' % epoch)
            loop_one_epoch(
                dataloader=train_dataloader,
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                logging_dict=logging_dict,
                epoch=epoch,
                loop_type='train',
                logging_name=logging_name
            )
            best_acc = loop_one_epoch(
                dataloader=val_dataloader,
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                logging_dict=logging_dict,
                epoch=epoch,
                loop_type='val',
                logging_name=logging_name,
                best_acc=best_acc
            )
            scheduler.step()
            
            if framework_name == 'tensorboard':
                for key, value in logging_dict.items():
                    if not isinstance(key, str):
                        writer.add_scalar(key[0], value[0], global_step=key[1] + epoch*value[1])
                    else:
                        writer.add_scalar(key, value, global_step=epoch)
            elif framework_name == 'wandb':
                tmp_dict = {}
                for key, value in logging_dict.items():
                    if not isinstance(key, str): tmp_dict[key[0].lower()] = value[0]
                    else: tmp_dict[key.lower()] = value
                wandb.log(tmp_dict)
        
        logging_dict = {}
        loop_one_epoch(
            dataloader=test_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='test',
            logging_name=logging_name
        )
        
        if framework_name == 'tensorboard':
            for key, value in logging_dict.items():
                if not isinstance(key, str):
                    writer.add_scalar(key[0], value[0], global_step=key[1] + epoch*value[1])
                else:
                    writer.add_scalar(key, value, global_step=epoch)
        elif framework_name == 'wandb':
            tmp_dict = {}
            for key, value in logging_dict.items():
                if not isinstance(key, str): tmp_dict[key[0].lower()] = value[0]
                else: tmp_dict[key.lower()] = value
            wandb.log(tmp_dict)
            
            figure = get_eigen_hessian_plot(
                name=logging_name, 
                net=net,
                criterion=criterion,
                dataloader=train_dataloader
            )
            wandb.log({'train/top5_eigenvalue_density': wandb.Image(figure)})
        
    except KeyboardInterrupt as e:
        save_dir = os.path.join('checkpoint', logging_name)
        logging_dir = os.path.join('runs', logging_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        print(f"Error: {e}")