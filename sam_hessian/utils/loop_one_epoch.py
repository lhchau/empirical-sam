import torch
import os
from .utils import *
from .bypass_bn import *

def loop_one_epoch(
    dataloader,
    net,
    criterion,
    optimizer,
    device,
    logging_dict,
    epoch,
    loop_type='train',
    logging_name=None,
    best_acc=0
    ):
    loss = 0
    correct = 0
    total = 0
    
    if loop_type == 'train': 
        net.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            opt_name = type(optimizer).__name__
            if opt_name == 'SGD':
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()
                optimizer.step(zero_grad=True)
            elif opt_name == 'SAMHESSIAN' or opt_name == 'SAMHESS':
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward(create_graph=True)        
                optimizer.first_step(zero_grad=True)
                # Zero the gradients explicitly
                for param in net.parameters():
                    param.grad = None
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
            else:
                enable_running_stats(net)  # <- this is the important line
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)
                first_loss.backward()        
                optimizer.first_step(zero_grad=True)
                
                disable_running_stats(net)  # <- this is the important line
                criterion(net(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                
            try: 
                logging_dict[(f'{loop_type.title()}/hessian_norm', batch_idx)] = [optimizer.hessian_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/d_t_grad_norm', batch_idx)] = [optimizer.d_t_grad_norm, len(dataloader)]
            except: pass
                     
            try: 
                logging_dict[(f'{loop_type.title()}/first_grad_norm', batch_idx)] = [optimizer.first_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/second_grad_norm', batch_idx)] = [optimizer.second_grad_norm, len(dataloader)]
            except: pass
            
            try: 
                logging_dict[(f'{loop_type.title()}/weight_norm', batch_idx)] = [optimizer.weight_norm, len(dataloader)]
            except: pass
            
            with torch.no_grad():
                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (loss_mean, acc, correct, total))
    else:
        if loop_type == 'test':
            print('==> Resuming from best checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            load_path = os.path.join('checkpoint', logging_name, 'ckpt_best.pth')
            checkpoint = torch.load(load_path)
            net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                first_loss = criterion(outputs, targets)

                loss += first_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss_mean = loss/(batch_idx+1)
                acc = 100.*correct/total
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (loss_mean, acc, correct, total))
        if loop_type == 'val':
            if acc > best_acc:
                print('Saving best checkpoint ...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'loss': loss,
                    'epoch': epoch
                }
                save_path = os.path.join('checkpoint', logging_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
                best_acc = acc
            logging_dict[f'{loop_type.title()}/best_acc'] = best_acc
            
    logging_dict[f'{loop_type.title()}/loss'] = loss_mean
    logging_dict[f'{loop_type.title()}/acc'] = acc

    if loop_type == 'val': 
        return best_acc