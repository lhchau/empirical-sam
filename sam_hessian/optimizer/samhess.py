import torch
import numpy as np


class SAMHESS(torch.optim.Optimizer):
    def __init__(self, params, adaptive=False, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAMHESS, self).__init__(params, defaults)
        self.beta2 = 0.9
        self.k = 10
        self.eps = 1e-8
        self.state['step'] = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):   
        self.state['step'] += 1
        step = self.state['step']
        
        if (step + 1) % self.k == 0 or step == 1:
            params = []
            grads = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad)
                        
            hut_traces = self.get_trace(params, grads)
            
            for group in self.param_groups:
                for p, hut_trace in zip(group['params'], hut_traces):
                    if p.grad is None: continue
                    param_state = self.state[p]
                    
                    if 'exp_hessian_diag_sq' not in param_state:
                        param_state['exp_hessian_diag_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    param_state['exp_hessian_diag_sq'].mul_(self.beta2).addcmul_(
                        hut_trace, hut_trace.conj(), value=1 - self.beta2
                    ) 
                    param_state['exp_hessian_diag'] = param_state['exp_hessian_diag_sq'].sqrt()
            self.hessian_norm = self._grad_norm(by='exp_hessian_diag')
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                param_state['d_t'] = p.grad.div(param_state['exp_hessian_diag'].sqrt().add(self.eps))
        if (step + 1) % 352:
            self.weight_norm = self._weight_norm()
            self.first_grad_norm = self._grad_norm()
             
        self.d_t_grad_norm = self._grad_norm('d_t')
        for group in self.param_groups:
            scale = group['rho'] / (self.d_t_grad_norm + self.eps)
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                
                param_state['e_w'] = e_w.clone()
        
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        step = self.state['step']
        if (step + 1) % 352:
            self.second_grad_norm = self._grad_norm()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                param_state = self.state[p]
                
                d_p = p.grad.data
                
                p.sub_(param_state['e_w'])  # get back to "w" from "w + e(w)"
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                    
                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)
                
                p.add_(param_state['exp_avg'], alpha=-step_size)
                
        if zero_grad: self.zero_grad()
        
    def get_trace(self, params, grads):
        """Get an estimate of Hessian Trace.
        This is done by computing the Hessian vector product with a random
        vector v at the current gradient point, to estimate Hessian trace by
        computing the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                msg = (
                    "Gradient tensor {:} does not have grad_fn. When "
                    "calling loss.backward(), make sure the option "
                    "create_graph is set to True."
                )
                raise RuntimeError(msg.format(i))

        v = [
            2
            * torch.randint_like(
                p, high=2, memory_format=torch.preserve_format
            )
            - 1
            for p in params
        ]

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        hvs = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces
                # the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        return hutchinson_trace

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def _grad_norm(self, by=None):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if by is None:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
        else:
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * self.state[p][by]).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm
    
    @torch.no_grad()
    def _weight_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
