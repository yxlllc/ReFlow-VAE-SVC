import torch
try:
    import torch_musa
except ImportError:
    pass
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class Bi_RectifiedFlow(nn.Module):
    def __init__(self, 
                velocity_fn,
                spec_min=-12, 
                spec_max=2):
        super().__init__()
        self.velocity_fn = velocity_fn
        self.spec_min = spec_min
        self.spec_max = spec_max
    
    def reflow_loss(self, x_1, x_0, t, cond=None, loss_type='l2_lognorm'):
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond=cond)
        
        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / ( 1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()

        return loss
    
    def sample_euler(self, x, t, dt, cond=None):
        x += self.velocity_fn(x, 1000 * t, cond=cond) * dt
        t += dt
        return x, t
        
    def sample_rk4(self, x, t, dt, cond=None):
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, 1000 * (t + 0.5 * dt), cond=cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, 1000 * (t + 0.5 * dt), cond=cond)
        k_4 = self.velocity_fn(x + k_3 * dt, 1000 * (t + dt), cond=cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t
        
    def sample_heun(self, x, t, dt, cond=None):
        # Predict
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred = x + k_1 * dt
        t_pred = t + dt
        # Correct
        k_2 = self.velocity_fn(x_pred, 1000 * t_pred, cond=cond)
        x += (k_1 + k_2) / 2 * dt
        t += dt
        return x, t

    def sample_PECECE(self, x, t, dt, cond=None):
        # Predict1
        k_1 = self.velocity_fn(x, 1000 * t, cond=cond)
        x_pred1 = x + k_1 * dt
        t_pred1 = t + dt
        # Correct1
        k_2 = self.velocity_fn(x_pred1, 1000 * t_pred1, cond=cond)
        x_corr1 = x + (k_1 + k_2) / 2 * dt
        # Predict2
        k_3 = self.velocity_fn(x_corr1, 1000 * (t + dt), cond=cond)
        x_pred2 = x_corr1 + k_3 * dt
        # Correct2
        k_4 = self.velocity_fn(x_pred2, 1000 * (t + 2*dt), cond=cond)
        x += (k_3 + k_4) / 2 * dt
        t += dt
        return x, t
        
    def forward(self, 
                infer=True,
                x_start=None,
                x_end=None,
                cond=None,
                t_start=0.0,
                t_end=1.0,
                infer_step=10,
                method='euler',
                use_tqdm=True):
        if cond is not None:
            cond = cond.transpose(1, 2) # [B, H, T]
        if not infer:
            x_0 = x_start.transpose(1, 2).unsqueeze(1) # [B, 1, M, T]
            x_1 = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1)  # [B, 1, M, T]
            t = torch.rand(x_0.shape[0], device=x_0.device)
            t = torch.clip(t, 1e-7, 1-1e-7)
            return self.reflow_loss(x_1, x_0, t, cond=cond)
        else:
            # initial condition and step size of the ODE      
            if t_start < 0.0:
                t_start = 0.0
            elif t_start > 1.0:
                t_start = 1.0
            if t_end < 0.0:
                t_end = 0.0
            elif t_end > 1.0:
                t_end = 1.0
            assert t_start < t_end
            
            if x_start is not None and x_end is None:
                x = x_start.transpose(1, 2).unsqueeze(1) # [B, 1, M, T]
                t = torch.full((x_start.shape[0],), t_start, device=x_start.device)
                dt = (t_end - t_start) / infer_step
            elif x_start is None and x_end is not None:
                x = self.norm_spec(x_end).transpose(1, 2).unsqueeze(1) # [B, 1, M, T]
                t = torch.full((x_end.shape[0],), t_end, device=x_end.device)
                dt = -(t_end - t_start) / infer_step
            
            # sampling
            if method == 'euler':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_euler(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_euler(x, t, dt, cond=cond)
            
            elif method == 'rk4':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_rk4(x, t, dt, cond=cond)
            
            elif method == 'heun':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_heun(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_heun(x, t, dt, cond=cond)
                        
            elif method == 'PECECE':
                if use_tqdm:
                    for i in tqdm(range(infer_step), desc='sample time step', total=infer_step):
                        x, t = self.sample_PECECE(x, t, dt, cond=cond)
                else:
                    for i in range(infer_step):
                        x, t = self.sample_PECECE(x, t, dt, cond=cond)
            
            else:
                raise NotImplementedError(method)
                
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
            
            if dt > 0:
                return self.denorm_spec(x)
            else:
                return x
                
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min