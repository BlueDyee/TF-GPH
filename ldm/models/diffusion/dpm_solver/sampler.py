"""SAMPLING ONLY."""
import torch
from tqdm import tqdm
# from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from scripts.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               batch_size=1,
               conditioning=None,
               inv_emb=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               t_start=None,
               t_end=None,
               DPMencode=False,
               order=3,
               width=None,
               height=None,
               ref=False,
               top=None, 
               left=None, 
               bottom=None, 
               right=None,
               segmentation_map=None,
               param=None,
               target_height=None, 
               target_width=None,
               center_row_rm=None,
               center_col_rm=None,
               tau_a=0.4,
               tau_b=0.8,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {steps}')

        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        if DPMencode:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )

            dpm_solver = DPM_Solver(model_fn, ns)
            
            
            data, _ = self.low_order_sample(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for step in tqdm(range(order, steps + 1), desc='DPM++ inversion (adding noise) Z0-->ZT'):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode)
                     
            return data['x'].to(device), None

        else:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )

            dpm_solver = DPM_Solver(model_fn, ns)

            x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=order,
                            skip_type='time_uniform', method='multistep')
            return x_sample
            
    
    def low_order_sample(self, x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=False, controller=None, inject=False, ref_init=None):
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N if t_end is None else t_end
        t_T = dpm_solver.noise_schedule.T if t_start is None else t_start
        
        total_controller = []
        assert steps >= order
        timesteps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=t_T, t_0=t_0, N=steps, device=device, DPMencode=DPMencode)
        assert timesteps.shape[0] - 1 == steps
        with torch.no_grad():
            vec_t = timesteps[0].expand((x.shape[0]))
            model_prev_list = [dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                   controller=[controller[0][0], controller[1][0], controller[2][0]] if isinstance(controller, list) else controller, 
                                                   inject=inject, ref_init=ref_init)]
            
            total_controller.append(controller)
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = timesteps[init_order].expand(x.shape[0])
                x = dpm_solver.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, vec_t, init_order,
                                                           solver_type='dpmsolver', DPMencode=DPMencode)
                model_prev_list.append(dpm_solver.model_fn(x, vec_t, DPMencode=DPMencode, 
                                                           controller=[controller[0][init_order], controller[1][init_order], controller[2][init_order]] if isinstance(controller, list) else controller,
                                                           inject=inject, ref_init=ref_init))
                total_controller.append(controller)
                t_prev_list.append(vec_t)
        
        return {'x': x, 'model_prev_list': model_prev_list, 't_prev_list': t_prev_list, 'timesteps':timesteps}, total_controller
    