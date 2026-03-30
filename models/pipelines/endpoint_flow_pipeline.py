from dataclasses import dataclass
from typing import List, Optional

import torch

from misc_utils.endpoint_flow_utils import sample_uniform_points, scale_points_from_flow, scale_points_to_flow, select_predicted_points
from models.endpoint_flow_matching import EndpointFlowMatchingModel

try:
    from diffusers import DiffusionPipeline
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - exercised on cluster env
    raise ImportError(
        'diffusers is required for EndpointFlowPipeline. Install it in the runtime environment.'
    ) from exc


@dataclass
class EndpointFlowPipelineOutput(BaseOutput):
    points: torch.FloatTensor
    selected_points: List[torch.FloatTensor]


class EndpointFlowPipeline(DiffusionPipeline):
    model_cpu_offload_seq = 'model'

    def __init__(self, model: EndpointFlowMatchingModel) -> None:
        super().__init__()
        self.register_modules(model=model)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> EndpointFlowPipelineOutput:
        device = images.device
        batch_size = images.shape[0]
        source_points_ext = sample_uniform_points(
            batch_size,
            self.model.num_points,
            device=device,
            dtype=images.dtype,
            generator=generator,
        )
        latents = scale_points_to_flow(source_points_ext)
        tau_schedule = torch.linspace(0.0, 1.0, steps=int(num_inference_steps) + 1, device=device, dtype=images.dtype)
        for step_idx in range(int(num_inference_steps)):
            tau = tau_schedule[step_idx]
            next_tau = tau_schedule[step_idx + 1]
            timestep = tau * float(max(self.model.num_train_timesteps - 1, 1))
            if guidance_scale != 1.0:
                model_input = torch.cat([latents, latents], dim=0)
                image_input = torch.cat([torch.zeros_like(images), images], dim=0)
                timestep_input = timestep.expand(model_input.shape[0]).to(device=device, dtype=images.dtype)
                model_output = self.model.predict(model_input, timestep_input, image_input)
                uncond_velocity, cond_velocity = model_output.sample.chunk(2, dim=0)
                velocity = uncond_velocity + guidance_scale * (cond_velocity - uncond_velocity)
            else:
                timestep_input = timestep.expand(batch_size).to(device=device, dtype=images.dtype)
                model_output = self.model.predict(latents, timestep_input, images)
                velocity = model_output.sample
            step_size = (next_tau - tau).to(dtype=latents.dtype)
            latents = latents + step_size * velocity

        points = scale_points_from_flow(latents).clamp(0.0, 1.0)
        selected_points = select_predicted_points(points)
        if not return_dict:
            return points, selected_points
        return EndpointFlowPipelineOutput(
            points=points,
            selected_points=selected_points,
        )
