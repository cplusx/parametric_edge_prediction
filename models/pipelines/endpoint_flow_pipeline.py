from dataclasses import dataclass
from typing import List, Optional

import torch

from misc_utils.endpoint_flow_utils import scale_points_from_flow, select_predicted_points
from models.endpoint_flow_matching import EndpointFlowMatchingModel

try:
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - exercised on cluster env
    raise ImportError(
        'diffusers is required for EndpointFlowPipeline. Install it in the runtime environment.'
    ) from exc


@dataclass
class EndpointFlowPipelineOutput(BaseOutput):
    points: torch.FloatTensor
    presence_probs: torch.FloatTensor
    selected_points: List[torch.FloatTensor]


class EndpointFlowPipeline(DiffusionPipeline):
    model_cpu_offload_seq = 'model'

    def __init__(self, model: EndpointFlowMatchingModel, scheduler: FlowMatchEulerDiscreteScheduler) -> None:
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        presence_threshold: float = 0.5,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> EndpointFlowPipelineOutput:
        device = images.device
        batch_size = images.shape[0]
        latents = torch.randn(
            (batch_size, self.model.num_points, 2),
            generator=generator,
            device=device,
            dtype=images.dtype,
        )
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        final_presence_logits = None
        for timestep in self.scheduler.timesteps:
            if guidance_scale != 1.0:
                model_input = torch.cat([latents, latents], dim=0)
                image_input = torch.cat([torch.zeros_like(images), images], dim=0)
                timestep_input = timestep.expand(model_input.shape[0]).to(device=device, dtype=images.dtype)
                model_output = self.model.predict(model_input, timestep_input, image_input)
                uncond_velocity, cond_velocity = model_output.sample.chunk(2, dim=0)
                uncond_presence, cond_presence = model_output.presence_logits.chunk(2, dim=0)
                velocity = uncond_velocity + guidance_scale * (cond_velocity - uncond_velocity)
                final_presence_logits = uncond_presence + guidance_scale * (cond_presence - uncond_presence)
            else:
                timestep_input = timestep.expand(batch_size).to(device=device, dtype=images.dtype)
                model_output = self.model.predict(latents, timestep_input, images)
                velocity = model_output.sample
                final_presence_logits = model_output.presence_logits
            latents = self.scheduler.step(velocity, timestep, latents).prev_sample

        points = scale_points_from_flow(latents).clamp(0.0, 1.0)
        presence_probs = final_presence_logits.sigmoid()
        selected_points = select_predicted_points(points, presence_probs, threshold=presence_threshold)
        if not return_dict:
            return points, presence_probs, selected_points
        return EndpointFlowPipelineOutput(
            points=points,
            presence_probs=presence_probs,
            selected_points=selected_points,
        )
