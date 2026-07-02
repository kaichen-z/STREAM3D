from typing import *
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


_EMPTY_STRUCTURE_RETRY_SEED_OFFSETS = (1, -1, 2)


class MultiImageFusionWeightSource(StrEnum):
    AVERAGE = "average"
    JOINT_ATTENTION_MASS = "joint_attention_mass"
    MASS_RELATIVE = "mass_relative"


@dataclass(frozen=True)
class MultiImageFusionConfig:
    weight_source: MultiImageFusionWeightSource | str = MultiImageFusionWeightSource.AVERAGE
    attention_layer: int = 6
    q_chunk_size: int = 128
    patch_start: int = 1
    patch_end: int | None = None
    jam_alpha: float = 1.0
    jam_kappa: float = 1.0
    min_weight: float = 0.01

    def __post_init__(self) -> None:
        weight_source = self.weight_source
        if str(weight_source) == "uniform":
            weight_source = MultiImageFusionWeightSource.AVERAGE
        object.__setattr__(self, "weight_source", MultiImageFusionWeightSource(weight_source))
        object.__setattr__(self, "attention_layer", int(self.attention_layer))
        object.__setattr__(self, "q_chunk_size", int(self.q_chunk_size))
        object.__setattr__(self, "patch_start", int(self.patch_start))
        object.__setattr__(
            self,
            "patch_end",
            None if self.patch_end is None else int(self.patch_end),
        )
        object.__setattr__(self, "jam_alpha", float(self.jam_alpha))
        object.__setattr__(self, "jam_kappa", float(self.jam_kappa))
        object.__setattr__(self, "min_weight", float(self.min_weight))
        if self.q_chunk_size <= 0:
            raise ValueError("q_chunk_size must be positive")
        if self.jam_alpha <= 0:
            raise ValueError("jam_alpha must be positive")


@dataclass
class FusionEvidence:
    patch_mass: torch.Tensor | None = None
    entropy_confidence: torch.Tensor | None = None
    num_records: int = 0


class FusionEvidenceCollector:
    def __init__(self, config: MultiImageFusionConfig):
        self.config = config
        self.evidence = FusionEvidence()

    def record(self, module_name: str, module: Any, x: Any, context: Any) -> None:
        if module_name != f"blocks.{self.config.attention_layer}.cross_attn":
            return
        if context is None:
            return
        if hasattr(x, "feats"):
            scores = self._sparse_attention_scores(module, x, context)
        else:
            scores = self._dense_attention_scores(module, x, context)
        patch_mass, entropy_confidence = self._patch_mass_and_entropy(scores)
        if self.evidence.patch_mass is None:
            self.evidence.patch_mass = patch_mass
            self.evidence.entropy_confidence = entropy_confidence
        else:
            self.evidence.patch_mass += patch_mass
            self.evidence.entropy_confidence += entropy_confidence
        self.evidence.num_records += 1

    def _dense_attention_scores(self, module: Any, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if not isinstance(context, torch.Tensor):
            raise TypeError(f"Expected dense context tensor, got {type(context)}")
        q = module.to_q(x)
        kv = module.to_kv(context)
        batch_size, num_queries, _ = q.shape
        num_context_tokens = context.shape[1]
        q = q.reshape(batch_size, num_queries, module.num_heads, -1)
        kv = kv.reshape(batch_size, num_context_tokens, 2, module.num_heads, -1)
        k, _ = kv.unbind(dim=2)
        if module.qk_rms_norm:
            q = module.q_rms_norm(q)
            k = module.k_rms_norm(k)
        q = q.float().permute(0, 2, 1, 3)
        k = k.float().permute(0, 2, 1, 3)
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    def _sparse_attention_scores(self, module: Any, x: Any, context: Any) -> torch.Tensor:
        q = module._linear(module.to_q, x)
        q = module._reshape_chs(q, (module.num_heads, -1))
        kv = module._linear(module.to_kv, context)
        kv = module._fused_pre(kv, num_fused=2)
        if module.qk_rms_norm:
            q = module.q_rms_norm(q)
            k, _ = kv.unbind(dim=-3)
            k = module.k_rms_norm(k)
        else:
            k, _ = kv.unbind(dim=-3)
        q_feats = q.feats.float().permute(1, 0, 2)
        if hasattr(k, "feats"):
            k_feats = k.feats.float().permute(1, 0, 2)
        else:
            k_feats = k.float().squeeze(0).permute(1, 0, 2)
        return torch.einsum("hqd,htd->hqt", q_feats, k_feats) / math.sqrt(q_feats.shape[-1])

    def _patch_mass_and_entropy(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = torch.softmax(scores.to(torch.float32), dim=-1)
        patch_start = min(int(self.config.patch_start), int(attention.shape[-1]) - 1)
        patch_end = int(attention.shape[-1]) if self.config.patch_end is None else int(self.config.patch_end)
        patch_end = min(max(patch_end, patch_start + 1), int(attention.shape[-1]))
        patch_attention = attention[..., patch_start:patch_end]
        patch_mass = patch_attention.sum(dim=-1)
        distribution = patch_attention / patch_mass.unsqueeze(-1).clamp(min=1e-10)
        num_patches = int(patch_attention.shape[-1])
        if num_patches == 1:
            entropy_confidence = torch.ones_like(patch_mass)
        else:
            entropy = -(distribution * distribution.clamp_min(1e-10).log()).sum(dim=-1)
            entropy_confidence = (1.0 - entropy / math.log(num_patches)).clamp(min=0.0, max=1.0)
        while patch_mass.dim() > 1:
            patch_mass = patch_mass.mean(dim=0)
            entropy_confidence = entropy_confidence.mean(dim=0)
        return (
            patch_mass.detach().cpu().to(torch.float32).flatten().contiguous(),
            entropy_confidence.detach().cpu().to(torch.float32).flatten().contiguous(),
        )


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @staticmethod
    def _num_conditions(cond) -> int:
        if isinstance(cond, torch.Tensor):
            return int(cond.shape[0])
        if isinstance(cond, (list, tuple)):
            return len(cond)
        return 1

    @staticmethod
    def _select_condition(cond, index: int):
        if isinstance(cond, torch.Tensor):
            return cond[index:index + 1]
        if isinstance(cond, tuple):
            return (cond[index],)
        if isinstance(cond, list):
            return [cond[index]]
        return cond

    @staticmethod
    def _canonical_prediction(prediction):
        if isinstance(prediction, (list, tuple)) and len(prediction) == 1:
            return prediction[0]
        return prediction

    @staticmethod
    def _scale_prediction(prediction, scale: float):
        if isinstance(prediction, torch.Tensor):
            return prediction * scale
        if isinstance(prediction, SparseTensor):
            return prediction.replace(prediction.feats * scale)
        if isinstance(prediction, tuple):
            return tuple(Trellis2ImageTo3DPipeline._scale_prediction(item, scale) for item in prediction)
        if isinstance(prediction, list):
            return [Trellis2ImageTo3DPipeline._scale_prediction(item, scale) for item in prediction]
        raise TypeError(f"Unsupported prediction type for scaling: {type(prediction)}")

    @staticmethod
    def _add_predictions(left, right):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left + right
        if isinstance(left, SparseTensor) and isinstance(right, SparseTensor):
            return left.replace(left.feats + right.feats)
        if isinstance(left, tuple) and isinstance(right, tuple):
            if len(left) != len(right):
                raise ValueError(f"Prediction tuple lengths differ: {len(left)} vs {len(right)}")
            return tuple(
                Trellis2ImageTo3DPipeline._add_predictions(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise ValueError(f"Prediction list lengths differ: {len(left)} vs {len(right)}")
            return [
                Trellis2ImageTo3DPipeline._add_predictions(left_item, right_item)
                for left_item, right_item in zip(left, right)
            ]
        raise TypeError(f"Cannot add prediction types {type(left)} and {type(right)}")

    @staticmethod
    def _average_predictions(predictions: list):
        pred = Trellis2ImageTo3DPipeline._canonical_prediction(predictions[0])
        for item in predictions[1:]:
            pred = Trellis2ImageTo3DPipeline._add_predictions(
                pred,
                Trellis2ImageTo3DPipeline._canonical_prediction(item),
            )
        return Trellis2ImageTo3DPipeline._scale_prediction(pred, 1.0 / len(predictions))

    @staticmethod
    def _normalize_fusion_confidence(
        confidence_by_view: list[torch.Tensor],
        config: MultiImageFusionConfig,
    ) -> list[torch.Tensor]:
        confidence = torch.stack(
            [item.detach().cpu().to(torch.float32).flatten().contiguous() for item in confidence_by_view],
            dim=0,
        ).clamp(min=0.0)
        if config.jam_alpha != 1.0:
            confidence = confidence.pow(config.jam_alpha)
        denom = confidence.sum(dim=0, keepdim=True)
        uniform = torch.full_like(confidence, 1.0 / int(confidence.shape[0]))
        weights = torch.where(denom > 1e-10, confidence / denom.clamp(min=1e-10), uniform)
        if config.min_weight > 0:
            weights = weights.clamp(min=config.min_weight)
            weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-10)
        return [weights[index].contiguous() for index in range(int(weights.shape[0]))]

    @staticmethod
    def _fusion_weights(
        evidence_by_view: list[FusionEvidence],
        config: MultiImageFusionConfig,
    ) -> list[torch.Tensor]:
        if config.weight_source is MultiImageFusionWeightSource.AVERAGE:
            raise ValueError("Average fusion does not use attention weights")
        for evidence in evidence_by_view:
            if evidence.patch_mass is None or evidence.entropy_confidence is None:
                raise RuntimeError(
                    f"No fusion evidence was collected at layer {config.attention_layer}."
                )

        patch_mass = [evidence.patch_mass / max(1, evidence.num_records) for evidence in evidence_by_view]
        entropy_confidence = [
            evidence.entropy_confidence / max(1, evidence.num_records)
            for evidence in evidence_by_view
        ]
        if config.weight_source is MultiImageFusionWeightSource.JOINT_ATTENTION_MASS:
            confidence = [
                (mass * entropy).clamp(min=0.0)
                for mass, entropy in zip(patch_mass, entropy_confidence)
            ]
        elif config.weight_source is MultiImageFusionWeightSource.MASS_RELATIVE:
            mass_stack = torch.stack([mass.to(torch.float32) for mass in patch_mass], dim=0)
            entropy_stack = torch.stack(
                [entropy.to(torch.float32) for entropy in entropy_confidence],
                dim=0,
            )
            relative_mass = 1.0 + config.jam_kappa * (
                mass_stack - mass_stack.mean(dim=0, keepdim=True)
            )
            confidence = [
                (relative_mass[index].clamp(min=1e-6) * entropy_stack[index]).contiguous()
                for index in range(int(relative_mass.shape[0]))
            ]
        else:
            raise ValueError(f"Unsupported fusion weight source: {config.weight_source}")
        return Trellis2ImageTo3DPipeline._normalize_fusion_confidence(confidence, config)

    @staticmethod
    def _weighted_predictions(
        predictions: list,
        evidence_by_view: list[FusionEvidence],
        config: MultiImageFusionConfig,
    ):
        weights = Trellis2ImageTo3DPipeline._fusion_weights(evidence_by_view, config)
        first = predictions[0]
        if isinstance(first, torch.Tensor):
            fused = torch.zeros_like(first)
            num_queries = int(first[0].numel() // first.shape[1])
            for pred, weight in zip(predictions, weights):
                if int(weight.numel()) != num_queries:
                    raise ValueError(
                        f"Fusion weight length {int(weight.numel())} does not match dense prediction queries {num_queries}."
                    )
                view_weight = weight.to(device=pred.device, dtype=pred.dtype).view(
                    1,
                    1,
                    *pred.shape[2:],
                )
                fused = fused + pred * view_weight
            return fused

        if isinstance(first, SparseTensor):
            fused_feats = torch.zeros_like(first.feats)
            for pred, weight in zip(predictions, weights):
                if int(weight.numel()) != int(pred.feats.shape[0]):
                    raise ValueError(
                        f"Fusion weight length {int(weight.numel())} does not match sparse prediction tokens {int(pred.feats.shape[0])}."
                    )
                view_weight = weight.to(device=pred.device, dtype=pred.dtype).view(-1, 1)
                fused_feats = fused_feats + pred.feats * view_weight
            return first.replace(fused_feats)

        raise TypeError(f"Unsupported prediction type for weighted fusion: {type(first)}")

    @staticmethod
    def _normalize_fusion_config(
        config: MultiImageFusionConfig | dict | None,
        stage_name: str | None = None,
    ) -> MultiImageFusionConfig:
        if config is None:
            return MultiImageFusionConfig()
        if isinstance(config, MultiImageFusionConfig):
            return config
        stage_keys = {"sparse_structure", "shape_slat", "tex_slat"}
        base_config = {
            key: value
            for key, value in config.items()
            if key not in stage_keys
        }
        if stage_name is not None and stage_name in config:
            base_config.update(config[stage_name])
        return MultiImageFusionConfig(**base_config)

    @contextmanager
    def collect_fusion_evidence(
        self,
        model: nn.Module,
        collector: FusionEvidenceCollector,
    ):
        from trellis2.modules.attention.modules import MultiHeadAttention
        from trellis2.modules.sparse.attention.modules import SparseMultiHeadAttention

        originals = []
        for module_name, module in model.named_modules():
            if not isinstance(module, (MultiHeadAttention, SparseMultiHeadAttention)):
                continue
            if module._type != "cross":
                continue
            original_forward = module.forward

            def wrapped_forward(self, x, context=None, phases=None, *, _name=module_name, _forward=original_forward):
                collector.record(_name, self, x, context)
                if phases is None:
                    return _forward(x, context=context)
                return _forward(x, context=context, phases=phases)

            module.forward = wrapped_forward.__get__(module, type(module))
            originals.append((module, original_forward))

        if not originals:
            raise RuntimeError(f"No TRELLIS cross-attention modules found in {type(model).__name__}")
        try:
            yield
        finally:
            for module, original_forward in originals:
                module.forward = original_forward

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        fusion_config: MultiImageFusionConfig | dict | None = None,
    ):
        """
        Inject a TRELLIS.2 sampler with multiple image conditions.

        This follows the official TRELLIS1 multi-image API shape while adapting
        TRELLIS.2's guidance_strength/guidance_interval/guidance_rescale sampler
        arguments.
        """
        fusion_config = self._normalize_fusion_config(fusion_config)
        sampler = getattr(self, sampler_name)
        original_inference_model = sampler._inference_model

        if mode == 'stochastic':
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than "
                    f"number of steps for {sampler_name}. This may lead to performance "
                    "degradation.\033[0m"
                )
            cond_indices = (np.arange(num_steps) % num_images).tolist()
            cond_idx_counter = [0]

            def _new_inference_model(self, model, x_t, t, cond=None, **kwargs):
                if cond is None or Trellis2ImageTo3DPipeline._num_conditions(cond) != num_images:
                    return original_inference_model(model, x_t, t, cond, **kwargs)
                cond_idx = cond_indices[cond_idx_counter[0] % len(cond_indices)]
                cond_idx_counter[0] += 1
                cond_i = Trellis2ImageTo3DPipeline._select_condition(cond, cond_idx)
                return original_inference_model(model, x_t, t, cond_i, **kwargs)

        elif mode == 'multidiffusion':
            from .samplers import FlowEulerSampler

            owner = self

            def _positive_prediction(self, model, x_t, t, cond_i, model_kwargs):
                if fusion_config.weight_source is MultiImageFusionWeightSource.AVERAGE:
                    return (
                        FlowEulerSampler._inference_model(
                            self, model, x_t, t, cond_i, **model_kwargs
                        ),
                        None,
                    )
                collector = FusionEvidenceCollector(fusion_config)
                with owner.collect_fusion_evidence(model, collector):
                    pred = FlowEulerSampler._inference_model(
                        self, model, x_t, t, cond_i, **model_kwargs
                    )
                return Trellis2ImageTo3DPipeline._canonical_prediction(pred), collector.evidence

            def _new_inference_model(self, model, x_t, t, cond=None, **kwargs):
                if cond is None or Trellis2ImageTo3DPipeline._num_conditions(cond) != num_images:
                    return original_inference_model(model, x_t, t, cond, **kwargs)

                model_kwargs = dict(kwargs)
                neg_cond = model_kwargs.pop('neg_cond', None)
                guidance_strength = model_kwargs.pop('guidance_strength', None)
                guidance_interval = model_kwargs.pop('guidance_interval', None)
                guidance_rescale = model_kwargs.pop('guidance_rescale', 0.0)

                pred_records = [
                    _positive_prediction(
                        self,
                        model,
                        x_t,
                        t,
                        Trellis2ImageTo3DPipeline._select_condition(cond, view_idx),
                        model_kwargs,
                    )
                    for view_idx in range(num_images)
                ]
                preds = [record[0] for record in pred_records]
                if fusion_config.weight_source is MultiImageFusionWeightSource.AVERAGE:
                    pred_pos = Trellis2ImageTo3DPipeline._average_predictions(preds)
                else:
                    evidence = [record[1] for record in pred_records]
                    pred_pos = Trellis2ImageTo3DPipeline._weighted_predictions(
                        preds,
                        evidence,
                        fusion_config,
                    )

                if guidance_interval is not None and not (guidance_interval[0] <= t <= guidance_interval[1]):
                    return pred_pos
                if guidance_strength is None or guidance_strength == 1 or neg_cond is None:
                    return pred_pos
                if guidance_strength == 0:
                    return _positive_prediction(self, model, x_t, t, neg_cond, model_kwargs)[0]

                if Trellis2ImageTo3DPipeline._num_conditions(neg_cond) == num_images:
                    neg_preds = [
                        _positive_prediction(
                            self,
                            model,
                            x_t,
                            t,
                            Trellis2ImageTo3DPipeline._select_condition(neg_cond, view_idx),
                            model_kwargs,
                        )[0]
                        for view_idx in range(num_images)
                    ]
                    pred_neg = Trellis2ImageTo3DPipeline._average_predictions(neg_preds)
                else:
                    pred_neg = _positive_prediction(self, model, x_t, t, neg_cond, model_kwargs)[0]
                pred = Trellis2ImageTo3DPipeline._add_predictions(
                    Trellis2ImageTo3DPipeline._scale_prediction(pred_pos, guidance_strength),
                    Trellis2ImageTo3DPipeline._scale_prediction(pred_neg, 1 - guidance_strength),
                )

                if guidance_rescale > 0:
                    x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                    x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                    pred = self._xstart_to_pred(x_t, t, x_0)

                return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))
        try:
            yield
        finally:
            sampler._inference_model = original_inference_model

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        fallback_to_max_logit: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded_logits = decoder(z_s)
        decoded = decoded_logits > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        if not bool(decoded.any()):
            if fallback_to_max_logit:
                # Keep downstream sparse tensors non-empty after retry seeds are exhausted.
                return self._fallback_sparse_structure_coords(decoded_logits, resolution)
            return torch.empty((0, 4), dtype=torch.int32, device=decoded.device)
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    @staticmethod
    def _fallback_sparse_structure_coords(
        decoded_logits: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        scores = decoded_logits.detach().float()
        if scores.shape[1] > 1:
            scores = scores.max(dim=1, keepdim=True).values
        if resolution != scores.shape[2]:
            ratio = scores.shape[2] // resolution
            scores = torch.nn.functional.max_pool3d(scores, ratio, ratio, 0)
        flat_indices = scores[:, 0].reshape(scores.shape[0], -1).argmax(dim=1)
        height, width = scores.shape[3], scores.shape[4]
        plane = height * width
        z = flat_indices // plane
        y = (flat_indices % plane) // width
        x = flat_indices % width
        batch = torch.arange(scores.shape[0], device=scores.device)
        return torch.stack((batch, z, y, x), dim=1).int()

    @staticmethod
    def _sparse_structure_seed_attempts(seed: int) -> list[int]:
        base_seed = int(seed)
        attempts = [base_seed]
        seen = {base_seed}
        for offset in _EMPTY_STRUCTURE_RETRY_SEED_OFFSETS:
            attempt_seed = base_seed + int(offset)
            if attempt_seed < 0 or attempt_seed in seen:
                continue
            attempts.append(attempt_seed)
            seen.add(attempt_seed)
        return attempts

    def _sample_sparse_structure_with_seed_retry(
        self,
        cond: dict,
        resolution: int,
        seed: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        seed_attempts = self._sparse_structure_seed_attempts(seed)
        coords = torch.empty((0, 4), dtype=torch.int32, device=self.device)
        for attempt_index, attempt_seed in enumerate(seed_attempts):
            torch.manual_seed(int(attempt_seed))
            coords = self.sample_sparse_structure(
                cond,
                resolution,
                num_samples,
                sampler_params,
                fallback_to_max_logit=attempt_index + 1 == len(seed_attempts),
            )
            if coords.shape[0] > 0:
                return coords
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self._sample_sparse_structure_with_seed_retry(
            cond_512, ss_res,
            seed, num_samples, sparse_structure_sampler_params
        )
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
        fusion_config: MultiImageFusionConfig | dict | None = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with multiple image conditions.

        Multi-Diffusion mode fuses per-view positive velocity predictions at
        every sampler step while sharing one unconditional branch for CFG.
        """
        if num_samples != 1:
            raise ValueError("TRELLIS.2 multi-image generation currently supports num_samples=1.")
        if len(images) == 0:
            raise ValueError("run_multi_image requires at least one input image.")

        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        ss_fusion_config = self._normalize_fusion_config(fusion_config, "sparse_structure")
        shape_fusion_config = self._normalize_fusion_config(fusion_config, "shape_slat")
        tex_fusion_config = self._normalize_fusion_config(fusion_config, "tex_slat")

        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]

        torch.manual_seed(seed)
        cond_512 = self.get_cond(images, 512)
        cond_512['neg_cond'] = cond_512['neg_cond'][:1]
        cond_1024 = None
        if pipeline_type != '512':
            cond_1024 = self.get_cond(images, 1024)
            cond_1024['neg_cond'] = cond_1024['neg_cond'][:1]

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        ss_sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        shape_sampler_params = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        tex_sampler_params = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}

        with self.inject_sampler_multi_image(
            'sparse_structure_sampler',
            len(images),
            int(ss_sampler_params.get('steps', 50)),
            mode=mode,
            fusion_config=ss_fusion_config,
        ):
            coords = self._sample_sparse_structure_with_seed_retry(
                cond_512, ss_res,
                seed, num_samples, sparse_structure_sampler_params
            )

        if pipeline_type == '512':
            with self.inject_sampler_multi_image(
                'shape_slat_sampler',
                len(images),
                int(shape_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=shape_fusion_config,
            ):
                shape_slat = self.sample_shape_slat(
                    cond_512, self.models['shape_slat_flow_model_512'],
                    coords, shape_slat_sampler_params
                )
            with self.inject_sampler_multi_image(
                'tex_slat_sampler',
                len(images),
                int(tex_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=tex_fusion_config,
            ):
                tex_slat = self.sample_tex_slat(
                    cond_512, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params
                )
            res = 512
        elif pipeline_type == '1024':
            with self.inject_sampler_multi_image(
                'shape_slat_sampler',
                len(images),
                int(shape_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=shape_fusion_config,
            ):
                shape_slat = self.sample_shape_slat(
                    cond_1024, self.models['shape_slat_flow_model_1024'],
                    coords, shape_slat_sampler_params
                )
            with self.inject_sampler_multi_image(
                'tex_slat_sampler',
                len(images),
                int(tex_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=tex_fusion_config,
            ):
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params
                )
            res = 1024
        elif pipeline_type == '1024_cascade':
            with self.inject_sampler_multi_image(
                'shape_slat_sampler',
                len(images),
                int(shape_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=shape_fusion_config,
            ):
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1024,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )
            with self.inject_sampler_multi_image(
                'tex_slat_sampler',
                len(images),
                int(tex_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=tex_fusion_config,
            ):
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params
                )
        elif pipeline_type == '1536_cascade':
            with self.inject_sampler_multi_image(
                'shape_slat_sampler',
                len(images),
                int(shape_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=shape_fusion_config,
            ):
                shape_slat, res = self.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                    512, 1536,
                    coords, shape_slat_sampler_params,
                    max_num_tokens
                )
            with self.inject_sampler_multi_image(
                'tex_slat_sampler',
                len(images),
                int(tex_sampler_params.get('steps', 50)),
                mode=mode,
                fusion_config=tex_fusion_config,
            ):
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params
                )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh
