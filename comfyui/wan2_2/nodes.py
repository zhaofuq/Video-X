"""Modified from https://github.com/kijai/ComfyUI-EasyAnimateWrapper/blob/main/nodes.py
"""
import copy
import gc
import json
import os

import comfy.model_management as mm
import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar, load_torch_file
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ...videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                              get_closest_ratio)
from ...videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                                 WanT5EncoderModel, Wan2_2Transformer3DModel)
from ...videox_fun.pipeline import Wan2_2I2VPipeline, Wan2_2Pipeline, Wan2_2TI2VPipeline
from ...videox_fun.ui.controller import all_cheduler_dict
from ...videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8, convert_weight_dtype_wrapper, replace_parameters_by_name)
from ...videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from ...videox_fun.utils.utils import (get_image_to_video_latent, filter_kwargs,
                                      get_video_to_video_latent,
                                      save_videos_grid)
from ...videox_fun.models.cache_utils import get_teacache_coefficients
from ..comfyui_utils import eas_cache_dir, script_directory, to_pil

# Used in lora cache
transformer_cpu_cache       = {}
transformer_high_cpu_cache  = {}
# lora path before
lora_path_before            = ""
lora_high_path_before       = ""

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

class LoadWan2_2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        'Wan2.2-T2V-A14B',
                        'Wan2.2-I2V-A14B',
                        'Wan2.2-TI2V-5B',
                    ],
                    {
                        "default": 'Wan2.2-T2V-A14B',
                    }
                ),
                "GPU_memory_mode":(
                    ["model_full_load", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"],
                    {
                        "default": "model_cpu_offload",
                    }
                ),
                "config": (
                    [
                        "wan2.2/wan_civitai_t2v.yaml",
                        "wan2.2/wan_civitai_i2v.yaml",
                        "wan2.2/wan_civitai_5b.yaml",
                    ],
                    {
                        "default": "wan2.2/wan_civitai_t2v.yaml",
                    }
                ),
                "precision": (
                    ['fp16', 'bf16'],
                    {
                        "default": 'fp16'
                    }
                ),
            },
        }

    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "loadmodel"
    CATEGORY = "CogVideoXFUNWrapper"

    def loadmodel(self, GPU_memory_mode, model, precision, config):
        # Init weight_dtype and device
        device          = mm.get_torch_device()
        offload_device  = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        # Init processbar
        pbar = ProgressBar(5)

        # Load config
        config_path = f"{script_directory}/config/{config}"
        config = OmegaConf.load(config_path)

        # Detect model is existing or not
        possible_folders = ["CogVideoX_Fun", "Fun_Models", "VideoX_Fun"] + \
                [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/Diffusion_Transformer")] # Possible folder names to check
        # Initialize model_name as None
        model_name = None

        # Check if the model exists in any of the possible folders within folder_paths.models_dir
        for folder in possible_folders:
            candidate_path = os.path.join(folder_paths.models_dir, folder, model)
            if os.path.exists(candidate_path):
                model_name = candidate_path
                break

        # If model_name is still None, check eas_cache_dir for each possible folder
        if model_name is None and os.path.exists(eas_cache_dir):
            for folder in possible_folders:
                candidate_path = os.path.join(eas_cache_dir, folder, model)
                if os.path.exists(candidate_path):
                    model_name = candidate_path
                    break

        # If model_name is still None, prompt the user to download the model
        if model_name is None:
            print(f"Please download cogvideoxfun model to one of the following directories:")
            for folder in possible_folders:
                print(f"- {os.path.join(folder_paths.models_dir, folder)}")
                if os.path.exists(eas_cache_dir):
                    print(f"- {os.path.join(eas_cache_dir, folder)}")
            raise ValueError("Please download Fun model")

        # Get Vae
        Choosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Choosen_AutoencoderKL.from_pretrained(
            os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)
        # Update pbar
        pbar.update(1)

        # Load Sampler
        print("Load Sampler.")
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        # Update pbar
        pbar.update(1)
        
        # Get Transformer
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
            transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
                os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                low_cpu_mem_usage=True,
                torch_dtype=weight_dtype,
            )
        else:
            transformer_2 = None
        # Update pbar
        pbar.update(1) 

        # Get tokenizer and text_encoder
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        pbar.update(1) 

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        pbar.update(1) 

        # Get pipeline
        model_type = "Inpaint"
        if model_type == "Inpaint":
            if "wan_civitai_5b" in config_path:
                pipeline = Wan2_2TI2VPipeline(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    transformer=transformer,
                    transformer_2=transformer_2,
                    scheduler=scheduler,
                )
            else:
                if transformer.config.in_channels != vae.config.latent_channels:
                    pipeline = Wan2_2I2VPipeline(
                        transformer=transformer,
                        transformer_2=transformer_2,
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                    )
                else:
                    pipeline = Wan2_2Pipeline(
                        transformer=transformer,
                        transformer_2=transformer_2,
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        scheduler=scheduler,
                    )
        else:
            raise ValueError(f"Model type {model_type} not supported")

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation",], device=device)
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        funmodels = {
            'pipeline': pipeline, 
            'dtype': weight_dtype,
            'model_name': model_name,
            'model_type': model_type,
            'loras': [],
            'strength_model': [],
            'config': config,
        }
        return (funmodels,)

class LoadWan2_2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": ("FunModels",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "lora_high_name": (folder_paths.get_filename_list("loras"), {"default": None,}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_cache":([False, True],  {"default": False,}),
            }
        }
    RETURN_TYPES = ("FunModels",)
    RETURN_NAMES = ("funmodels",)
    FUNCTION = "load_lora"
    CATEGORY = "CogVideoXFUNWrapper"

    def load_lora(self, funmodels, lora_name, lora_high_name, strength_model, lora_cache):
        if lora_name is not None:
            funmodels['lora_cache'] = lora_cache
            funmodels['loras'] = funmodels.get("loras", []) + [folder_paths.get_full_path("loras", lora_name)]
            funmodels['loras_high'] = funmodels.get("loras_high", []) + [folder_paths.get_full_path("loras", lora_high_name)]
            funmodels['strength_model'] = funmodels.get("strength_model", []) + [strength_model]
            return (funmodels,)
        else:
            return (funmodels,)

class Wan2_2T2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT", 
                ),
                "negative_prompt": (
                    "STRING_PROMPT", 
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 5, "max": 161, "step": 4}
                ),
                "width": (
                    "INT", {"default": 832, "min": 64, "max": 2048, "step": 16}
                ),
                "height": (
                    "INT", {"default": 480, "min": 64, "max": 2048, "step": 16}
                ),
                "is_image":(
                    [
                        False,
                        True
                    ], 
                    {
                        "default": False,
                    }
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Flow",
                    ],
                    {
                        "default": 'Flow'
                    }
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
            },
            "optional":{
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, width, height, is_image, seed, steps, cfg, scheduler, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, riflex_k=0):
        global transformer_cpu_cache
        global transformer_high_cpu_cache
        global lora_path_before
        global lora_high_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()

        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        config = funmodels['config']
        weight_dtype = funmodels['dtype']

        # Get boundary for wan
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler](**filter_kwargs(all_cheduler_dict[scheduler], OmegaConf.to_container(config['scheduler_kwargs'])))

        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        else:
            pipeline.transformer.disable_teacache()
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator= torch.Generator(device).manual_seed(seed)

        video_length = 1 if is_image else video_length
        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if pipeline.transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                   
                    if pipeline.transformer_2 is not None:
                        # Save the original weights to cpu
                        if len(transformer_high_cpu_cache) == 0:
                            print('Save transformer high state_dict to cpu memory')
                            transformer_high_state_dict = pipeline.transformer_2.state_dict()
                            for key in transformer_high_state_dict:
                                transformer_high_cpu_cache[key] = transformer_high_state_dict[key].clone().cpu()

                        lora_high_path_now = str(funmodels.get("loras_high", []) + funmodels.get("strength_model", []))
                        if lora_high_path_now != lora_high_path_before:
                            print('Merge Lora High with Cache')
                            lora_high_path_before = copy.deepcopy(lora_high_path_now)
                            pipeline.transformer_2.load_state_dict(transformer_cpu_cache)
                            for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                                pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
            else:
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if pipeline.transformer_2 is not None:
                    if len(transformer_high_cpu_cache) != 0:
                        pipeline.transformer_2.load_state_dict(transformer_high_cpu_cache)
                        transformer_high_cpu_cache = {}
                        lora_high_path_before = ""
                        gc.collect()

                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,
                boundary     = boundary,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                if pipeline.transformer_2 is not None:
                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
        return (videos,)   


class Wan2_2I2VSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "funmodels": (
                    "FunModels", 
                ),
                "prompt": (
                    "STRING_PROMPT",
                ),
                "negative_prompt": (
                    "STRING_PROMPT",
                ),
                "video_length": (
                    "INT", {"default": 81, "min": 5, "max": 161, "step": 4}
                ),
                "base_resolution": (
                    [ 
                        512,
                        640,
                        768,
                        896,
                        960,
                        1024,
                    ], {"default": 640}
                ),
                "seed": (
                    "INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}
                ),
                "steps": (
                    "INT", {"default": 50, "min": 1, "max": 200, "step": 1}
                ),
                "cfg": (
                    "FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01}
                ),
                "scheduler": (
                    [ 
                        "Flow",
                    ],
                    {
                        "default": 'Flow'
                    }
                ),
                "teacache_threshold": (
                    "FLOAT", {"default": 0.10, "min": 0.00, "max": 1.00, "step": 0.005}
                ),
                "enable_teacache":(
                    [False, True],  {"default": True,}
                ),
                "num_skip_start_steps": (
                    "INT", {"default": 5, "min": 0, "max": 50, "step": 1}
                ),
                "teacache_offload":(
                    [False, True],  {"default": True,}
                ),
                "cfg_skip_ratio":(
                    "FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}
                ),
            },
            "optional":{
                "start_img": ("IMAGE",),
                "riflex_k": ("RIFLEXT_ARGS",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("images",)
    FUNCTION = "process"
    CATEGORY = "CogVideoXFUNWrapper"

    def process(self, funmodels, prompt, negative_prompt, video_length, base_resolution, seed, steps, cfg, scheduler, teacache_threshold, enable_teacache, num_skip_start_steps, teacache_offload, cfg_skip_ratio, start_img=None, end_img=None, riflex_k=0):
        global transformer_cpu_cache
        global transformer_high_cpu_cache
        global lora_path_before
        global lora_high_path_before
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        gc.collect()
        
        # Get Pipeline
        pipeline = funmodels['pipeline']
        model_name = funmodels['model_name']
        config = funmodels['config']
        weight_dtype = funmodels['dtype']

        start_img = [to_pil(_start_img) for _start_img in start_img] if start_img is not None else None
        end_img = [to_pil(_end_img) for _end_img in end_img] if end_img is not None else None
        # Count most suitable height and width
        spatial_compression_ratio = pipeline.vae.config.spatial_compression_ratio if hasattr(pipeline.vae.config, "spatial_compression_ratio") else 8
        aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        original_width, original_height = start_img[0].size if type(start_img) is list else Image.open(start_img).size
        closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
        height, width = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]
        
        # Get boundary for wan
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

        # Load Sampler
        pipeline.scheduler = all_cheduler_dict[scheduler](**filter_kwargs(all_cheduler_dict[scheduler], OmegaConf.to_container(config['scheduler_kwargs'])))
        coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)
        else:
            pipeline.transformer.disable_teacache()
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.disable_teacache()

        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, steps)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        generator= torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_img, end_img, video_length=video_length, sample_size=(height, width))

            if riflex_k > 0:
                latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if pipeline.transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

            # Apply lora
            if funmodels.get("lora_cache", False):
                if len(funmodels.get("loras", [])) != 0:
                    # Save the original weights to cpu
                    if len(transformer_cpu_cache) == 0:
                        print('Save transformer state_dict to cpu memory')
                        transformer_state_dict = pipeline.transformer.state_dict()
                        for key in transformer_state_dict:
                            transformer_cpu_cache[key] = transformer_state_dict[key].clone().cpu()
                    
                    lora_path_now = str(funmodels.get("loras", []) + funmodels.get("strength_model", []))
                    if lora_path_now != lora_path_before:
                        print('Merge Lora with Cache')
                        lora_path_before = copy.deepcopy(lora_path_now)
                        pipeline.transformer.load_state_dict(transformer_cpu_cache)
                        for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                            pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                   
                    if pipeline.transformer_2 is not None:
                        # Save the original weights to cpu
                        if len(transformer_high_cpu_cache) == 0:
                            print('Save transformer high state_dict to cpu memory')
                            transformer_high_state_dict = pipeline.transformer_2.state_dict()
                            for key in transformer_high_state_dict:
                                transformer_high_cpu_cache[key] = transformer_high_state_dict[key].clone().cpu()

                        lora_high_path_now = str(funmodels.get("loras_high", []) + funmodels.get("strength_model", []))
                        if lora_high_path_now != lora_high_path_before:
                            print('Merge Lora High with Cache')
                            lora_high_path_before = copy.deepcopy(lora_high_path_now)
                            pipeline.transformer_2.load_state_dict(transformer_cpu_cache)
                            for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                                pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
            else:
                print('Merge Lora')
                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if len(transformer_cpu_cache) != 0:
                    pipeline.transformer.load_state_dict(transformer_cpu_cache)
                    transformer_cpu_cache = {}
                    lora_path_before = ""
                    gc.collect()
                
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)

                # Clear lora when switch from lora_cache=True to lora_cache=False.
                if pipeline.transformer_2 is not None:
                    if len(transformer_high_cpu_cache) != 0:
                        pipeline.transformer_2.load_state_dict(transformer_high_cpu_cache)
                        transformer_high_cpu_cache = {}
                        lora_high_path_before = ""
                        gc.collect()

                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = merge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")

            sample = pipeline(
                prompt, 
                num_frames = video_length,
                negative_prompt = negative_prompt,
                height      = height,
                width       = width,
                generator   = generator,
                guidance_scale = cfg,
                num_inference_steps = steps,

                video        = input_video,
                mask_video   = input_video_mask,
                boundary     = boundary,
                comfyui_progressbar = True,
            ).videos
            videos = rearrange(sample, "b c t h w -> (b t) h w c")

            if not funmodels.get("lora_cache", False):
                print('Unmerge Lora')
                for _lora_path, _lora_weight in zip(funmodels.get("loras", []), funmodels.get("strength_model", [])):
                    pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype)
                if pipeline.transformer_2 is not None:
                    for _lora_path, _lora_weight in zip(funmodels.get("loras_high", []), funmodels.get("strength_model", [])):
                        pipeline = unmerge_lora(pipeline, _lora_path, _lora_weight, device="cuda", dtype=weight_dtype, sub_transformer_name="transformer_2")
        return (videos,)   