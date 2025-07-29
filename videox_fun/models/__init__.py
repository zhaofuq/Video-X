import importlib.util

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from .cogvideox_transformer3d import CogVideoXTransformer3DModel
from .cogvideox_vae import AutoencoderKLCogVideoX
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (Wan2_2Transformer3DModel, WanSelfAttention,
                                WanTransformer3DModel)
from .wan_vae import AutoencoderKLWan, AutoencoderKLWan_

# The pai_fuser is an internally developed acceleration package, which can be used on PAI.
if importlib.util.find_spec("pai_fuser") is not None:
    # The simple_wrapper is used to solve the problem about conflicts between cython and torch.compile
    def simple_wrapper(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    from ..dist import parallel_magvit_vae
    AutoencoderKLWan_.decode = simple_wrapper(parallel_magvit_vae(0.2, 8)(AutoencoderKLWan_.decode))

    import torch
    from pai_fuser.core.attention import wan_sparse_attention_wrapper
    
    WanSelfAttention.forward = simple_wrapper(wan_sparse_attention_wrapper()(WanSelfAttention.forward))
    print("Import Sparse Attention")

    WanTransformer3DModel.forward = simple_wrapper(WanTransformer3DModel.forward)

    import os
    from pai_fuser.core import (cfg_skip_turbo, disable_cfg_skip,
                                enable_cfg_skip)

    WanTransformer3DModel.enable_cfg_skip = enable_cfg_skip()(WanTransformer3DModel.enable_cfg_skip)
    WanTransformer3DModel.disable_cfg_skip = disable_cfg_skip()(WanTransformer3DModel.disable_cfg_skip)
    print("Import CFG Skip Turbo")

    from pai_fuser.core.rope import ENABLE_KERNEL, fast_rope_apply_qk

    if ENABLE_KERNEL:
        import types
        from . import wan_transformer3d

        def deepcopy_function(f):
            return types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,closure=f.__closure__)

        local_rope_apply_qk = deepcopy_function(wan_transformer3d.rope_apply_qk)
        def adaptive_fast_rope_apply_qk(q, k, grid_sizes, freqs):
            if torch.is_grad_enabled():
                return local_rope_apply_qk(q, k, grid_sizes, freqs)
            else:
                return fast_rope_apply_qk(q, k, grid_sizes, freqs)
            
        wan_transformer3d.rope_apply_qk = adaptive_fast_rope_apply_qk
        rope_apply_qk = adaptive_fast_rope_apply_qk
        print("Import PAI Fast rope")
