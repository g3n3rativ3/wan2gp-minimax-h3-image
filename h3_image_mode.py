"""Core logic for the MiniMax H3 Image Mode plugin.

Two independent patches turn H3 into a still-image generator:

1.  ``patch_model_def`` enables WanGP's native "Text to Video / Text to Image"
    tab switch on every H3 model definition, and lowers ``frames_minimum`` to
    the value the H3 pipeline itself accepts (see
    ``models/minimax_h3/pipeline.py``, which calls
    ``normalize_frame_count(frame_num, 5, 17, 5)``).  Without this, WanGP's
    ``floor_frame_count`` would raise any requested length back up to 107.

2.  ``patch_pipeline`` wraps ``MiniMaxH3Pipeline.generate`` so that, in image
    mode, it runs at the model's native frame rate and returns a single decoded
    frame instead of a short clip.
"""

from __future__ import annotations

import copy


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

H3_ARCHITECTURES = (
    "minimax_h3_fl2va",
    "minimax_h3_fl2va_pruned",
    "minimax_h3_ref2va",
    "minimax_h3_ref2va_pruned",
)

REFERENCE_ARCHITECTURES = ("minimax_h3_ref2va", "minimax_h3_ref2va_pruned")

# Marker written into a patched model_def so the patch stays idempotent.
PATCH_FLAG = "_h3_image_mode_patched"

# Optional server_config key (WanGP's settings JSON).  There is deliberately no
# form widget for this: WanGP custom settings cannot be filtered by image_mode,
# so one would sit permanently in the Text to Video form, and the five custom
# setting slots are shared with the model and any other plugin.
FRAME_INDEX_CONFIG_KEY = "h3_image_frame_index"

# Smallest generation that can actually be DECODED.
#
# The H3 pipeline accepts frame_num=5 (normalize_frame_count(frame_num, 5, 17, 5))
# and denoising works fine, but WanGP's chunked VAE decoder cannot handle the
# resulting 2 latent frames: with tokens_chunk_size=5 and token_drop=3 it
# computes num_chunks = (2+3)//5 - 1 = 0, never enters its decode loop, and
# raises "MiniMax H3 VAE decoded 0 frames, expected 5".
# 22 frames -> 7 latent frames -> 1 chunk, which is the real floor.
# (ComfyUI gets away with length=5 because it uses a different VAE decoder.)
H3_MIN_FRAMES = 22

# Frame counts WanGP's image-mode dropdown can produce: 5 + 17n.
H3_FRAME_GRID = (5, 22, 39, 56)

# H3 is a 24 fps model.  WanGP passes fps=1 for image outputs, which would
# inflate the audio latent length by 24x (audio_t = frames / fps * 40).
H3_FALLBACK_FPS = 24.0

MAX_IMAGE_BATCH = 4

# Reduced guide choices for image mode.  H3's video-oriented choices ("Use Two
# Reference Videos", "Transfer Depth Map From Control Video"...) make no sense
# once the guide widget is a still image, and WanGP would rewrite their labels
# to nonsense by replacing "Video" with "Image".
REFERENCE_GUIDE_CHOICES_IMAGE = {
    "choices": [
        ("Generate without a Control Image", ""),
        ("Provide Generic Control Image", "GV"),
    ],
    "letters_filter": "UGPDEV+-",
    "default": "",
    "label": "Control Image",
}

FIRSTLAST_GUIDE_CHOICES_IMAGE = {
    "choices": [
        ("Generate without a Control Image", ""),
        ("Use Control Image (editing)", "GV"),
    ],
    "letters_filter": "GVKFI",
    "default": "",
    "label": "Control Image",
}


# --------------------------------------------------------------------------
# Model definition patching
# --------------------------------------------------------------------------

def is_h3_model_def(model_def) -> bool:
    if not isinstance(model_def, dict):
        return False
    architecture = model_def.get("architecture", "")
    return architecture in H3_ARCHITECTURES


def patch_model_def(model_def) -> bool:
    """Enable WanGP's native image mode on one H3 model definition.

    Returns True when the definition was modified.
    """
    if not is_h3_model_def(model_def) or model_def.get(PATCH_FLAG, False):
        return False

    architecture = model_def.get("architecture", "")
    reference_mode = architecture in REFERENCE_ARCHITECTURES

    # Native "Text to Video" / "Text to Image" tabs.  image_mode stays 0 by
    # default (set by the H3 handler's update_default_settings), so the plugin
    # is inert until the user picks the Text to Image tab.
    model_def["v2i_switch_supported"] = True

    # Let floor_frame_count reach the shortest decodable generation.  The
    # native "Generate additional frames before keeping the first image"
    # dropdown still lists 5, but floor_frame_count clamps it up to 22, so no
    # UI choice can reach the broken 2-latent-frame case.
    model_def["frames_minimum"] = H3_MIN_FRAMES

    model_def["image_batch_size_max"] = MAX_IMAGE_BATCH
    model_def["batch_size_label"] = "Number of Images to Generate"

    model_def["guide_custom_choices_image"] = copy.deepcopy(
        REFERENCE_GUIDE_CHOICES_IMAGE if reference_mode else FIRSTLAST_GUIDE_CHOICES_IMAGE
    )

    _patch_prompt_enhancer(model_def)
    _alias_image_enhancer_instructions(model_def)

    model_def[PATCH_FLAG] = True
    return True


def _patch_prompt_enhancer(model_def) -> None:
    """Make H3's prompt rewriter available in image mode.

    H3 declares its enhancer entries as "TV" / "TIV".  WanGP filters those keys
    on the letters "VP" and only keeps entries carrying "P" when image_mode > 0,
    so without this the enhancer dropdown would be empty in image mode - and an
    LLM rewriter is precisely what H3 needs to produce good stills.
    """
    enhancer_def = model_def.get("prompt_enhancer_def")
    if not isinstance(enhancer_def, dict):
        return
    labels = enhancer_def.get("labels")
    if not isinstance(labels, dict):
        return

    patched = copy.deepcopy(enhancer_def)
    patched["labels"] = {
        (key if "P" in key else key + "P"): label
        for key, label in labels.items()
    }
    model_def["prompt_enhancer_def"] = patched


def _alias_image_enhancer_instructions(model_def) -> None:
    """Point the image_* enhancer keys at H3's video_* ones.

    resolve_prompt_enhancer_settings() looks up
    "{image|video}_prompt_enhancer_instructions" from the model definition.
    The H3 handler only declares the text_* and video_* variants, so in image
    mode WanGP finds nothing and falls back to its generic instructions - and
    because the "TI" mode contains an "I", the fallback to H3's own text
    instructions is skipped too.  The enhancer then just captions the first
    reference image and discards the user's prompt.

    Aliasing keeps H3's own system prompts in play.  They are written for video
    output, which is what we want anyway: the generation really is a (very
    short) video, and the recommended prompt style is to open with something
    like "A still frame shot of...".
    """
    for suffix in ("instructions", "max_tokens"):
        prefix = f"video_prompt_enhancer_{suffix}"
        for key in [k for k in model_def if k.startswith(prefix)]:
            image_key = "image" + key[len("video"):]
            if image_key not in model_def:
                model_def[image_key] = model_def[key]


def patch_all_model_defs(models_def) -> int:
    """Patch every H3 entry of WanGP's models_def dictionary."""
    if not isinstance(models_def, dict):
        return 0
    return sum(1 for model_def in models_def.values() if patch_model_def(model_def))


# --------------------------------------------------------------------------
# Pipeline patching
# --------------------------------------------------------------------------

_pipeline_patched = False
_model_def_provider = None
_server_config = None


def set_model_def_provider(get_model_def) -> None:
    """Register WanGP's get_model_def so the wrapper can read the model fps."""
    global _model_def_provider
    if callable(get_model_def):
        _model_def_provider = get_model_def


def set_server_config(server_config) -> None:
    global _server_config
    if isinstance(server_config, dict):
        _server_config = server_config


def build_get_model_def_wrapper(original_get_model_def):
    """Wrap WanGP's get_model_def so H3 definitions are always patched.

    WanGP rebuilds every model definition from disk whenever
    ``refresh_model_defs()`` runs - the Refresh Models button, the finetune
    editor, or another plugin.  A one-shot patch applied at startup silently
    disappears at that point, and the image mode tabs vanish with it.  Patching
    on read makes the plugin immune to that: the definition is repaired the
    moment anything asks for it, including the generation form rebuild.
    """
    if getattr(original_get_model_def, "_h3_image_mode_wrapper", False):
        return original_get_model_def

    def get_model_def(model_type):
        model_def = original_get_model_def(model_type)
        if model_def is not None and not model_def.get(PATCH_FLAG, False):
            patch_model_def(model_def)
        return model_def

    get_model_def._h3_image_mode_wrapper = True
    get_model_def._h3_image_mode_original = original_get_model_def
    return get_model_def


def patch_pipeline(get_model_def=None) -> bool:
    """Wrap MiniMaxH3Pipeline.generate for image mode. Idempotent."""
    global _pipeline_patched
    set_model_def_provider(get_model_def)
    if _pipeline_patched:
        return True

    try:
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline
    except Exception as error:  # pragma: no cover - depends on the install
        print(f"[H3 Image Mode] MiniMax H3 pipeline unavailable, not patched: {error}")
        return False

    original_generate = MiniMaxH3Pipeline.generate
    if getattr(original_generate, "_h3_image_mode_wrapper", False):
        _pipeline_patched = True
        return True

    def generate(self, *args, **kwargs):
        image_mode = _as_int(kwargs.get("image_mode", 0))
        if image_mode <= 0:
            return original_generate(self, *args, **kwargs)

        # WanGP sets fps=1 for image outputs.  H3 derives its audio latent
        # length from frames / fps, so leaving it at 1 would allocate ~24x too
        # many audio tokens and desynchronise the joint attention.
        kwargs["fps"] = _resolve_fps(kwargs)

        frame_index = _configured_frame_index()
        # Safety net: clamp to what this pipeline's VAE can actually decode,
        # in case a caller or a finetuned VAE bypasses the model_def floor.
        minimum = _min_decodable_frames(self)
        if _as_int(kwargs.get("frame_num", 0)) < minimum:
            kwargs["frame_num"] = minimum
        batch_size = max(1, min(MAX_IMAGE_BATCH, _as_int(kwargs.get("batch_size", 1), 1)))

        base_seed = _as_int(kwargs.get("seed", 0))
        set_status = kwargs.get("set_progress_status")

        frames = []
        for image_no in range(batch_size):
            if batch_size > 1:
                kwargs["seed"] = base_seed + image_no
                if callable(set_status):
                    try:
                        set_status(f"Image {image_no + 1}/{batch_size}")
                    except Exception:
                        pass

            result = original_generate(self, *args, **kwargs)
            if result is None:
                return None  # aborted

            video = result.get("x") if isinstance(result, dict) else result
            if video is None:
                return None
            frames.append(_pick_frame(video, frame_index))

        import torch

        return {"x": torch.cat(frames, dim=1) if len(frames) > 1 else frames[0]}

    generate._h3_image_mode_wrapper = True
    generate._h3_image_mode_original = original_generate
    MiniMaxH3Pipeline.generate = generate
    _pipeline_patched = True
    return True


def _pick_frame(video, frame_index: int):
    """Keep one frame from a (C, F, H, W) decoded tensor."""
    frame_count = video.shape[1]
    index = max(0, min(int(frame_index), frame_count - 1))
    return video[:, index:index + 1].clone()


def _resolve_fps(kwargs) -> float:
    if callable(_model_def_provider):
        try:
            model_def = _model_def_provider(kwargs.get("model_type", "")) or {}
            fps = float(model_def.get("fps", 0) or 0)
            if fps > 0:
                return fps
        except Exception:
            pass
    return H3_FALLBACK_FPS


def _min_decodable_frames(pipeline) -> int:
    """Shortest frame count this pipeline's VAE can decode, from its geometry."""
    try:
        vae = pipeline.vae
        chunk = int(vae.tokens_chunk_size)
        drop = int(vae.config.token_drop)
    except Exception:
        return H3_MIN_FRAMES
    for frames in H3_FRAME_GRID:
        latents = 2 + ((frames - 5) // 17) * 5
        tokens = latents + drop
        pad = (-tokens) % chunk
        if (tokens + pad) // chunk - int(drop > 0) >= 1:
            return frames
    return H3_MIN_FRAMES


def _configured_frame_index() -> int:
    """Which decoded frame to keep. 0 unless overridden in WanGP's config."""
    if isinstance(_server_config, dict):
        return _as_int(_server_config.get(FRAME_INDEX_CONFIG_KEY, 0))
    return 0


def _as_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
