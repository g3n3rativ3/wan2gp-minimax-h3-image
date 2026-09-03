# MiniMax H3 Image Mode — WanGP plugin

Turns MiniMax H3 into a still-image generator, the same way the ComfyUI trick
does: run the shortest possible generation, throw away the video, keep the
first decoded frame.

Works with every H3 variant WanGP ships — FL2VA, Ref2VA, their Pruned 20B
counterparts and the PDD 8-step versions — plus any finetune built on those
architectures.

## Install

Copy the `wan2gp-h3-image` folder into WanGP's `plugins/` directory, or paste
this repository's URL into **Plugins → Install New Plugin**. Then enable it in
the **Plugins** tab, save, and **restart WanGP**.

No extra Python dependencies.

## Usage

1. Select any MiniMax H3 model in the Media Generator.
2. A **Text to Video / Text to Image** tab pair appears above the prompt.
   **Text to Video is selected by default** — nothing changes until you switch.
3. Pick **Text to Image**. WanGP hides the duration slider, the audio section
   and the video post-processing options, and switches the output to JPEG.
4. Add your reference images (Ref2VA) or a control image (FL2VA), write your
   prompt, generate.

### Settings that matter

**Generate additional frames before keeping the first image** (Advanced →
Quality, image mode only). This is WanGP's own dropdown; the plugin just makes
it reachable for H3. It offers 5 / 22 / 39 / 56 frames because H3's frame grid
is `5 + 17n`. Leave it on *Shortest generation* — 22 frames, the cheapest run
that WanGP can decode.

**The 5-frame entry is a trap and the plugin clamps it to 22.** H3 will happily
denoise 5 frames, but WanGP's chunked VAE decoder cannot decode the resulting 2
latent frames: with `tokens_chunk_size=5` and `token_drop=3` it computes
`num_chunks = (2+3)//5 - 1 = 0`, never enters its decode loop, and raises
*MiniMax H3 VAE decoded 0 frames, expected 5* after the denoising has already
run. 22 frames gives 7 latent frames and one full chunk. ComfyUI can use
`length=5` because it decodes with a different VAE implementation.

**Which frame is kept.** Always the first decoded one — the cleanest, since
later frames accumulate VAE degradation. There is deliberately no widget for
this: WanGP custom settings cannot be filtered by image mode, so any such
dropdown would sit permanently in the Text to Video form, and the five custom
setting slots are shared with the model and every other plugin. If you really
want a later frame, add `"h3_image_frame_index": 1` to WanGP's settings JSON.

**Number of Images to Generate**. Up to 4. Runs sequentially with seed, seed+1,
seed+2… and writes one JPEG per image.

**Resolution**. Push it up — the ComfyUI users doing this run 3–5 megapixels.
H3 handles large stills far better than it handles long clips.

**Prompt enhancer**. H3 wants its own structured prompt format, and the plugin
keeps *Write H3 Prompt* available in image mode (WanGP would otherwise filter it
out as video-only). Strongly recommended. For control-image editing, start the
prompt with something like *"A still frame shot of…"*.

## How it works

Two small patches, both reversible by disabling the plugin.

**Model definitions.** The plugin wraps WanGP's `get_model_def` so H3 definitions
are repaired on every read. Patching them once at startup is not enough:
`refresh_model_defs()` rebuilds every definition from disk — the Refresh Models
button, the finetune editor, another plugin — and a one-shot patch disappears
there, taking the image-mode tabs with it and silently forcing `image_mode`
back to 0.

The patch sets `v2i_switch_supported`, which is what makes WanGP render its
native image-mode tabs. It also
lowers `frames_minimum` from 107 to 5. That number is not a model limit — it
mirrors MiniMax's documented 4–15 s duration — and the H3 pipeline itself calls
`normalize_frame_count(frame_num, 5, 17, 5)`, so 5 is the real floor. Without
this, `floor_frame_count` would push every image-mode request back up to 107
frames. Side effect: in video mode the duration slider can now go below 107
frames too.

The plugin additionally supplies a reduced guide dropdown for image mode (H3's
video choices like *Use Two Reference Videos* become meaningless once the guide
widget is a still), and rewrites the prompt-enhancer entry keys from `TV`/`TIV`
to `TVP`/`TIVP` so they survive WanGP's image-mode filter.

**Pipeline.** `MiniMaxH3Pipeline.generate` is wrapped. In video mode the wrapper
returns immediately without touching anything. In image mode it:

- Restores `fps` to the model's native 24. WanGP passes `fps=1` for image
  outputs, and H3 sizes its audio latent as `round(frames / fps * 40)` — at
  `fps=1` that is 200 audio tokens for a 5-frame clip instead of 8, wasting
  compute and desynchronising the joint video/audio attention.
- Keeps a single frame from the decoded `(C, F, H, W)` tensor and drops the
  audio track, so WanGP writes one JPEG instead of five.
- Loops for batches larger than one.

## Known limitations

- The frame-to-keep has no UI, for the reason given above.
- `min_frames_if_references` is stripped from saved image metadata by WanGP's
  own metadata cleaner (it is retained only for VACE and T2V class models), so
  reloading settings from a generated image will not restore your frame count.
- Batch generation reuses the same conditioning and only varies the seed. If a
  batch misbehaves, set **Number of Images to Generate** back to 1.
- Sliding windows, soundtracks and temporal upsampling are unavailable in image
  mode. That is WanGP's normal behaviour for image outputs, not a plugin
  restriction.

## Credit

The technique comes from a StableDiffusion community post describing the same
modification to the ComfyUI Ref2V workflow: remove the save-video node, take the
image at batch index 0, save as image.
