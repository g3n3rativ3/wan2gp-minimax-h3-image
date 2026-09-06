"""MiniMax H3 Image Mode - a WanGP extension plugin.

Adds the native "Text to Image" tab to every MiniMax H3 model.  When that tab
is selected, WanGP runs the shortest H3 generation it can (5 frames by default)
and the plugin keeps a single decoded frame, saved as a JPEG instead of a video.

The plugin is inert until the user selects the Text to Image tab: H3's own
defaults set image_mode to 0, so the Text to Video tab stays selected.
"""

from shared.utils.plugins import WAN2GPPlugin

from .h3_image_mode import (
    build_get_model_def_wrapper,
    patch_all_model_defs,
    patch_pipeline,
    set_model_def_provider,
    set_server_config,
)


class H3ImageModePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "MiniMax H3 Image Mode"
        self.version = "1.3.0"
        self.description = (
            "Adds WanGP's native Text to Image tab to every MiniMax H3 model. "
            "Generates the shortest possible clip and saves its first frame as "
            "an image."
        )
        self.type = ["extension"]
        self._get_model_def_wrapped = False

    # -- lifecycle ---------------------------------------------------------

    def setup_ui(self):
        # setup_ui runs before inject_globals, so no WanGP global exists yet.
        # The pipeline patch has no such dependency; the rest is installed from
        # __setattr__ as each requested global arrives, which still happens
        # before the Gradio UI is built.
        self.request_global("get_model_def")
        self.request_global("server_config")
        self.request_global("models_def")
        patch_pipeline()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "get_model_def" and callable(value):
            set_model_def_provider(value)
            self._install_get_model_def_wrapper(value)
        elif name == "server_config" and isinstance(value, dict):
            set_server_config(value)
        elif name == "models_def" and isinstance(value, dict):
            self._patch_model_defs(value)

    # -- helpers -----------------------------------------------------------

    def _install_get_model_def_wrapper(self, original) -> None:
        """Repair H3 definitions on every read, not just once at startup.

        Without this, any refresh_model_defs() call - the Refresh Models
        button, the finetune editor, another plugin - rebuilds the definitions
        from disk and silently drops v2i_switch_supported, which makes the
        image mode tabs disappear and forces image_mode back to 0.
        """
        if self._get_model_def_wrapped:
            return
        wrapper = build_get_model_def_wrapper(original)
        if wrapper is original:
            self._get_model_def_wrapped = True
            return
        error = self.set_global("get_model_def", wrapper)
        if error:
            print(f"[H3 Image Mode] Could not install the get_model_def wrapper: {error}")
            return
        self._get_model_def_wrapped = True

    def _patch_model_defs(self, models_def) -> None:
        try:
            patched = patch_all_model_defs(models_def)
        except Exception as error:
            print(f"[H3 Image Mode] Could not patch model definitions: {error}")
            return
        if patched:
            print(f"[H3 Image Mode] Image mode enabled on {patched} MiniMax H3 model(s).")
        patch_pipeline(getattr(self, "get_model_def", None))
