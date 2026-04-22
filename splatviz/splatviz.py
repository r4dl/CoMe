from imgui_bundle import imgui
import numpy as np
import torch
import sys

# how nice, an absolute path
# this is setup to work in 3dgs-meshing/splatviz
sys.path.append("./../")
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *
from splatviz_utils.dict_utils import EasyDict
from widgets import (
    edit_widget,
    eval_widget,
    performance_widget,
    load_widget_pkl,
    load_widget_ply,
    video_widget,
    cam_widget,
    capture_widget,
    latent_widget,
    render_widget,
    training_widget,
)
import json
from pathlib import Path


class Splatviz(imgui_window.ImguiWindow):
    def __init__(self, data_path, mode, host, port, ggd_path=""):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="splatviz",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()
        self._zoom = 1
        self._img_dim = [1024, 1024]
        
        # try to restore the cameras
        self.cameras = []
        try:
            camera_path = Path(data_path)
            cameras_path = None
            for _ in range(3):
                candidate = camera_path / "cameras.json"
                if candidate.exists():
                    cameras_path = candidate
                    break
                camera_path = camera_path.parent
            assert cameras_path is not None and cameras_path.exists(), (
                "Cameras file not found within 3 parent levels."
            )
            with open(cameras_path, "r") as file:
                self.cameras = json.load(file)
        except Exception as e:
            print(f"Error restoring cameras: {e}")

        # Internals.
        self._last_error_print = None

        self.widgets = []
        update_all_the_time = False
        if mode == "default":
            renderer = GaussianRenderer()
            self.widgets = [
                load_widget_ply.LoadWidget(self, data_path),
                cam_widget.CamWidget(self),
                performance_widget.PerformanceWidget(self),
                video_widget.VideoWidget(self),
                capture_widget.CaptureWidget(self),
                render_widget.RenderWidget(self),
                edit_widget.EditWidget(self),
                eval_widget.EvalWidget(self),
            ]
        elif mode == "attach":
            self.widgets = [
                cam_widget.CamWidget(self),
                performance_widget.PerformanceWidget(self),
                video_widget.VideoWidget(self),
                render_widget.RenderWidget(self),
                edit_widget.EditWidget(self),
                training_widget.TrainingWidget(self),
            ]
            renderer = AttachRenderer(host=host, port=port)
            update_all_the_time = True
        else:
            raise NotImplementedError(f"Mode '{mode}' not recognized.")

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None
        self.eval_result = ""
        self.method_dir = None

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()

    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _set_sizes(self):
        self.pane_w = max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        if self.method_dir is not None and not self.cameras:
            # try to get the cameras now
            self.args.method_dir = self.method_dir
            with open(Path.cwd().parent / self.method_dir / "cameras.json", "r") as file:
                self.cameras = json.load(file)
        self._set_sizes()

        # Control pane
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
        control_pane_flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE
        imgui.begin("##control_pane", p_open=True, flags=control_pane_flags)

        # Display
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if self._tex_obj is not None:
            self.args.tex_pos = [
                pos[0] - (self._tex_obj.width * self._zoom) / 2,
                pos[1] - (self._tex_obj.height * self._zoom) / 2
            ]
        self.args.zoom = self._zoom

        # Widgets
        for widget in self.widgets:
            expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
            imgui.indent()
            widget(expanded)
            imgui.unindent()

        # imgui.show_style_editor()

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result
                if "method_dir" in result:
                    self.method_dir = result.method_dir

        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            # need to store these for later
            self._zoom = zoom
            
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=2,
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)
        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()
