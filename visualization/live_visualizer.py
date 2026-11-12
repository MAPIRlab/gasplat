import os
import shutil
import numpy as np
import pyqtgraph as pg
import subprocess
import queue
import threading
import imageio_ffmpeg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from typing import List, cast

from config import Config
from gs_model import GasSplattingModel
from utils.sim_utils import EnvironmentContext

THEME = {
    # Application & Layout Colors
    'bg': '#f3f4f6',          # Main application background (soft gray)
    'fg': '#374151',          # Primary foreground for default text and labels (dark slate)
    'header_bg': '#ffffff',   # Background color specifically for the top HTML banner
    'border': '#e5e7eb',      # Color for dividers, borders, and layout separators

    # Typography
    'font_family': 'Segoe UI',
    'title': '#1f2937',         # Color for the plot titles
    'title_size': '18pt',         # Font size for plot titles
    'text_light': '#8b92a0',    # Muted text color for the static labels inside the banner
    'axis_text': '#6b7280',     # Color for the numerical tick values on the graphs
    
    # Plotting & Visuals
    'colormap': 'viridis',  # Colour map of the estimated and ground truth maps (any pyqtgraph/matplotlib name)
    'grid_alpha': 0.10,   # Opacity (0.0 to 1.0) of the background grid in the loss plot
    'wall': (180, 180, 180, 255),         # Obstacles on every map
    'beam_pen': (59, 111, 182, 70),       # Beams: thin, translucent blue
    'pose_brush': (59, 111, 182, 255),    # Vantage poses (triangles)
    'ellipse_pen_strong': (255, 255, 255, 70),  # 1-sigma outline of the Gaussians holding most gas
    'ellipse_pen_faint': (255, 255, 255, 25),    # ... and of the weaker ones

    # Accents
    'accent': '#4f46e5',                # Primary brand color (Indigo) for banner numbers AND the loss curve line
    'accent_fill': (79, 70, 229, 30),     # Highly transparent version of the accent color for shading under the loss curve
    
    # Gaussians drawn as their 1-sigma ellipse: only those holding at least ellipse_min of the
    # largest concentration, the ones above ellipse_strong_level of it more opaque
    'ellipse_min': 0.05,
    'ellipse_strong_level': 0.3,
    'ellipse_points': 40,

    # --- Window Dimensions ---
    'window_width_ratio': 0.5,
    'min_window_width': 900,
    'min_window_height': 600,
    'margin': 20,           # Margins around the plots and inside the layout
    'col_min_width': 200    # Minimum width for each of the plot columns
}

class CleanLogAxis(pg.AxisItem):
    """Custom Y-axis that prevents overlapping text on log scales by only labeling 1, 2, and 5."""
    def tickStrings(self, values, scale, spacing):
        strings = super().tickStrings(values, scale, spacing)
        
        cleaned = []
        for v, s in zip(values, strings):
            real_val = 10 ** v
            leading_digit = f"{real_val:.3e}"[0]
            if leading_digit in ['1', '2', '5']:
                cleaned.append(s)
            else:
                cleaned.append("") 
                
        return cleaned

class LiveVisualizer:
    def __init__(self, cfg: Config, environment: EnvironmentContext | None = None):
        """Three panels over the loss curve: the beams measured so far, the estimated map with every
        Gaussian drawn as its 1-sigma ellipse, and the ground truth when there is one. Maps use
        THEME['colormap'], with the obstacles in grey on every panel."""
        self.map_size = cfg.env.map_size
        self.cell_size = cfg.env.cell_size
        self.quiet = cfg.quiet
        ground_truth = environment.ground_truth.gas_map if environment and environment.ground_truth else None
        obstacles = environment.obstacles if environment else None
        # Beams, estimated map and, when there is one, the ground truth
        self.has_ground_truth = ground_truth is not None
        self.num_cols = 3 if self.has_ground_truth else 2

        # Asynchronous frame queue for GIF generation
        self.frame_queue = queue.Queue()
        self.frame_count = 0
        self.temp_dir = "plots/temp_pg_frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Frame writer thread to handle saving frames without blocking the main UI
        self.writer_thread = threading.Thread(target=self._frame_writer, daemon=True)
        self.writer_thread.start()

        self._init_qt()
        self._init_window()
        self.app.processEvents()
        self._setup_plots()
        if ground_truth is not None:
            self._show_ground_truth(ground_truth)
        if obstacles is not None:
            self._show_obstacles(obstacles)
        self.num_beams_shown = 0

    def _frame_writer(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                self.frame_queue.task_done()
                break
            
            frame_path, qimage = item
            # Save the QImage to disk as PNG
            qimage.save(frame_path, "PNG")
            self.frame_queue.task_done()

    def _init_qt(self):
        """Initializes Qt and applies global theme settings."""
        self.app = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        pg.setConfigOption('background', THEME['bg'])
        pg.setConfigOption('foreground', THEME['fg'])
        pg.setConfigOptions(antialias=True)

        self.base_font = QtGui.QFont(THEME['font_family'], 13)
        self.base_font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
        self.app.setFont(self.base_font)

    def _init_window(self):
        """Calculates and centers the main window, now using a central QWidget."""
        # Wrap everything in a main QWidget
        self.main_win = QtWidgets.QWidget()
        self.main_win.setWindowTitle("Real-time Gas Splatting")
        self.layout = QtWidgets.QVBoxLayout(self.main_win)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header Banner
        self.header_banner = QtWidgets.QLabel()
        self.header_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.header_banner.setStyleSheet(f"""
            QLabel {{
                background-color: {THEME['header_bg']};
                font-family: '{THEME['font_family']}';
                font-size: 14pt;
                padding: 18px;
                border-bottom: 2px solid {THEME['border']};
            }}
        """)
        self.layout.addWidget(self.header_banner, stretch=0)

        # Pyqtgraph Canvas
        self.canvas = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.canvas, stretch=1)
        
        # Sizing Logic
        screen_rect = self.app.primaryScreen().availableGeometry()

        target_width = int(screen_rect.width() * THEME['window_width_ratio'])
        single_map_width = target_width / self.num_cols

        map_aspect_ratio = self.map_size[0] / self.map_size[1]
        map_height = single_map_width / map_aspect_ratio

        ideal_height = int(2 * map_height)

        final_width = max(target_width, THEME['min_window_width'])
        final_height = max(ideal_height, THEME['min_window_height'])
        max_height = int(screen_rect.height() * 0.9)
        final_height = min(final_height, max_height)

        self.main_win.resize(final_width, final_height)
        self.main_win.move((screen_rect.width() - final_width) // 2, (screen_rect.height() - final_height) // 2)
        self.main_win.show()

    def _setup_plots(self):
        """Builds the internal widgets and applies the theme."""
        def make_title(text):
            return f'<b style="font-family: {THEME["font_family"]}, sans-serif;">{text}</b>'

        def map_plot(col, title):
            p = self.canvas.addPlot(row=0, col=col)
            p.setTitle(make_title(title), size=THEME['title_size'], color=THEME['title'])
            p.hideAxis('left')
            p.hideAxis('bottom')
            return p

        # Beams measured so far and their vantage poses (Row 0, Col 0)
        self.p_beams = map_plot(0, "Beams")
        self.beam_lines = pg.PlotDataItem(pen=pg.mkPen(color=THEME['beam_pen'], width=1), connect='pairs')
        self.p_beams.addItem(self.beam_lines)
        self.pose_markers = pg.ScatterPlotItem(symbol='t1', size=11, brush=pg.mkBrush(THEME['pose_brush']),
                                               pen=pg.mkPen('w', width=1))
        self.pose_markers.setZValue(2)
        self.p_beams.addItem(self.pose_markers)

        # Estimated map with the Gaussians as ellipses (Row 0, Col 1)
        self.p_map = map_plot(1, "Estimated map")
        self.img_map_item = pg.ImageItem()
        self.img_map_item.setColorMap(THEME['colormap'])
        self.p_map.addItem(self.img_map_item)
        self.ellipses_faint = pg.PlotDataItem(pen=pg.mkPen(color=THEME['ellipse_pen_faint'], width=1), connect='finite')
        self.ellipses_strong = pg.PlotDataItem(pen=pg.mkPen(color=THEME['ellipse_pen_strong'], width=1.2), connect='finite')
        for item in (self.ellipses_faint, self.ellipses_strong):
            item.setZValue(2)
            self.p_map.addItem(item)

        map_plots = [self.p_beams, self.p_map]

        # Ground Truth (Row 0, Col 2), if available
        if self.has_ground_truth:
            self.p_gt = map_plot(2, "Ground truth")
            self.img_gt_item = pg.ImageItem()
            self.img_gt_item.setColorMap(THEME['colormap'])
            self.p_gt.addItem(self.img_gt_item)
            map_plots.append(self.p_gt)

        # Obstacles drawn over every map (gray, transparent elsewhere)
        self.obstacle_items = []
        for p in map_plots:
            item = pg.ImageItem()
            item.setZValue(1)
            p.addItem(item)
            self.obstacle_items.append(item)

        # Loss Curve (Row 1, all columns)
        custom_y_axis = CleanLogAxis(orientation='left')
        self.p_loss = self.canvas.addPlot(row=1, col=0, colspan=self.num_cols, axisItems={'left': custom_y_axis})
        self.p_loss.setTitle(make_title("Loss History"), size=THEME['title_size'], color=THEME['title'])
        self.p_loss.setLabel('bottom', "Iteration")
        self.p_loss.setLogMode(x=False, y=True)
        self.p_loss.showGrid(x=True, y=True, alpha=THEME['grid_alpha'])
        self.p_loss.getAxis('left').enableAutoSIPrefix(False) # Force Y-axis to always show the real value

        axis_font = QtGui.QFont(THEME['font_family'], 10)
        for axis in ['bottom', 'left']:
            self.p_loss.getAxis(axis).setTickFont(axis_font)
            self.p_loss.getAxis(axis).setTextPen(THEME['axis_text'])
        
        self.loss_curve = self.p_loss.plot(
            pen=pg.mkPen(color=THEME['accent'], width=2.5),
            fillBrush=pg.mkBrush(THEME['accent_fill'])
        )
        self.p_loss.setContentsMargins(0, 0, THEME['margin'], 0)

        for p in map_plots:
            # Force the exact physical dimensions
            p.setXRange(0, self.map_size[0], padding=0.0)
            p.setYRange(0, self.map_size[1], padding=0.0)
            p.setAspectLocked(True)

            # Left and right margins
            margin = THEME['margin']
            p.setContentsMargins(margin, 0, margin, 0)
            
            # Prevent the plots from independently resizing themselves 
            # when dynamic data (Gaussians/Estimated Map) gets close to the edges.
            p.getViewBox().disableAutoRange()
            
            # Disable user interaction completely
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons() # hide auto-scale button

        main_layout = self.canvas.ci.layout

        if main_layout is not None:
            total_w = max(0, int(self.canvas.size().width()))
            col_w = max(THEME['col_min_width'], int(total_w / self.num_cols) - 2 * THEME['margin'])
            for col in range(self.num_cols):
                main_layout.setColumnMinimumWidth(col, col_w)

    def _show_ground_truth(self, ground_truth: np.ndarray):
        """Sets the static ground truth image and stores its levels to lock the color scale."""
        self.img_gt_item.setImage(ground_truth.T, autoLevels=True)
        self.img_gt_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))
        # Save levels and apply them to both images so both maps use identical color scale
        self.gt_levels = (float(ground_truth.min()), float(ground_truth.max()))
        self.img_gt_item.setLevels(self.gt_levels)
        self.img_map_item.setLevels(self.gt_levels)

    def _show_obstacles(self, obstacles: np.ndarray):
        """Overlays the obstacle cells (H, W) on every map panel."""
        rgba = np.zeros(obstacles.shape + (4,), dtype=np.ubyte)
        rgba[obstacles > 0.5] = THEME['wall']
        for item in self.obstacle_items:
            item.setImage(rgba.transpose(1, 0, 2), levels=(0, 255))
            item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

    def _show_beams(self, beams: np.ndarray):
        """Draws the beams (N, 2, 2) in meters and, for sweeps from vantage poses, the poses: the
        common origins of the beams (procedural beams start at distinct points and show no pose)."""
        self.beam_lines.setData(beams[:, :, 0].ravel(), beams[:, :, 1].ravel())
        origins = np.unique(np.round(beams[:, 0], 3), axis=0)
        if len(origins) <= len(beams) / 4:
            self.pose_markers.setData(origins[:, 0], origins[:, 1])
        self.num_beams_shown = len(beams)

    def _ellipse_lines(self, pos: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """1-sigma ellipses of the Gaussians as one polyline, separated by NaN (drawn with connect='finite')."""
        if len(pos) == 0:
            return np.array([]), np.array([])
        t = np.linspace(0, 2 * np.pi, THEME['ellipse_points'])
        circle = np.stack([np.cos(t), np.sin(t)])                         # (2, T)
        w, v = np.linalg.eigh(cov)                                        # cov = V diag(w) V^T
        points = pos[:, :, None] + (v * np.sqrt(np.maximum(w, 0))[:, None, :]) @ circle   # (K, 2, T)
        points = np.concatenate([points, np.full((len(pos), 2, 1), np.nan)], axis=2)
        return points[:, 0].ravel(), points[:, 1].ravel()

    def update(self, iteration: int, loss_history: List[float], model: GasSplattingModel, beams=None):
        """Extracts the latest parameters from the model, updates the UI components, and saves the frame state.
        beams: the (N, 2, 2) beams trained on, redrawn when their number changes (a new batch)."""
        # Update Loss
        valid_losses = [l for l in loss_history if not np.isnan(l) and not np.isinf(l)]
        x_data = np.arange(len(valid_losses))
        if valid_losses:
            min_loss = min(valid_losses)
            min_loss_log = np.log10(min_loss)
            safe_bottom = min_loss_log - 0.1 
            
            self.loss_curve.setFillLevel(safe_bottom)
            self.loss_curve.setData(x_data, valid_losses)

        if beams is not None and len(beams) != self.num_beams_shown:
            self._show_beams(beams.detach().cpu().numpy())

        # Extract tensors
        pos = model.get_pos().detach().cpu().numpy()
        cov = model.get_covariance().detach().cpu().numpy()
        conc = model.get_concentration().detach().cpu().numpy()

        # Update the Gaussians' ellipses: only those holding visible gas, the strongest more opaque
        relative = conc / max(float(conc.max()), 1e-12) if len(conc) else conc
        strong = relative >= THEME['ellipse_strong_level']
        faint = (relative >= THEME['ellipse_min']) & ~strong
        self.ellipses_strong.setData(*self._ellipse_lines(pos[strong], cov[strong]))
        self.ellipses_faint.setData(*self._ellipse_lines(pos[faint], cov[faint]))

        # Update Header Banner
        accent = THEME['accent']
        text_color = THEME['text_light']
        spacer = '&nbsp;' * 10
        
        formatted_text = (
            f'<span style="color: {text_color};">Iteration:</span> '
            f'<b style="color: {accent};">{iteration}</b>{spacer}'
            f'<span style="color: {text_color};">Gaussians:</span> '
            f'<b style="color: {accent};">{len(pos)}</b>'
        )
        self.header_banner.setText(formatted_text)

        # Update Rendered Map
        map_data = model.render_map(cell_size=self.cell_size).T
        if hasattr(self, 'gt_levels'):
            self.img_map_item.setImage(map_data, levels=self.gt_levels)
        else:
            self.img_map_item.setImage(map_data, autoLevels=True)
        self.img_map_item.setRect(QtCore.QRectF(0, 0, self.map_size[0], self.map_size[1]))

        self.app.processEvents()

        # Frame Capture for GIF Generation
        qimage = self.main_win.grab().toImage()
        frame_path = os.path.join(self.temp_dir, f"frame_{self.frame_count:05d}.png")
        
        # Send the frame to the writer thread via the queue
        self.frame_queue.put((frame_path, qimage))
        self.frame_count += 1

    def save_gif(self, filepath):
        """Waits for all frames to be written, then generates a GIF from the captured frames using ffmpeg."""
        if self.frame_count == 0:
            print("Could not generate GIF. No frames captured.")
            return
            
        if not self.quiet: print("Waiting for frame capture to complete...")
        # Send stop signal to the writer thread and wait for it to finish
        self.frame_queue.put(None)
        self.frame_queue.join()
        self.writer_thread.join()

        if not self.quiet: print(f"Generating GIF from {self.frame_count} frames...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create GIF from saved frames using ffmpeg
        input_pattern = f"{self.temp_dir}/frame_%05d.png"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg_exe, '-y', '-framerate', '10', 
            '-loglevel', 'error',
            '-i', input_pattern,
            '-filter_complex', 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            '-loop', '0', 
            filepath
        ], check=True)
        print(f"[+] Training GIF saved in: {filepath}")
            
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.main_win.close()
        self.main_win.deleteLater()
        self.app.processEvents()
        self.app.quit()