import multiprocessing
import tkinter as tk
import ctypes
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import ImageGrab

# --- Windows DPI 适配，防止高分屏下坐标偏移 ---
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

def _snipping_process(queue):
    """独立的进程运行 Tkinter，防止阻塞 FastAPI 的异步事件循环"""
    class SnippingTool:
        def __init__(self):
            self.root = tk.Tk()
            self.root.attributes('-alpha', 0.3) # 透明度
            self.root.attributes('-fullscreen', True)
            self.root.attributes('-topmost', True)
            self.root.config(cursor="cross")
            
            self.canvas = tk.Canvas(self.root, cursor="cross", bg="black")
            self.canvas.pack(fill="both", expand=True)
            
            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
            
            self.start_x = self.start_y = 0
            self.rect = None

        def on_press(self, event):
            self.start_x, self.start_y = event.x, event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='#00ffcc', width=2, fill="#333333")

        def on_drag(self, event):
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

        def on_release(self, event):
            # 获取绝对坐标 (x1, y1, x2, y2)
            bbox = (min(self.start_x, event.x), min(self.start_y, event.y),
                    max(self.start_x, event.x), max(self.start_y, event.y))
            queue.put(bbox)
            self.root.quit()

    app = SnippingTool()
    app.root.mainloop()
    app.root.destroy()

def extract_board_from_memory(bbox):
    """截图棋盘区域，识别 8x8 格哪些格子有方块（1），哪些为空（0）。"""
    pil_img = ImageGrab.grab(bbox=bbox)
    img_np = np.array(pil_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    debug_run_id = _new_debug_run_id("board")
    _save_debug_capture(img_cv2)
    _save_tmp_step(debug_run_id, "01_capture_bgr", img_cv2)

    h, w = img_cv2.shape[:2]
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    _save_tmp_step(debug_run_id, "02_hsv_as_bgr", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # 棋盘背景为紫色（低饱和/低亮度），实际方块为高饱和高亮度的鲜艳色彩。
    # 先用宽松阈值生成候选掩码，再在格内用加权得分而非简单二值化。
    # 宽掩码（采集调试热力图用）：S>55 & V>55
    colored_loose = ((sat > 55) & (val > 55)).astype(np.float32)
    _save_tmp_step(debug_run_id, "03_colored_loose", colored_loose)

    # 严格掩码（用于实际判断）：S>110 & V>130，过滤掉背景紫色
    colored_strict = ((sat > 110) & (val > 130)).astype(np.float32)
    _save_tmp_step(debug_run_id, "03b_colored_strict", colored_strict)

    cell_h = h / 8.0
    cell_w = w / 8.0
    margin = 0.18  # 每格四周留 18% 边距，避免格线噪声

    # 用于调试的逐格 fill_ratio 热力图
    heatmap = np.zeros((h, w), dtype=np.float32)

    board = []
    for r in range(8):
        row = []
        for c in range(8):
            y1 = int(r * cell_h + cell_h * margin)
            y2 = int((r + 1) * cell_h - cell_h * margin)
            x1 = int(c * cell_w + cell_w * margin)
            x2 = int((c + 1) * cell_w - cell_w * margin)
            roi = colored_strict[y1:y2, x1:x2]
            fill_ratio = float(roi.mean()) if roi.size > 0 else 0.0
            heatmap[y1:y2, x1:x2] = fill_ratio
            row.append(1 if fill_ratio > 0.20 else 0)
        board.append(row)

    _save_tmp_step(debug_run_id, "04_fill_heatmap", heatmap)

    # 最终棋盘可视化
    board_img = np.array(board, dtype=np.uint8) * 255
    board_vis = cv2.resize(board_img, (w, h), interpolation=cv2.INTER_NEAREST)
    _save_tmp_step(debug_run_id, "05_board_binary", board_vis)

    # 在原图上叠加格子与识别结果，方便肉眼校验
    overlay = img_cv2.copy()
    for r in range(8):
        for c in range(8):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)
            color = (0, 220, 80) if board[r][c] else (60, 60, 60)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(overlay, str(board[r][c]), (x1 + 4, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    _save_tmp_step(debug_run_id, "06_board_overlay", overlay)

    return board


def get_screen_bbox():
    """唤起屏幕截取UI并返回坐标"""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_snipping_process, args=(q,))
    p.start()
    p.join()
    if not q.empty():
        return q.get()
    return None


def _save_debug_capture(img_cv2):
    """将当前识别截图保存到项目根目录，最多保留 3 张（新图覆盖最旧图）。"""
    root_dir = Path(__file__).resolve().parent
    slots = [root_dir / f"cv_capture_{i}.png" for i in range(1, 4)]

    target_path = None
    for path in slots:
        if not path.exists():
            target_path = path
            break

    if target_path is None:
        target_path = min(slots, key=lambda p: p.stat().st_mtime)

    cv2.imwrite(str(target_path), img_cv2)
    return target_path


def _new_debug_run_id(tag):
    """为每次识别生成唯一调试 run id。"""
    return f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _to_debug_image(img):
    """将 bool/float/灰度图统一转换为可写入 PNG 的 uint8 图像。"""
    if img is None:
        return None

    arr = np.asarray(img)
    if arr.size == 0:
        return None

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    if arr.dtype.kind == 'f':
        max_v = float(np.max(arr))
        if max_v <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _save_tmp_step(run_id, step_name, img):
    """将识别流程中的中间图像保存到项目 tmp 目录。"""
    debug_img = _to_debug_image(img)
    if debug_img is None:
        return None

    root_dir = Path(__file__).resolve().parent
    tmp_dir = root_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_step = "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in step_name)
    file_path = tmp_dir / f"{run_id}_{safe_step}.png"
    cv2.imwrite(str(file_path), debug_img)
    return file_path


def _draw_blob_debug(base_img, blobs, color=(0, 255, 0)):
    """在原图上绘制候选框，便于核对 contour 过滤结果。"""
    vis = base_img.copy()
    for b in blobs:
        x, y, w, h = int(b['x']), int(b['y']), int(b['w']), int(b['h'])
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    return vis


def _draw_cell_candidates_debug(base_img, cells):
    """在原图上绘制 cell 候选中心与估计框。"""
    vis = base_img.copy()
    for c in cells:
        cx, cy = int(c['cx']), int(c['cy'])
        w, h = int(c.get('w', 0)), int(c.get('h', 0))
        if w > 0 and h > 0:
            x1 = int(round(cx - w / 2.0))
            y1 = int(round(cy - h / 2.0))
            cv2.rectangle(vis, (x1, y1), (x1 + w, y1 + h), (0, 220, 255), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
    return vis


def _draw_groups_debug(base_img, groups):
    """将最终分组结果上色标记，方便观察分组是否正确。"""
    vis = base_img.copy()
    palette = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
    ]

    for gi, group in enumerate(groups):
        color = palette[gi % len(palette)]
        for s in group:
            cx, cy = int(s['cx']), int(s['cy'])
            cv2.circle(vis, (cx, cy), 4, color, -1)
            cv2.putText(vis, str(gi), (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis
# ── 三个投影工具函数（放在 extract_blocks_from_memory 之前）──────────────

def _proj_runs(bool_arr):
    """布尔数组 → 连续 True 区段列表 [(start, end), ...]"""
    runs, start = [], None
    for i, v in enumerate(bool_arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(bool_arr)))
    return runs


def _merge_runs(runs, max_gap=10):
    """合并间距 ≤ max_gap 的相邻区段（处理格间细缝）"""
    if not runs:
        return runs
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


def _proj_peak_centers(proj, threshold, min_gap=8):
    """1D 投影 → 超过 threshold 的每个连续区段的中心坐标列表"""
    if proj.max() == 0:
        return []
    runs = _proj_runs(proj > threshold)
    centers = []
    for s, e in runs:
        c = (s + e) // 2
        if centers and c - centers[-1] < min_gap:
            centers[-1] = (centers[-1] + c) // 2  # 过近则合并
        else:
            centers.append(c)
    return centers


def _hue_circular_distance(h1, h2):
    """HSV Hue 在 [0,180) 空间中的环形距离。"""
    d = abs(float(h1) - float(h2))
    return min(d, 180.0 - d)


def _recover_vertical_tail_cell(crop_img, tail_img, squares, coords, base_side, debug_run_id, slot_idx):
    """针对 4x1 竖条的补偿：在 slot 下方 tail 区检测是否存在被挤出的第 5 格。"""
    if tail_img is None or tail_img.size == 0:
        return coords

    if len(coords) != 4:
        return coords

    cols = {c for _, c in coords}
    if len(cols) != 1:
        return coords

    rows = sorted(r for r, _ in coords)
    if rows != [0, 1, 2, 3]:
        return coords

    crop_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    h_samples = []
    for sq in squares:
        cx, cy = int(sq['cx']), int(sq['cy'])
        w, h = int(sq.get('w', base_side)), int(sq.get('h', base_side))
        hx = max(4, int(round(w * 0.3)))
        hy = max(4, int(round(h * 0.3)))
        x1 = max(cx - hx, 0)
        x2 = min(cx + hx, crop_hsv.shape[1])
        y1 = max(cy - hy, 0)
        y2 = min(cy + hy, crop_hsv.shape[0])
        roi = crop_hsv[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        keep = (roi[:, :, 1] > 70) & (roi[:, :, 2] > 70)
        if np.any(keep):
            h_samples.append(roi[:, :, 0][keep])

    if not h_samples:
        return coords

    piece_hue = float(np.median(np.concatenate(h_samples)))

    tail_hsv = cv2.cvtColor(tail_img, cv2.COLOR_BGR2HSV)
    tail_h, tail_w = tail_hsv.shape[:2]
    search_h = min(tail_h, max(8, int(round(base_side * 0.9))))

    tail_hue = tail_hsv[:search_h, :, 0].astype(np.float32)
    tail_sat = tail_hsv[:search_h, :, 1]
    tail_val = tail_hsv[:search_h, :, 2]

    hue_dist = np.abs(tail_hue - piece_hue)
    hue_dist = np.minimum(hue_dist, 180.0 - hue_dist)
    tail_mask = np.where((tail_sat > 70) & (tail_val > 70) & (hue_dist <= 14.0), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    tail_mask = cv2.medianBlur(tail_mask, 3)
    tail_mask = cv2.morphologyEx(tail_mask, cv2.MORPH_OPEN, kernel)

    cx_med = int(round(float(np.median([s['cx'] for s in squares]))))
    band_half = max(8, int(round(base_side * 0.75)))
    bx1 = max(cx_med - band_half, 0)
    bx2 = min(cx_med + band_half, tail_w)

    band_mask = np.zeros_like(tail_mask)
    band_mask[:, bx1:bx2] = tail_mask[:, bx1:bx2]
    _save_tmp_step(debug_run_id, f"06_slot{slot_idx}_tail_band_mask", band_mask)

    contours, _ = cv2.findContours(band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        fill_ratio = area / float(max(w * h, 1))

        if w < max(8, int(round(base_side * 0.45))):
            continue
        if h < max(6, int(round(base_side * 0.18))) or h > int(round(base_side * 0.8)):
            continue
        if area < (base_side * base_side * 0.08):
            continue
        if fill_ratio < 0.35:
            continue

        next_row = max(r for r, _ in coords) + 1
        next_col = coords[0][1]
        repaired = sorted(coords + [[next_row, next_col]], key=lambda t: (t[0], t[1]))
        return repaired

    return coords


def _recover_vertical_by_projection(mask, squares, coords, base_side, debug_run_id, slot_idx):
    """在主 mask 上做列投影：当 4x1 因底部粘连漏成 4 格时，恢复成 5x1。"""
    if len(coords) != 4:
        return coords

    cols = {c for _, c in coords}
    if len(cols) != 1:
        return coords

    rows = sorted(r for r, _ in coords)
    if rows != [0, 1, 2, 3]:
        return coords

    if not squares:
        return coords

    cx_med = int(round(float(np.median([s['cx'] for s in squares]))))
    band_half = max(8, int(round(base_side * 0.7)))
    bx1 = max(cx_med - band_half, 0)
    bx2 = min(cx_med + band_half, mask.shape[1])
    if bx2 - bx1 < 6:
        return coords

    band = (mask[:, bx1:bx2] > 0).astype(np.uint8)
    y_proj = band.sum(axis=1)
    y_threshold = max(8, int(round((bx2 - bx1) * 0.55)))
    runs = _proj_runs(y_proj > y_threshold)
    runs = [r for r in runs if (r[1] - r[0]) >= max(6, int(round(base_side * 0.45)))]
    _save_tmp_step(debug_run_id, f"06_slot{slot_idx}_projection_band", band * 255)

    if len(runs) < 5:
        return coords

    runs = sorted(runs, key=lambda t: t[0])[:5]
    run_centers = [0.5 * (s + e) for s, e in runs]
    run_lens = [e - s for s, e in runs]

    sq_centers = sorted(float(s['cy']) for s in squares)[:4]
    if len(sq_centers) < 4:
        return coords

    align_err = float(np.mean([abs(run_centers[i] - sq_centers[i]) for i in range(4)]))
    if align_err > base_side * 0.45:
        return coords

    step = float(np.median(np.diff(sq_centers))) if len(sq_centers) >= 2 else base_side
    if step <= 1.0:
        return coords

    gap = run_centers[4] - sq_centers[-1]
    if gap < step * 0.55 or gap > step * 1.7:
        return coords

    if not all(base_side * 0.55 <= l <= base_side * 1.6 for l in run_lens):
        return coords

    next_row = max(r for r, _ in coords) + 1
    next_col = coords[0][1]
    repaired = sorted(coords + [[next_row, next_col]], key=lambda t: (t[0], t[1]))
    return repaired


def _detect_blocks_in_crop(crop_img, debug_run_id, slot_idx, tail_img=None, return_details=False):
    """对单个候选区裁图进行方块识别。

    默认返回归一化坐标 [[row, col], ...]；
    当 return_details=True 时，返回 {"coords": ..., "cell_points": ...}。
    """
    img_h, img_w = crop_img.shape[:2]

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = np.where((sat > 70) & (val > 70), 255, 0).astype(np.uint8)
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    _em = 2
    mask[:_em, :] = 0;  mask[-_em:, :] = 0
    mask[:, :_em] = 0;  mask[:, -_em:] = 0
    _save_tmp_step(debug_run_id, f"03_slot{slot_idx}_mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 裁图内最小边：按裁图宽度估算（单格约占 1/5 宽）
    min_side = max(int(img_w * 0.07), 10)
    # 最大边宽松：允许长条方块跨满裁图宽/高
    max_side = int(max(img_h, img_w) * 0.95)

    raw_blobs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        short_side = min(w, h)
        fill_ratio = area / float(max(w * h, 1))

        if short_side < min_side:
            continue
        if area < (min_side * min_side * 0.35):
            continue
        if fill_ratio < 0.55:
            continue
        if area > (img_h * img_w * 0.85):
            continue

        raw_blobs.append({"x": x, "y": y, "w": w, "h": h})

    _save_tmp_step(debug_run_id, f"04_slot{slot_idx}_raw_blobs", _draw_blob_debug(crop_img, raw_blobs))

    if not raw_blobs:
        return []

    base_side = float(np.median([min(b['w'], b['h']) for b in raw_blobs]))
    merge_ratio = 1.45
    cell_candidates = []

    def _est_split(long_side, short_side):
        if short_side > base_side * 1.6:
            return 0
        if long_side > base_side * 8.5:
            return 0
        split_n = int(round(long_side / max(base_side, 1.0)))
        return max(2, min(8, split_n))

    for blob in raw_blobs:
        x, y, w, h = blob['x'], blob['y'], blob['w'], blob['h']
        side = max(w, h)

        if abs(w - h) <= max(6, int(side * 0.25)):
            cell_candidates.append({"cx": x + w // 2, "cy": y + h // 2, "w": w, "h": h})
            continue

        if w > h * merge_ratio and h >= base_side * 0.7:
            split_n = _est_split(w, h)
            if split_n == 0:
                continue
            step = w / split_n
            for i in range(split_n):
                cx = int(round(x + (i + 0.5) * step))
                cell_candidates.append({"cx": cx, "cy": y + h // 2, "w": int(round(step)), "h": h})
            continue

        if h > w * merge_ratio and w >= base_side * 0.7:
            split_n = _est_split(h, w)
            if split_n == 0:
                continue
            step = h / split_n
            for i in range(split_n):
                cy = int(round(y + (i + 0.5) * step))
                cell_candidates.append({"cx": x + w // 2, "cy": cy, "w": w, "h": int(round(step))})

    # 二次补偿：高亮阈值捡回被背景吞并的方格
    bright_mask = np.where((sat > 120) & (val > 200), 255, 0).astype(np.uint8)
    bright_mask = cv2.medianBlur(bright_mask, 3)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in bright_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        side = max(w, h)
        short_side = min(w, h)
        fill_ratio = area / float(max(w * h, 1))

        if short_side < min_side or side > max_side:
            continue
        if abs(w - h) > max(6, int(side * 0.25)):
            continue
        if area < (min_side * min_side * 0.35) or fill_ratio < 0.75:
            continue

        cell_candidates.append({"cx": x + w // 2, "cy": y + h // 2, "w": w, "h": h})

    if not cell_candidates:
        return []

    # 去重
    dedup_threshold = max(4.0, base_side * 0.35)
    cell_candidates.sort(key=lambda s: (s['cy'], s['cx']))
    squares = []
    for cand in cell_candidates:
        if not any(abs(cand['cx'] - k['cx']) <= dedup_threshold
                   and abs(cand['cy'] - k['cy']) <= dedup_threshold
                   for k in squares):
            squares.append(cand)

    _save_tmp_step(debug_run_id, f"05_slot{slot_idx}_squares", _draw_cell_candidates_debug(crop_img, squares))

    if not squares:
        return []

    # 归一化坐标
    min_x = min(s['cx'] for s in squares)
    min_y = min(s['cy'] for s in squares)

    x_vals = sorted(s['cx'] for s in squares)
    y_vals = sorted(s['cy'] for s in squares)
    x_diffs = [x_vals[i + 1] - x_vals[i] for i in range(len(x_vals) - 1) if x_vals[i + 1] - x_vals[i] > 2]
    y_diffs = [y_vals[i + 1] - y_vals[i] for i in range(len(y_vals) - 1) if y_vals[i + 1] - y_vals[i] > 2]

    grid_step = base_side * 1.15
    step_x = max(float(np.median(x_diffs)) if x_diffs else grid_step, grid_step * 0.75)
    step_y = max(float(np.median(y_diffs)) if y_diffs else grid_step, grid_step * 0.75)

    coord_set = set()
    cell_point_map = {}
    for sq in squares:
        col = int(round((sq['cx'] - min_x) / step_x))
        row = int(round((sq['cy'] - min_y) / step_y))
        coord_set.add((row, col))
        cell_point_map.setdefault((row, col), []).append((sq['cx'], sq['cy']))

    coords = [[r, c] for (r, c) in sorted(coord_set, key=lambda t: (t[0], t[1]))]
    coords = _recover_vertical_by_projection(mask, squares, coords, base_side, debug_run_id, slot_idx)
    coords = _recover_vertical_tail_cell(crop_img, tail_img, squares, coords, base_side, debug_run_id, slot_idx)

    if not return_details:
        return coords

    # 记录每个归一化格子的局部像素中心；补偿新增格子用步长估算中心。
    cell_points = []
    for row, col in coords:
        samples = cell_point_map.get((row, col), [])
        if samples:
            avg_x = int(round(float(np.mean([p[0] for p in samples]))))
            avg_y = int(round(float(np.mean([p[1] for p in samples]))))
        else:
            avg_x = int(round(min_x + col * step_x))
            avg_y = int(round(min_y + row * step_y))
        cell_points.append({"row": int(row), "col": int(col), "cx": avg_x, "cy": avg_y})

    return {"coords": coords, "cell_points": cell_points}


def extract_blocks_from_memory(bbox):
    """静默截图，先按固定比例裁切 4 个候选区，再分别识别归一化坐标。

    布局规律：
      - 每个候选区为正方形，高度 = 图片宽度
      - 无用区高度 = (总高 - 宽 × 4) / 3
      - 顺序：候选区 - 无用区 - 候选区 - 无用区 - 候选区 - 无用区 - 候选区
    """
    pil_img = ImageGrab.grab(bbox=bbox)
    img_np = np.array(pil_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    debug_run_id = _new_debug_run_id("blocks")
    _save_debug_capture(img_cv2)
    _save_tmp_step(debug_run_id, "01_capture_bgr", img_cv2)

    img_h, img_w = img_cv2.shape[:2]
    slot_h = img_w                              # 每个候选区高度 = 图片宽度
    gap_h = (img_h - slot_h * 4) / 3.0         # 无用区高度

    # 在全图上标注裁切线，方便调试
    crop_vis = img_cv2.copy()
    final_blocks = []

    # 仅将下方 tail 作为“4x1->5x1”补偿输入，主识别仍使用纯正方形候选区，避免污染。
    extra_h = int(slot_h * 0.35)

    for i in range(4):
        y1 = int(round(i * (slot_h + gap_h)))
        y2_base = int(round(y1 + slot_h))
        y2_tail = min(y2_base + extra_h, img_h)
        # 调试图：原始 slot 边界用青色，tail 区用橙色
        cv2.rectangle(crop_vis, (0, y1), (img_w - 1, y2_base), (0, 255, 255), 2)
        cv2.rectangle(crop_vis, (0, y2_base), (img_w - 1, y2_tail), (0, 128, 255), 1)
        cv2.putText(crop_vis, f"slot{i}", (4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        crop = img_cv2[y1:y2_base, 0:img_w]
        tail = img_cv2[y2_base:y2_tail, 0:img_w]
        _save_tmp_step(debug_run_id, f"02_crop_{i}", crop)
        _save_tmp_step(debug_run_id, f"02_tail_{i}", tail)

        blocks = _detect_blocks_in_crop(crop, debug_run_id, i, tail_img=tail)
        final_blocks.append(blocks)

    _save_tmp_step(debug_run_id, "06_crop_regions", crop_vis)

    return final_blocks


def extract_blocks_with_screen_points(bbox):
    """识别 4 个槽位并返回每个格子的屏幕坐标，用于拖放执行器。"""
    pil_img = ImageGrab.grab(bbox=bbox)
    img_np = np.array(pil_img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    debug_run_id = _new_debug_run_id("exec")

    img_h, img_w = img_cv2.shape[:2]
    slot_h = img_w
    gap_h = (img_h - slot_h * 4) / 3.0
    extra_h = int(slot_h * 0.35)

    abs_x1 = int(bbox[0])
    abs_y1 = int(bbox[1])

    slot_details = []
    for i in range(4):
        y1 = int(round(i * (slot_h + gap_h)))
        y2_base = int(round(y1 + slot_h))
        y2_tail = min(y2_base + extra_h, img_h)

        crop = img_cv2[y1:y2_base, 0:img_w]
        tail = img_cv2[y2_base:y2_tail, 0:img_w]
        detail = _detect_blocks_in_crop(
            crop,
            debug_run_id,
            i,
            tail_img=tail,
            return_details=True,
        )

        coords = detail["coords"]
        cell_points = []
        for p in detail["cell_points"]:
            cell_points.append({
                "row": int(p["row"]),
                "col": int(p["col"]),
                "x": int(abs_x1 + p["cx"]),
                "y": int(abs_y1 + y1 + p["cy"]),
            })

        slot_details.append({
            "slot_index": i,
            "coords": coords,
            "cell_points": cell_points,
            "slot_bbox": [
                int(abs_x1),
                int(abs_y1 + y1),
                int(abs_x1 + img_w),
                int(abs_y1 + y2_base),
            ],
        })

    return slot_details