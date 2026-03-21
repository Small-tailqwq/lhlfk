import ctypes
import time
from typing import Any, Dict, List, Sequence, Tuple

from vision_core import extract_blocks_with_screen_points


TIMING_PROFILES: Dict[str, Dict[str, float]] = {
    # 优先稳定性，避免游戏漏收鼠标拖放事件。
    "safe": {
        "pre_press_sec": 0.05,
        "hold_before_drag_sec": 0.08,
        "travel_sec": 0.24,
        "hold_after_arrive_sec": 0.06,
        "release_settle_sec": 0.12,
        "inter_step_sec": 0.28,
        "min_steps": 9.0,
        "px_per_step": 45.0,
    },
    "balanced": {
        "pre_press_sec": 0.03,
        "hold_before_drag_sec": 0.05,
        "travel_sec": 0.17,
        "hold_after_arrive_sec": 0.04,
        "release_settle_sec": 0.08,
        "inter_step_sec": 0.16,
        "min_steps": 7.0,
        "px_per_step": 60.0,
    },
    "fast": {
        "pre_press_sec": 0.015,
        "hold_before_drag_sec": 0.02,
        "travel_sec": 0.10,
        "hold_after_arrive_sec": 0.015,
        "release_settle_sec": 0.05,
        "inter_step_sec": 0.05,
        "min_steps": 4.0,
        "px_per_step": 110.0,
    },
}


def is_running_as_admin() -> bool:
    """Windows 下检查当前进程是否以管理员权限运行。"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _normalize_coords(coords: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for p in coords:
        if len(p) < 2:
            continue
        points.append((int(p[0]), int(p[1])))

    if not points:
        return []

    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    normalized = [(r - min_r, c - min_c) for r, c in points]
    return sorted(set(normalized), key=lambda t: (t[0], t[1]))


def choose_piece_anchor(coords: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """按块形状选择拖拽锚点（抓取点对应的格子）。"""
    norm = _normalize_coords(coords)
    if not norm:
        return (0, 0)

    row_max = max(r for r, _ in norm)
    col_max = max(c for _, c in norm)
    h = row_max + 1
    w = col_max + 1

    # 规则优先：匹配用户观察到的人类拖拽焦点。
    if h % 2 == 1 and w % 2 == 1:
        candidate = (h // 2, w // 2)
    elif h % 2 == 0 and w % 2 == 1:
        candidate = (h // 2 - 1, w // 2)
    elif h % 2 == 1 and w % 2 == 0:
        candidate = (h // 2, w // 2 - 1)
    else:
        # 偶数 x 偶数：取中心四格中的左上（2x2 时即 [0,0]）。
        candidate = (h // 2 - 1, w // 2 - 1)

    norm_set = set(norm)
    if candidate in norm_set:
        return candidate

    # 奇数 x 奇数但中心缺失时，优先取中心列中最靠上的真实方块。
    if h % 2 == 1 and w % 2 == 1:
        center_col = w // 2
        col_hits = [rc for rc in norm if rc[1] == center_col]
        if col_hits:
            return min(col_hits, key=lambda rc: (rc[0], rc[1]))

    # 回退：取离几何中心最近的真实格子。
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    return min(
        norm,
        key=lambda rc: (
            (rc[0] - center_r) ** 2 + (rc[1] - center_c) ** 2,
            abs(rc[0] - center_r),
            abs(rc[1] - center_c),
            rc[0],
            rc[1],
        ),
    )


def _find_source_point(slot_detail: Dict[str, Any], anchor: Tuple[int, int]) -> Tuple[int, int]:
    points = slot_detail.get("cell_points", [])
    if not points:
        x1, y1, x2, y2 = slot_detail.get("slot_bbox", [0, 0, 0, 0])
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    ar, ac = anchor
    exact = [p for p in points if int(p.get("row", -999)) == ar and int(p.get("col", -999)) == ac]
    if exact:
        return (int(exact[0]["x"]), int(exact[0]["y"]))

    nearest = min(
        points,
        key=lambda p: (
            abs(int(p.get("row", 0)) - ar) + abs(int(p.get("col", 0)) - ac),
            abs(int(p.get("row", 0)) - ar),
            abs(int(p.get("col", 0)) - ac),
        ),
    )
    return (int(nearest["x"]), int(nearest["y"]))


def _board_cell_center(board_bbox: Sequence[int], row: int, col: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = [int(v) for v in board_bbox]
    cell_w = (x2 - x1) / 8.0
    cell_h = (y2 - y1) / 8.0
    x = int(round(x1 + (col + 0.5) * cell_w))
    y = int(round(y1 + (row + 0.5) * cell_h))
    return (x, y)


def _load_mouse_backend(preferred: str):
    preferred = (preferred or "auto").lower()
    errors: List[str] = []

    if preferred in ("auto", "direct", "pydirectinput"):
        try:
            import pydirectinput  # type: ignore

            pydirectinput.PAUSE = 0

            return {
                "name": "pydirectinput",
                "move_to": lambda x, y: pydirectinput.moveTo(int(x), int(y)),
                "mouse_down": lambda: pydirectinput.mouseDown(),
                "mouse_up": lambda: pydirectinput.mouseUp(),
            }
        except Exception as exc:
            errors.append(f"pydirectinput 不可用: {exc}")

    if preferred in ("auto", "pyautogui", "win32"):
        try:
            import pyautogui  # type: ignore

            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False

            return {
                "name": "pyautogui",
                "move_to": lambda x, y: pyautogui.moveTo(int(x), int(y)),
                "mouse_down": lambda: pyautogui.mouseDown(),
                "mouse_up": lambda: pyautogui.mouseUp(),
            }
        except Exception as exc:
            errors.append(f"pyautogui 不可用: {exc}")

    joined = " | ".join(errors) if errors else "未匹配到可用输入后端"
    raise RuntimeError(joined)


def _resolve_timing_profile(profile_name: str) -> Dict[str, float]:
    key = str(profile_name or "safe").lower()
    if key not in TIMING_PROFILES:
        key = "safe"
    return TIMING_PROFILES[key]


def _drag_mouse(backend: Dict[str, Any], src: Tuple[int, int], dst: Tuple[int, int], timing: Dict[str, float]) -> None:
    sx, sy = src
    dx, dy = dst

    backend["move_to"](sx, sy)
    time.sleep(float(timing["pre_press_sec"]))
    backend["mouse_down"]()
    time.sleep(float(timing["hold_before_drag_sec"]))

    px_span = max(abs(dx - sx), abs(dy - sy))
    min_steps = int(round(float(timing["min_steps"])))
    px_per_step = max(float(timing["px_per_step"]), 1.0)
    steps = max(min_steps, int(px_span / px_per_step))
    step_sleep = max(float(timing["travel_sec"]) / max(steps, 1), 0.006)

    for i in range(1, steps + 1):
        nx = int(round(sx + (dx - sx) * i / steps))
        ny = int(round(sy + (dy - sy) * i / steps))
        backend["move_to"](nx, ny)
        time.sleep(step_sleep)

    time.sleep(float(timing["hold_after_arrive_sec"]))
    backend["mouse_up"]()
    time.sleep(float(timing["release_settle_sec"]))


def execute_solution_steps(
    steps: List[Dict[str, Any]],
    capture_bbox: Sequence[int],
    board_bbox: Sequence[int],
    backend: str = "auto",
    dry_run: bool = False,
    timing_profile: str = "safe",
) -> Dict[str, Any]:
    if not dry_run and not is_running_as_admin():
        return {
            "status": "fail",
            "msg": (
                "执行器需要管理员权限：当前 Python 进程未提升权限。"
                "当游戏以管理员身份运行时，Windows 会拦截低权限进程的鼠标注入。"
                "请以管理员身份启动 VS Code/终端后再运行后端。"
            ),
        }

    if len(capture_bbox) != 4 or len(board_bbox) != 4:
        return {"status": "fail", "msg": "执行器参数无效：bbox 必须是 4 元组。"}

    if not steps:
        return {"status": "fail", "msg": "执行器参数无效：steps 为空。"}

    timing = _resolve_timing_profile(timing_profile)

    slot_details = extract_blocks_with_screen_points(tuple(int(v) for v in capture_bbox))
    if len(slot_details) != 4:
        return {"status": "fail", "msg": f"执行器识别异常：检测到 {len(slot_details)} 个槽位，期望 4 个。"}

    input_backend = None
    backend_name = "dry-run"
    if not dry_run:
        input_backend = _load_mouse_backend(backend)
        backend_name = input_backend["name"]

    actions: List[Dict[str, Any]] = []
    ordered_steps = sorted(steps, key=lambda s: int(s.get("step_order", 0)))

    for step in ordered_steps:
        block_index = int(step.get("block_index", -1))
        place_row = int(step.get("place_row", 0))
        place_col = int(step.get("place_col", 0))

        if block_index < 0 or block_index >= len(slot_details):
            return {"status": "fail", "msg": f"执行器索引越界：block_index={block_index}"}

        slot = slot_details[block_index]
        step_coords = step.get("coords") or slot.get("coords", [])
        anchor_row, anchor_col = choose_piece_anchor(step_coords)
        src = _find_source_point(slot, (anchor_row, anchor_col))

        target_row = place_row + anchor_row
        target_col = place_col + anchor_col
        if not (0 <= target_row < 8 and 0 <= target_col < 8):
            return {
                "status": "fail",
                "msg": f"执行器目标越界：row={target_row}, col={target_col}",
            }

        dst = _board_cell_center(board_bbox, target_row, target_col)

        if not dry_run and input_backend is not None:
            _drag_mouse(input_backend, src, dst, timing)
            time.sleep(float(timing["inter_step_sec"]))

        actions.append(
            {
                "step_order": int(step.get("step_order", len(actions) + 1)),
                "block_index": block_index,
                "anchor": [anchor_row, anchor_col],
                "source": [int(src[0]), int(src[1])],
                "target": [int(dst[0]), int(dst[1])],
                "target_cell": [target_row, target_col],
            }
        )

    return {
        "status": "success",
        "backend": backend_name,
        "dry_run": bool(dry_run),
        "timing_profile": str(timing_profile or "safe").lower(),
        "actions": actions,
    }
