import ctypes
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from vision_core import extract_blocks_with_screen_points


TIMING_PROFILES: Dict[str, Dict[str, float]] = {
    # 优先稳定性，避免游戏漏收鼠标拖放事件。
    # - pre_press_sec: 鼠标移动到起点后等待的时间
    # - hold_before_drag_sec: 按下鼠标后等待的时间
    # - travel_sec: 从起点拖动到终点的时间
    # - hold_after_arrive_sec: 到达终点后继续按住的时间
    # - release_settle_sec: 松开鼠标后等待的时间
    # - inter_step_sec: 步骤间的额外等待时间
    # - clear_extra_sec: 触发消除后额外等待的时间
    # - min_steps: 拖动分多少步，步数越多越慢越平滑，反之越快但可能不稳定
    # - px_per_step: 每步拖动的像素数，数值越小越平滑但越慢，数值越大越快但可能不稳定
    "safe": {
        "pre_press_sec": 0.05,
        "hold_before_drag_sec": 0.08,
        "travel_sec": 0.24,
        "hold_after_arrive_sec": 0.06,
        "release_settle_sec": 0.12,
        "inter_step_sec": 0.28,
        "clear_extra_sec": 1.0,
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
        "clear_extra_sec": 1.0,
        "min_steps": 7.0,
        "px_per_step": 60.0,
    },
    "fast": {
        "pre_press_sec": 0.02,
        "hold_before_drag_sec": 0.05,
        "travel_sec": 0.10,
        "hold_after_arrive_sec": 0.015,
        "release_settle_sec": 0.05,
        "inter_step_sec": 0.05,
        "clear_extra_sec": 1.0,
        "min_steps": 5.0,
        "px_per_step": 80.0,
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
    norm_set = set(norm)

    # 规则优先：匹配用户观察到的人类拖拽焦点。
    if h % 2 == 1 and w % 2 == 1:
        candidate = (h // 2, w // 2)
    elif h % 2 == 0 and w % 2 == 1:
        center_col = w // 2
        top_row = h // 2 - 1
        bottom_row = h // 2
        top_cell = (top_row, center_col)
        bottom_cell = (bottom_row, center_col)

        if top_cell in norm_set and bottom_cell not in norm_set:
            candidate = top_cell
        elif bottom_cell in norm_set and top_cell not in norm_set:
            candidate = bottom_cell
        elif top_cell in norm_set and bottom_cell in norm_set:
            row_mean = float(sum(r for r, _ in norm)) / float(len(norm))
            if row_mean > (top_row + bottom_row) / 2.0:
                candidate = bottom_cell
            elif row_mean < (top_row + bottom_row) / 2.0:
                candidate = top_cell
            else:
                # 重心居中时，保持传统矩形块习惯：选上侧。
                candidate = top_cell
        else:
            candidate = top_cell
    elif h % 2 == 1 and w % 2 == 0:
        center_row = h // 2
        left_col = w // 2 - 1
        right_col = w // 2
        left_cell = (center_row, left_col)
        right_cell = (center_row, right_col)

        if left_cell in norm_set and right_cell not in norm_set:
            candidate = left_cell
        elif right_cell in norm_set and left_cell not in norm_set:
            candidate = right_cell
        elif left_cell in norm_set and right_cell in norm_set:
            # 两侧都可选时，偏向“更靠近重心”的一侧。
            col_mean = float(sum(c for _, c in norm)) / float(len(norm))
            if col_mean > (left_col + right_col) / 2.0:
                candidate = right_cell
            elif col_mean < (left_col + right_col) / 2.0:
                candidate = left_cell
            else:
                # 重心居中时，选左侧以保持矩形块的稳定习惯。
                candidate = left_cell
        else:
            candidate = left_cell
    else:
        top_row = h // 2 - 1
        bottom_row = h // 2
        left_col = w // 2 - 1
        right_col = w // 2
        center_cells = [
            (top_row, left_col),
            (top_row, right_col),
            (bottom_row, left_col),
            (bottom_row, right_col),
        ]
        present_centers = [rc for rc in center_cells if rc in norm_set]

        if len(present_centers) == 4:
            # 规则保留：完整 2x2（或完整中心四格）锚点取左上。
            candidate = (top_row, left_col)
        elif present_centers:
            # 非完整中心块时，按重心选择最近中心格；并在并列时偏向右下。
            row_mean = float(sum(r for r, _ in norm)) / float(len(norm))
            col_mean = float(sum(c for _, c in norm)) / float(len(norm))
            candidate = min(
                present_centers,
                key=lambda rc: (
                    (rc[0] - row_mean) ** 2 + (rc[1] - col_mean) ** 2,
                    -rc[0],
                    -rc[1],
                ),
            )
        else:
            candidate = (top_row, left_col)

    if candidate in norm_set:
        return candidate

    # 奇数 x 奇数但中心缺失时，优先取“贴近中心”的相邻格。
    if h % 2 == 1 and w % 2 == 1:
        center_row = h // 2
        center_col = w // 2
        adjacent_hits = [
            rc for rc in norm
            if abs(rc[0] - center_row) + abs(rc[1] - center_col) == 1
        ]
        if adjacent_hits:
            return min(adjacent_hits, key=lambda rc: (rc[0], rc[1]))

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


def _interruptible_sleep(duration: float, check_stop: Callable[[], bool] = None):
    if not check_stop:
        time.sleep(duration)
        return
    elapsed = 0.0
    while elapsed < duration:
        if check_stop():
            raise KeyboardInterrupt("用户请求立刻停止执行")
        step = min(0.01, duration - elapsed)
        time.sleep(step)
        elapsed += step


def _drag_mouse(backend: Dict[str, Any], src: Tuple[int, int], dst: Tuple[int, int], timing: Dict[str, float], check_stop: Callable[[], bool] = None) -> None:
    sx, sy = src
    dx, dy = dst

    backend["move_to"](sx, sy)
    _interruptible_sleep(float(timing["pre_press_sec"]), check_stop)
    backend["mouse_down"]()
    _interruptible_sleep(float(timing["hold_before_drag_sec"]), check_stop)

    px_span = max(abs(dx - sx), abs(dy - sy))
    min_steps = int(round(float(timing["min_steps"])))
    px_per_step = max(float(timing["px_per_step"]), 1.0)
    steps = max(min_steps, int(px_span / px_per_step))
    step_sleep = max(float(timing["travel_sec"]) / max(steps, 1), 0.006)

    for i in range(1, steps + 1):
        if check_stop and check_stop():
            backend["mouse_up"]()
            raise KeyboardInterrupt("用户请求立刻停止执行")
        nx = int(round(sx + (dx - sx) * i / steps))
        ny = int(round(sy + (dy - sy) * i / steps))
        backend["move_to"](nx, ny)
        _interruptible_sleep(step_sleep, check_stop)

    _interruptible_sleep(float(timing["hold_after_arrive_sec"]), check_stop)
    backend["mouse_up"]()
    _interruptible_sleep(float(timing["release_settle_sec"]), check_stop)


def _drag_mouse_profiled(
    backend: Dict[str, Any],
    src: Tuple[int, int],
    dst: Tuple[int, int],
    timing: Dict[str, float],
    check_stop: Callable[[], bool] = None,
) -> Dict[str, float]:
    started = time.perf_counter()
    sx, sy = src
    dx, dy = dst

    backend["move_to"](sx, sy)

    wait_started = time.perf_counter()
    _interruptible_sleep(float(timing["pre_press_sec"]), check_stop)
    backend["mouse_down"]()
    _interruptible_sleep(float(timing["hold_before_drag_sec"]), check_stop)
    press_wait_ms = (time.perf_counter() - wait_started) * 1000.0

    px_span = max(abs(dx - sx), abs(dy - sy))
    min_steps = int(round(float(timing["min_steps"])))
    px_per_step = max(float(timing["px_per_step"]), 1.0)
    steps = max(min_steps, int(px_span / px_per_step))
    step_sleep = max(float(timing["travel_sec"]) / max(steps, 1), 0.006)

    move_started = time.perf_counter()
    for i in range(1, steps + 1):
        if check_stop and check_stop():
            backend["mouse_up"]()
            raise KeyboardInterrupt("用户请求立刻停止执行")
        nx = int(round(sx + (dx - sx) * i / steps))
        ny = int(round(sy + (dy - sy) * i / steps))
        backend["move_to"](nx, ny)
        _interruptible_sleep(step_sleep, check_stop)
    move_ms = (time.perf_counter() - move_started) * 1000.0

    release_started = time.perf_counter()
    _interruptible_sleep(float(timing["hold_after_arrive_sec"]), check_stop)
    backend["mouse_up"]()
    _interruptible_sleep(float(timing["release_settle_sec"]), check_stop)
    release_wait_ms = (time.perf_counter() - release_started) * 1000.0

    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "drag_total_ms": total_ms,
        "press_wait_ms": press_wait_ms,
        "move_ms": move_ms,
        "release_wait_ms": release_wait_ms,
        "move_steps": float(steps),
    }


def execute_solution_steps(
    steps: List[Dict[str, Any]],
    capture_bbox: Sequence[int],
    board_bbox: Sequence[int],
    backend: str = "auto",
    dry_run: bool = False,
    timing_profile: str = "safe",
    enable_profiling: bool = False,
    check_stop: Callable[[], bool] = None,
) -> Dict[str, Any]:
    total_started = time.perf_counter()
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

    recognize_started = time.perf_counter()
    slot_details = extract_blocks_with_screen_points(tuple(int(v) for v in capture_bbox))
    recognize_ms = (time.perf_counter() - recognize_started) * 1000.0
    if len(slot_details) < 2 or len(slot_details) > 4:
        return {"status": "fail", "msg": f"执行器识别异常：检测到 {len(slot_details)} 个槽位，期望 2~4 个。"}

    required_indexes = []
    for step in steps:
        try:
            required_indexes.append(int(step.get("block_index", -1)))
        except Exception:
            required_indexes.append(-1)
    max_required_index = max(required_indexes) if required_indexes else -1
    if max_required_index >= len(slot_details):
        return {
            "status": "fail",
            "msg": (
                f"执行器识别异常：步骤引用最大 block_index={max_required_index}，"
                f"但当前仅识别到 {len(slot_details)} 个槽位。"
            ),
        }

    input_backend = None
    backend_name = "dry-run"
    backend_load_ms = 0.0
    if not dry_run:
        backend_started = time.perf_counter()
        input_backend = _load_mouse_backend(backend)
        backend_load_ms = (time.perf_counter() - backend_started) * 1000.0
        backend_name = input_backend["name"]

    actions: List[Dict[str, Any]] = []
    profile_steps: List[Dict[str, Any]] = []
    ordered_steps = sorted(steps, key=lambda s: int(s.get("step_order", 0)))

    for step in ordered_steps:
        step_started = time.perf_counter()
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

        drag_profile = {
            "drag_total_ms": 0.0,
            "press_wait_ms": 0.0,
            "move_ms": 0.0,
            "release_wait_ms": 0.0,
            "move_steps": 0.0,
        }
        inter_step_wait_ms = 0.0
        clear_wait_ms = 0.0
        if not dry_run and input_backend is not None:
            if enable_profiling:
                drag_profile = _drag_mouse_profiled(input_backend, src, dst, timing, check_stop)
            else:
                _drag_mouse(input_backend, src, dst, timing, check_stop)

            inter_wait_started = time.perf_counter()
            time.sleep(float(timing["inter_step_sec"]))
            inter_step_wait_ms = (time.perf_counter() - inter_wait_started) * 1000.0

            score_gained = int(step.get("score_gained", 0) or 0)
            if score_gained > 0:
                # 触发消除后棋盘会短暂锁定，额外等待以避免下一步被吞。
                clear_wait_started = time.perf_counter()
                time.sleep(float(timing["clear_extra_sec"]))
                clear_wait_ms = (time.perf_counter() - clear_wait_started) * 1000.0
        else:
            score_gained = int(step.get("score_gained", 0) or 0)

        actions.append(
            {
                "step_order": int(step.get("step_order", len(actions) + 1)),
                "block_index": block_index,
                "anchor": [anchor_row, anchor_col],
                "source": [int(src[0]), int(src[1])],
                "target": [int(dst[0]), int(dst[1])],
                "target_cell": [target_row, target_col],
                "score_gained": int(step.get("score_gained", 0) or 0),
            }
        )

        if enable_profiling:
            profile_steps.append(
                {
                    "step_order": int(step.get("step_order", len(actions))),
                    "block_index": block_index,
                    "drag_total_ms": round(float(drag_profile["drag_total_ms"]), 3),
                    "press_wait_ms": round(float(drag_profile["press_wait_ms"]), 3),
                    "move_ms": round(float(drag_profile["move_ms"]), 3),
                    "release_wait_ms": round(float(drag_profile["release_wait_ms"]), 3),
                    "inter_step_wait_ms": round(float(inter_step_wait_ms), 3),
                    "clear_wait_ms": round(float(clear_wait_ms), 3),
                    "move_steps": int(round(float(drag_profile["move_steps"]))),
                    "step_total_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                }
            )

    result = {
        "status": "success",
        "backend": backend_name,
        "dry_run": bool(dry_run),
        "timing_profile": str(timing_profile or "safe").lower(),
        "actions": actions,
    }

    if enable_profiling:
        result["perf"] = {
            "enabled": True,
            "recognize_slots_ms": round(recognize_ms, 3),
            "load_backend_ms": round(backend_load_ms, 3),
            "steps": profile_steps,
            "drag_total_ms": round(sum(item["drag_total_ms"] for item in profile_steps), 3),
            "inter_step_wait_total_ms": round(sum(item["inter_step_wait_ms"] for item in profile_steps), 3),
            "clear_wait_total_ms": round(sum(item["clear_wait_ms"] for item in profile_steps), 3),
            "execute_total_ms": round((time.perf_counter() - total_started) * 1000.0, 3),
        }

    return result
