import numpy as np
from numba import njit
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import itertools
import threading
import time
import ctypes
import json
from pathlib import Path
from typing import Optional
import uvicorn
from vision_core import (
    get_screen_bbox,
    extract_blocks_from_memory,
    extract_board_from_memory,
    set_tmp_debug_enabled,
    get_tmp_debug_enabled,
)
from executor import execute_solution_steps, is_running_as_admin

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 核心算法 (Numba JIT 保持极致性能) ---
PERMS_2 = np.array(list(itertools.permutations([0, 1])), dtype=np.int8)
PERMS_3 = np.array(list(itertools.permutations([0, 1, 2])), dtype=np.int8)
PERMS_4 = np.array(list(itertools.permutations([0, 1, 2, 3])), dtype=np.int8)
NO_SOLUTION_EVAL = -99999999.0
STOP_HOTKEY_VK = {
    "F8": 0x77,
    "F9": 0x78,
    "F10": 0x79,
}
SETTINGS_FILE = Path(__file__).resolve().parent / "app_settings.json"
DATA_DIR = Path(__file__).resolve().parent / "dataset"
RECORD_FILE = DATA_DIR / "turn_history.jsonl"
PERF_LOG_FILE = DATA_DIR / "perf_log.jsonl"
DATA_LOCK = threading.Lock()
DATA_DIR.mkdir(exist_ok=True)

@njit(fastmath=True, nogil=True, cache=True)
def _check_fit(board, piece, r, c):
    for i in range(piece.shape[0]):
        pr, pc = piece[i]
        if pr == -1: break
        nr, nc = r + pr, c + pc
        if nr < 0 or nr >= 8 or nc < 0 or nc >= 8: return False
        if board[nr * 8 + nc] > 0: return False
    return True


@njit(fastmath=True, nogil=True, cache=True)
def _is_anchored(board, piece, r, c):
    for i in range(piece.shape[0]):
        pr, pc = piece[i]
        if pr == -1:
            break

        nr, nc = r + pr, c + pc
        if nr == 0 or nr == 7 or nc == 0 or nc == 7:
            return True

        if nr > 0 and board[(nr - 1) * 8 + nc] > 0:
            return True
        if nr < 7 and board[(nr + 1) * 8 + nc] > 0:
            return True
        if nc > 0 and board[nr * 8 + nc - 1] > 0:
            return True
        if nc < 7 and board[nr * 8 + nc + 1] > 0:
            return True

    return False

@njit(fastmath=True, nogil=True, cache=True)
def _get_clear_score(lines):
    if lines == 0: return 0
    if lines == 1: return 200
    if lines == 2: return 500
    if lines == 3: return 1000
    if lines == 4: return 1500
    if lines >= 5: return 2000
    return 0

@njit(fastmath=True, nogil=True)
def _place_and_clear(board, piece, r, c):
    new_board = board.copy()
    for i in range(piece.shape[0]):
        pr, pc = piece[i]
        if pr == -1: break
        new_board[(r + pr) * 8 + (c + pc)] = 1
        
    rows_to_clear = np.zeros(8, dtype=np.bool_)
    cols_to_clear = np.zeros(8, dtype=np.bool_)
    
    for row in range(8):
        is_full = True
        for col in range(8):
            if new_board[row * 8 + col] == 0:
                is_full = False; break
        if is_full: rows_to_clear[row] = True

    for col in range(8):
        is_full = True
        for row in range(8):
            if new_board[row * 8 + col] == 0:
                is_full = False; break
        if is_full: cols_to_clear[col] = True

    cleared_lines = 0
    for row in range(8):
        if rows_to_clear[row]:
            cleared_lines += 1
    for col in range(8):
        if cols_to_clear[col]:
            cleared_lines += 1

    # 清行规则：
    # - 1(普通块) 被清除 -> 0
    # - 2(不可消除块) 保持 2
    # - 3(强化块) 第一次被清除 -> 1，第二次再清除才会消失
    for row in range(8):
        for col in range(8):
            if not rows_to_clear[row] and not cols_to_clear[col]:
                continue
            idx = row * 8 + col
            cell = new_board[idx]
            if cell == 1:
                new_board[idx] = 0
            elif cell == 3:
                new_board[idx] = 1
            
    return new_board, _get_clear_score(cleared_lines)

@njit(fastmath=True, nogil=True)
def _evaluate_board(board):
    empty_spaces = 0
    holes = 0
    transitions = 0
    almost_full_lines = 0
    empty_3x3_count = 0

    for r in range(8):
        row_sum = 0
        for c in range(8):
            idx = r * 8 + c
            cell = board[idx]

            cell_occ = 1 if cell > 0 else 0
            if cell == 0:
                empty_spaces += 1
                is_hole = True
                if r > 0 and board[idx - 8] == 0: is_hole = False
                elif r < 7 and board[idx + 8] == 0: is_hole = False
                elif c > 0 and board[idx - 1] == 0: is_hole = False
                elif c < 7 and board[idx + 1] == 0: is_hole = False
                if is_hole:
                    holes += 1
            else:
                row_sum += 1

            if c < 7:
                right_occ = 1 if board[idx + 1] > 0 else 0
                if cell_occ != right_occ:
                    transitions += 1
            if r < 7:
                down_occ = 1 if board[idx + 8] > 0 else 0
                if cell_occ != down_occ:
                    transitions += 1

        if row_sum == 6 or row_sum == 7:
            almost_full_lines += 1

    for c in range(8):
        col_sum = 0
        for r in range(8):
            if board[r * 8 + c] > 0:
                col_sum += 1
        if col_sum == 6 or col_sum == 7:
            almost_full_lines += 1

    for r in range(6):
        for c in range(6):
            is_3x3_empty = True
            for i in range(3):
                for j in range(3):
                    if board[(r + i) * 8 + (c + j)] > 0:
                        is_3x3_empty = False
                        break
                if not is_3x3_empty:
                    break
            if is_3x3_empty:
                empty_3x3_count += 1

    score = 0.0
    score -= holes * 800.0
    score -= transitions * 20.0
    score += almost_full_lines * 150.0
    score -= empty_3x3_count * 1500.0

    if empty_spaces >= 20:
        score -= empty_spaces * 35.0

    if empty_spaces < 20:
        diff = 20 - empty_spaces
        score -= (diff * diff) * 200.0

    return score

@njit(fastmath=True, nogil=True)
def _count_empty_spaces(board):
    empty_spaces = 0
    for i in range(64):
        if board[i] == 0:
            empty_spaces += 1
    return empty_spaces


@njit(fastmath=True, nogil=True)
def _compose_final_eval(board_after, score1, score2, score3, score4):
    base_eval = _evaluate_board(board_after)
    total_game_score = score1 + score2 + score3 + score4

    single_clears = 0
    double_clears = 0
    big_burst_score = 0

    if score1 == 200:
        single_clears += 1
    if score1 == 500:
        double_clears += 1
    if score1 >= 1000:
        big_burst_score += score1

    if score2 == 200:
        single_clears += 1
    if score2 == 500:
        double_clears += 1
    if score2 >= 1000:
        big_burst_score += score2

    if score3 == 200:
        single_clears += 1
    if score3 == 500:
        double_clears += 1
    if score3 >= 1000:
        big_burst_score += score3

    if score4 == 200:
        single_clears += 1
    if score4 == 500:
        double_clears += 1
    if score4 >= 1000:
        big_burst_score += score4

    empty_spaces = 0
    for i in range(64):
        if board_after[i] == 0:
            empty_spaces += 1

    tactical_score = 0.0
    if empty_spaces < 20:
        tactical_score += total_game_score * 6.0
        tactical_score += double_clears * 150.0
        tactical_score += big_burst_score * 3.0
        tactical_score -= single_clears * 50.0
    elif empty_spaces < 28:
        tactical_score -= single_clears * 700.0
        tactical_score += double_clears * 120.0
        tactical_score += big_burst_score * 5.0
        tactical_score += total_game_score * 0.6
    else:
        tactical_score -= single_clears * 1200.0
        tactical_score -= double_clears * 900.0
        tactical_score += big_burst_score * 8.0
        tactical_score += (40.0 - empty_spaces) * 35.0

    return base_eval + tactical_score, total_game_score


@njit(fastmath=True, nogil=True)
def _solve_turn_core_2(board, pieces, use_anchor_prune):
    best_eval = NO_SOLUTION_EVAL
    best_moves = np.full((4, 3), -1, dtype=np.int32)
    best_game_score = 0

    for perm_idx in range(2):
        order = PERMS_2[perm_idx]
        p1 = pieces[order[0]]
        for r1 in range(8):
            for c1 in range(8):
                if not _check_fit(board, p1, r1, c1):
                    continue
                if use_anchor_prune and not _is_anchored(board, p1, r1, c1):
                    continue
                b1, score1 = _place_and_clear(board, p1, r1, c1)

                p2 = pieces[order[1]]
                for r2 in range(8):
                    for c2 in range(8):
                        if not _check_fit(b1, p2, r2, c2):
                            continue
                        if use_anchor_prune and not _is_anchored(b1, p2, r2, c2):
                            continue
                        b2, score2 = _place_and_clear(b1, p2, r2, c2)

                        final_eval, total_game_score = _compose_final_eval(b2, score1, score2, 0, 0)
                        if final_eval > best_eval:
                            best_eval = final_eval
                            best_game_score = total_game_score
                            best_moves[0] = [order[0], r1, c1]
                            best_moves[1] = [order[1], r2, c2]

    return best_eval, best_game_score, best_moves


@njit(fastmath=True, nogil=True)
def _solve_turn_core_3(board, pieces, use_anchor_prune):
    best_eval = NO_SOLUTION_EVAL
    best_moves = np.full((4, 3), -1, dtype=np.int32)
    best_game_score = 0

    for perm_idx in range(6):
        order = PERMS_3[perm_idx]
        p1 = pieces[order[0]]
        for r1 in range(8):
            for c1 in range(8):
                if not _check_fit(board, p1, r1, c1):
                    continue
                if use_anchor_prune and not _is_anchored(board, p1, r1, c1):
                    continue
                b1, score1 = _place_and_clear(board, p1, r1, c1)

                p2 = pieces[order[1]]
                for r2 in range(8):
                    for c2 in range(8):
                        if not _check_fit(b1, p2, r2, c2):
                            continue
                        if use_anchor_prune and not _is_anchored(b1, p2, r2, c2):
                            continue
                        b2, score2 = _place_and_clear(b1, p2, r2, c2)

                        p3 = pieces[order[2]]
                        for r3 in range(8):
                            for c3 in range(8):
                                if not _check_fit(b2, p3, r3, c3):
                                    continue
                                if use_anchor_prune and not _is_anchored(b2, p3, r3, c3):
                                    continue
                                b3, score3 = _place_and_clear(b2, p3, r3, c3)

                                final_eval, total_game_score = _compose_final_eval(b3, score1, score2, score3, 0)
                                if final_eval > best_eval:
                                    best_eval = final_eval
                                    best_game_score = total_game_score
                                    best_moves[0] = [order[0], r1, c1]
                                    best_moves[1] = [order[1], r2, c2]
                                    best_moves[2] = [order[2], r3, c3]

    return best_eval, best_game_score, best_moves


@njit(fastmath=True, nogil=True)
def _solve_turn_core_4(board, pieces, use_anchor_prune):
    best_eval = NO_SOLUTION_EVAL
    best_moves = np.full((4, 3), -1, dtype=np.int32)
    best_game_score = 0

    for perm_idx in range(24):
        order = PERMS_4[perm_idx]
        p1 = pieces[order[0]]
        for r1 in range(8):
            for c1 in range(8):
                if not _check_fit(board, p1, r1, c1):
                    continue
                if use_anchor_prune and not _is_anchored(board, p1, r1, c1):
                    continue
                b1, score1 = _place_and_clear(board, p1, r1, c1)

                p2 = pieces[order[1]]
                for r2 in range(8):
                    for c2 in range(8):
                        if not _check_fit(b1, p2, r2, c2):
                            continue
                        if use_anchor_prune and not _is_anchored(b1, p2, r2, c2):
                            continue
                        b2, score2 = _place_and_clear(b1, p2, r2, c2)

                        p3 = pieces[order[2]]
                        for r3 in range(8):
                            for c3 in range(8):
                                if not _check_fit(b2, p3, r3, c3):
                                    continue
                                if use_anchor_prune and not _is_anchored(b2, p3, r3, c3):
                                    continue
                                b3, score3 = _place_and_clear(b2, p3, r3, c3)

                                p4 = pieces[order[3]]
                                for r4 in range(8):
                                    for c4 in range(8):
                                        if not _check_fit(b3, p4, r4, c4):
                                            continue
                                        if use_anchor_prune and not _is_anchored(b3, p4, r4, c4):
                                            continue
                                        b4, score4 = _place_and_clear(b3, p4, r4, c4)

                                        final_eval, total_game_score = _compose_final_eval(b4, score1, score2, score3, score4)
                                        if final_eval > best_eval:
                                            best_eval = final_eval
                                            best_game_score = total_game_score
                                            best_moves[0] = [order[0], r1, c1]
                                            best_moves[1] = [order[1], r2, c2]
                                            best_moves[2] = [order[2], r3, c3]
                                            best_moves[3] = [order[3], r4, c4]

    return best_eval, best_game_score, best_moves


@njit(fastmath=True, nogil=True)
def _solve_turn(board, pieces, piece_count):
    # 空盘或近空盘时优先启用锚点剪枝；若剪枝后无解则回退全搜索，避免误杀最优解。
    empty_spaces = _count_empty_spaces(board)

    if empty_spaces >= 40:
        if piece_count == 2:
            best_eval, best_game_score, best_moves = _solve_turn_core_2(board, pieces, True)
        elif piece_count == 3:
            best_eval, best_game_score, best_moves = _solve_turn_core_3(board, pieces, True)
        else:
            best_eval, best_game_score, best_moves = _solve_turn_core_4(board, pieces, True)
        if best_eval != NO_SOLUTION_EVAL:
            return best_eval, best_game_score, best_moves

    if piece_count == 2:
        return _solve_turn_core_2(board, pieces, False)
    if piece_count == 3:
        return _solve_turn_core_3(board, pieces, False)
    return _solve_turn_core_4(board, pieces, False)

# --- API 协议 ---
class SolveRequest(BaseModel):
    board: List[List[int]]
    blocks: List[List[List[int]]]


class ExecuteStep(BaseModel):
    step_order: int
    block_index: int
    place_row: int
    place_col: int
    coords: List[List[int]]


class ExecuteRequest(BaseModel):
    steps: List[ExecuteStep]
    backend: str = "auto"
    dry_run: bool = False
    timing_profile: str = "safe"


class AgentStartRequest(BaseModel):
    wait_ms: int = 900
    countdown_sec: int = 3
    backend: str = "auto"
    dry_run: bool = False
    timing_profile: str = "safe"
    stop_hotkey: str = "F8"


class SettingsUpdateRequest(BaseModel):
    tmp_debug_enabled: Optional[bool] = None
    perf_analysis_enabled: Optional[bool] = None
    timing_profile: Optional[str] = None
    record_data_enabled: Optional[bool] = None
    candidate_count: Optional[int] = None
    lab_features_enabled: Optional[bool] = None


AGENT_LOCK = threading.Lock()
AGENT_STATE: Dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "stop_requested": False,
    "countdown_left": 0,
    "cycle_count": 0,
    "last_msg": "",
    "last_error": "",
    "wait_ms": 900,
    "backend": "auto",
    "dry_run": False,
    "timing_profile": "safe",
    "stop_hotkey": "F8",
    "last_profile": None,
    "last_board": None,
    "thread": None,
}

# 全局存储框选坐标（支持持久化）
BBOX_4 = None
BBOX_3 = None
BBOX_2 = None
CANDIDATE_COUNT = 3
LAB_FEATURES_ENABLED = False
BOARD_BBOX = None
TIMING_PROFILE_CHOICE = "safe"
PERF_ANALYSIS_ENABLED = False
RECORD_DATA_ENABLED = False

def get_current_bbox():
    if CANDIDATE_COUNT == 4: return BBOX_4
    if CANDIDATE_COUNT == 3: return BBOX_3
    if CANDIDATE_COUNT == 2: return BBOX_2
    return None


def _normalize_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except Exception:
        return None

    if x2 - x1 <= 10 or y2 - y1 <= 10:
        return None
    return (x1, y1, x2, y2)


def _save_settings():
    payload = {
        "tmp_debug_enabled": bool(get_tmp_debug_enabled()),
        "timing_profile": str(TIMING_PROFILE_CHOICE),
        "perf_analysis_enabled": bool(PERF_ANALYSIS_ENABLED),
        "record_data_enabled": bool(RECORD_DATA_ENABLED),
        "candidate_count": int(CANDIDATE_COUNT),
        "lab_features_enabled": bool(LAB_FEATURES_ENABLED),
        "bbox_4": list(BBOX_4) if BBOX_4 else None,
        "bbox_3": list(BBOX_3) if BBOX_3 else None,
        "bbox_2": list(BBOX_2) if BBOX_2 else None,
        "board_bbox": list(BOARD_BBOX) if BOARD_BBOX else None,
    }
    SETTINGS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_settings():
    global PERF_ANALYSIS_ENABLED, TIMING_PROFILE_CHOICE, BBOX_4, BBOX_3, BBOX_2, CANDIDATE_COUNT, LAB_FEATURES_ENABLED, BOARD_BBOX, RECORD_DATA_ENABLED

    if not SETTINGS_FILE.exists():
        set_tmp_debug_enabled(False)
        return

    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        set_tmp_debug_enabled(False)
        return

    set_tmp_debug_enabled(bool(data.get("tmp_debug_enabled", False)))
    PERF_ANALYSIS_ENABLED = bool(data.get("perf_analysis_enabled", False))
    RECORD_DATA_ENABLED = bool(data.get("record_data_enabled", False))
    choice = str(data.get("timing_profile", "safe")).lower()
    if choice in ("safe", "balanced", "fast"):
        TIMING_PROFILE_CHOICE = choice

    CANDIDATE_COUNT = int(data.get("candidate_count", 3))
    LAB_FEATURES_ENABLED = bool(data.get("lab_features_enabled", False))
    BBOX_4 = _normalize_bbox(data.get("bbox_4"))
    
    old_bbox = _normalize_bbox(data.get("current_bbox"))
    if "bbox_3" in data:
        BBOX_3 = _normalize_bbox(data.get("bbox_3"))
    else:
        BBOX_3 = old_bbox

    BBOX_2 = _normalize_bbox(data.get("bbox_2"))
    BOARD_BBOX = _normalize_bbox(data.get("board_bbox"))


_load_settings()


def _record_turn_data(board_2d, blocks_3d, source: str, solve_result: Optional[Dict[str, Any]] = None):
    if not RECORD_DATA_ENABLED:
        return
    record = {
        "timestamp": time.time(),
        "source": source,
        "board": board_2d,
        "pieces": blocks_3d,
    }
    if solve_result is not None:
        record["solve_status"] = solve_result.get("status")
        record["evaluation"] = solve_result.get("evaluation")
        record["game_score_gained"] = solve_result.get("game_score_gained")
        record["steps"] = solve_result.get("steps")
    try:
        with DATA_LOCK:
            with RECORD_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 记录失败不影响主流程，避免线上执行中断
        pass


def _round_ms(value: float) -> float:
    return round(float(value), 3)


def _should_profile(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return bool(PERF_ANALYSIS_ENABLED)


def _record_perf_log(event_type: str, payload: Dict[str, Any], enabled: Optional[bool] = None):
    if not _should_profile(enabled):
        return

    record = {
        "timestamp": time.time(),
        "event": event_type,
        **payload,
    }
    try:
        with DATA_LOCK:
            with PERF_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _solve_with_steps(
    board: List[List[int]],
    blocks: List[List[List[int]]],
    source: str = "unknown",
    enable_profiling: Optional[bool] = None,
) -> Dict[str, Any]:
    profiling_enabled = _should_profile(enable_profiling)
    total_started = time.perf_counter()

    piece_count = int(len(blocks))
    if piece_count < 2 or piece_count > 4:
        result = {"status": "fail", "msg": "参数校验失败：方块数量必须在 2~4 之间"}
        if profiling_enabled:
            result["perf"] = {"enabled": True, "solve_total_ms": _round_ms((time.perf_counter() - total_started) * 1000.0)}
        _record_turn_data(board, blocks, source, result)
        _record_perf_log("solve", {"source": source, "status": result["status"], "perf": result.get("perf")}, profiling_enabled)
        return result

    board_convert_started = time.perf_counter()
    board_1d = np.array(board, dtype=np.int8).flatten()
    board_convert_ms = (time.perf_counter() - board_convert_started) * 1000.0
    if board_1d.size != 64:
        result = {"status": "fail", "msg": "参数校验失败：棋盘尺寸必须为 8x8"}
        if profiling_enabled:
            result["perf"] = {
                "enabled": True,
                "board_convert_ms": _round_ms(board_convert_ms),
                "solve_total_ms": _round_ms((time.perf_counter() - total_started) * 1000.0),
            }
        _record_turn_data(board, blocks, source, result)
        _record_perf_log("solve", {"source": source, "status": result["status"], "perf": result.get("perf")}, profiling_enabled)
        return result

    pieces_pack_started = time.perf_counter()
    pieces_arr = np.full((4, 16, 2), -1, dtype=np.int8)
    for i, block_coords in enumerate(blocks):
        for j, (r, c) in enumerate(block_coords):
            if j < 16:
                pieces_arr[i, j, 0] = r
                pieces_arr[i, j, 1] = c
    pieces_pack_ms = (time.perf_counter() - pieces_pack_started) * 1000.0

    solve_started = time.perf_counter()
    eval_score, game_score, moves = _solve_turn(board_1d, pieces_arr, piece_count)
    solve_core_ms = (time.perf_counter() - solve_started) * 1000.0
    if eval_score == NO_SOLUTION_EVAL:
        result = {"status": "fail", "msg": f"Game Over：当前盘面下这 {piece_count} 个方块无法全部放置。"}
        if profiling_enabled:
            result["perf"] = {
                "enabled": True,
                "board_convert_ms": _round_ms(board_convert_ms),
                "pieces_pack_ms": _round_ms(pieces_pack_ms),
                "solve_core_ms": _round_ms(solve_core_ms),
                "solve_total_ms": _round_ms((time.perf_counter() - total_started) * 1000.0),
            }
        _record_turn_data(board, blocks, source, result)
        _record_perf_log("solve", {"source": source, "status": result["status"], "perf": result.get("perf")}, profiling_enabled)
        return result

    build_steps_started = time.perf_counter()
    steps = []
    current_sim_board = board_1d.copy()
    for i in range(piece_count):
        move = moves[i]
        b_idx, r, c = int(move[0]), int(move[1]), int(move[2])
        if b_idx < 0 or b_idx >= piece_count:
            continue
        piece_coords = blocks[b_idx]

        board_before = current_sim_board.tolist()
        next_board, step_score = _place_and_clear(current_sim_board, pieces_arr[b_idx], r, c)

        steps.append({
            "step_order": len(steps) + 1,
            "block_index": b_idx,
            "place_row": r,
            "place_col": c,
            "coords": piece_coords,
            "score_gained": int(step_score),
            "board_before": board_before,
            "board_after": next_board.tolist(),
        })
        current_sim_board = next_board
    build_steps_ms = (time.perf_counter() - build_steps_started) * 1000.0

    result = {
        "status": "success",
        "game_score_gained": int(game_score),
        "evaluation": float(eval_score),
        "steps": steps,
    }
    if profiling_enabled:
        result["perf"] = {
            "enabled": True,
            "board_convert_ms": _round_ms(board_convert_ms),
            "pieces_pack_ms": _round_ms(pieces_pack_ms),
            "solve_core_ms": _round_ms(solve_core_ms),
            "build_steps_ms": _round_ms(build_steps_ms),
            "solve_total_ms": _round_ms((time.perf_counter() - total_started) * 1000.0),
        }
    _record_turn_data(board, blocks, source, result)
    _record_perf_log(
        "solve",
        {
            "source": source,
            "status": result["status"],
            "evaluation": result.get("evaluation"),
            "game_score_gained": result.get("game_score_gained"),
            "perf": result.get("perf"),
        },
        profiling_enabled,
    )
    return result


def _resolve_stop_hotkey(hotkey_name: str):
    key = str(hotkey_name or "F8").upper()
    if key not in STOP_HOTKEY_VK:
        raise ValueError(f"不支持的停止热键: {hotkey_name}，可选: {', '.join(STOP_HOTKEY_VK.keys())}")
    return key, STOP_HOTKEY_VK[key]


def _is_vk_down(vk_code: int) -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False


def _set_agent_state(**kwargs):
    with AGENT_LOCK:
        AGENT_STATE.update(kwargs)


def _agent_snapshot() -> Dict[str, Any]:
    with AGENT_LOCK:
        return {
            "running": bool(AGENT_STATE["running"]),
            "phase": AGENT_STATE["phase"],
            "countdown_left": int(AGENT_STATE["countdown_left"]),
            "cycle_count": int(AGENT_STATE["cycle_count"]),
            "last_msg": AGENT_STATE["last_msg"],
            "last_error": AGENT_STATE["last_error"],
            "wait_ms": int(AGENT_STATE["wait_ms"]),
            "backend": AGENT_STATE["backend"],
            "dry_run": bool(AGENT_STATE["dry_run"]),
            "timing_profile": AGENT_STATE["timing_profile"],
            "stop_hotkey": AGENT_STATE["stop_hotkey"],
            "last_profile": AGENT_STATE["last_profile"],
            "last_board": AGENT_STATE.get("last_board"),
        }


def _agent_worker(
    wait_ms: int,
    countdown_sec: int,
    backend: str,
    dry_run: bool,
    timing_profile: str,
    stop_hotkey: str,
    stop_vk: int,
):
    prev_hotkey_down = False
    try:
        for sec in range(countdown_sec, 0, -1):
            _set_agent_state(phase="countdown", countdown_left=sec, last_msg=f"代理将在 {sec} 秒后开始。")

            hotkey_down = _is_vk_down(stop_vk)
            if hotkey_down and not prev_hotkey_down:
                _set_agent_state(stop_requested=True, last_msg=f"检测到停止热键 {stop_hotkey}，正在停止。")
            prev_hotkey_down = hotkey_down

            with AGENT_LOCK:
                if AGENT_STATE["stop_requested"]:
                    break
            time.sleep(1.0)

        with AGENT_LOCK:
            if AGENT_STATE["stop_requested"]:
                return

        _set_agent_state(phase="running", countdown_left=0, last_msg="代理已启动，循环执行中。")

        while True:
            with AGENT_LOCK:
                if AGENT_STATE["stop_requested"]:
                    break

            hotkey_down = _is_vk_down(stop_vk)
            if hotkey_down and not prev_hotkey_down:
                _set_agent_state(stop_requested=True, last_msg=f"检测到停止热键 {stop_hotkey}，正在停止。")
            prev_hotkey_down = hotkey_down

            with AGENT_LOCK:
                if AGENT_STATE["stop_requested"]:
                    break

            try:
                cycle_started = time.perf_counter()
                current_bbox = get_current_bbox()
                if not current_bbox or not BOARD_BBOX:
                    raise RuntimeError("代理执行失败：缺少预备区或棋盘框选。")

                board_started = time.perf_counter()
                board = extract_board_from_memory(BOARD_BBOX)
                _set_agent_state(last_board=board)
                board_ms = (time.perf_counter() - board_started) * 1000.0

                blocks_started = time.perf_counter()
                blocks = extract_blocks_from_memory(current_bbox)
                blocks_ms = (time.perf_counter() - blocks_started) * 1000.0

                solve_result = _solve_with_steps(board, blocks, source="agent", enable_profiling=PERF_ANALYSIS_ENABLED)
                if solve_result.get("status") != "success":
                    raise RuntimeError(solve_result.get("msg", "推导失败"))

                execute_started = time.perf_counter()
                execute_result = execute_solution_steps(
                    steps=solve_result["steps"],
                    capture_bbox=current_bbox,
                    board_bbox=BOARD_BBOX,
                    backend=backend,
                    dry_run=dry_run,
                    timing_profile=timing_profile,
                    enable_profiling=PERF_ANALYSIS_ENABLED,
                    check_stop=lambda: _is_vk_down(stop_vk) or AGENT_STATE["stop_requested"],
                )
                execute_ms = (time.perf_counter() - execute_started) * 1000.0
                if execute_result.get("status") != "success":
                    raise RuntimeError(execute_result.get("msg", "执行失败"))

                cycle_profile = None
                if PERF_ANALYSIS_ENABLED:
                    cycle_profile = {
                        "board_recognize_ms": _round_ms(board_ms),
                        "block_recognize_ms": _round_ms(blocks_ms),
                        "solve_ms": _round_ms(solve_result.get("perf", {}).get("solve_total_ms", 0.0)),
                        "execute_ms": _round_ms(execute_result.get("perf", {}).get("execute_total_ms", execute_ms)),
                        "cycle_total_ms": _round_ms((time.perf_counter() - cycle_started) * 1000.0),
                    }
                    _record_perf_log(
                        "agent_cycle",
                        {
                            "status": "success",
                            "cycle_count": int(AGENT_STATE["cycle_count"]) + 1,
                            "profile": cycle_profile,
                            "solve_perf": solve_result.get("perf"),
                            "execute_perf": execute_result.get("perf"),
                        },
                        True,
                    )

                with AGENT_LOCK:
                    AGENT_STATE["cycle_count"] += 1
                    AGENT_STATE["last_error"] = ""
                    AGENT_STATE["last_msg"] = f"第 {AGENT_STATE['cycle_count']} 轮完成。"
                    AGENT_STATE["last_profile"] = cycle_profile
            except KeyboardInterrupt as e:
                _set_agent_state(last_error=str(e), last_msg="已成功响应紧急中止指令，正在退出本轮。")
                break
            except Exception as e:
                _record_perf_log(
                    "agent_cycle",
                    {
                        "status": "fail",
                        "error": str(e),
                    },
                    PERF_ANALYSIS_ENABLED,
                )
                _set_agent_state(last_error=str(e), last_msg="本轮失败，将在等待后重试。")

            wait_sec = max(0.05, float(wait_ms) / 1000.0)
            waited = 0.0
            while waited < wait_sec:
                with AGENT_LOCK:
                    if AGENT_STATE["stop_requested"]:
                        break

                hotkey_down = _is_vk_down(stop_vk)
                if hotkey_down and not prev_hotkey_down:
                    _set_agent_state(stop_requested=True, last_msg=f"检测到停止热键 {stop_hotkey}，正在停止。")
                prev_hotkey_down = hotkey_down

                step = min(0.05, wait_sec - waited)
                time.sleep(step)
                waited += step
    finally:
        _set_agent_state(
            running=False,
            phase="stopped",
            stop_requested=False,
            countdown_left=0,
            thread=None,
            last_msg="代理已停止。",
        )

@app.post("/solve")
def solve(req: SolveRequest):
    return _solve_with_steps(req.board, req.blocks, source="api", enable_profiling=PERF_ANALYSIS_ENABLED)

@app.get("/api/set_bbox/{count}")
def api_set_bbox(count: int):
    global BBOX_4, BBOX_3, BBOX_2, CANDIDATE_COUNT
    bbox = get_screen_bbox()
    if bbox and (bbox[2] - bbox[0] > 10 and bbox[3] - bbox[1] > 10):
        if count == 4:
            BBOX_4 = bbox
        elif count == 3:
            BBOX_3 = bbox
        elif count == 2:
            BBOX_2 = bbox
        else:
            return {"status": "fail", "msg": f"不支持的候选区数量: {count}"}
        
        CANDIDATE_COUNT = count
        _save_settings()
        return {"status": "success", "bbox": bbox, "msg": f"{count} 候选区域已锁定: {bbox}"}
    return {"status": "fail", "msg": "框选无效或被取消"}

@app.get("/api/auto_recognize")
def api_auto_recognize():
    current_bbox = get_current_bbox()
    if not current_bbox:
        return {"status": "fail", "msg": "请先点击对应候选数的框选预备区设定识别范围！"}
    
    try:
        started = time.perf_counter()
        blocks = extract_blocks_from_memory(current_bbox)
        if len(blocks) < 2 or len(blocks) > 4:
            result = {
                "status": "fail",
                "msg": (
                    f"识别数量异常：检测到 {len(blocks)} 个槽位，期望值为 2~4。"
                    "请确保框选区域干净且包含所有方块。"
                    "已在项目根目录保存调试截图（cv_capture_1.png ~ cv_capture_3.png，自动轮转覆盖）。"
                ),
            }
            if PERF_ANALYSIS_ENABLED:
                result["perf"] = {
                    "enabled": True,
                    "recognize_blocks_ms": _round_ms((time.perf_counter() - started) * 1000.0),
                }
                _record_perf_log("recognize_blocks", result, True)
            return result

        result = {"status": "success", "blocks": blocks, "slot_count": len(blocks)}
        if PERF_ANALYSIS_ENABLED:
            result["perf"] = {
                "enabled": True,
                "recognize_blocks_ms": _round_ms((time.perf_counter() - started) * 1000.0),
            }
            _record_perf_log("recognize_blocks", result, True)
        return result
    except Exception as e:
        result = {"status": "fail", "msg": f"视觉处理异常: {str(e)}"}
        _record_perf_log("recognize_blocks", result, PERF_ANALYSIS_ENABLED)
        return result

@app.get("/api/set_board_bbox")
def api_set_board_bbox():
    global BOARD_BBOX
    bbox = get_screen_bbox()
    if bbox and (bbox[2] - bbox[0] > 10 and bbox[3] - bbox[1] > 10):
        BOARD_BBOX = bbox
        _save_settings()
        return {"status": "success", "bbox": bbox, "msg": f"棋盘区域已锁定: {bbox}"}
    return {"status": "fail", "msg": "框选无效或被取消"}


@app.get("/api/settings")
def api_get_settings():
    return {
        "status": "success",
        "settings": {
            "tmp_debug_enabled": bool(get_tmp_debug_enabled()),
            "perf_analysis_enabled": bool(PERF_ANALYSIS_ENABLED),
            "record_data_enabled": bool(RECORD_DATA_ENABLED),
            "timing_profile": str(TIMING_PROFILE_CHOICE),
            "candidate_count": int(CANDIDATE_COUNT),
            "lab_features_enabled": bool(LAB_FEATURES_ENABLED),
            "bbox_4": list(BBOX_4) if BBOX_4 else None,
            "bbox_3": list(BBOX_3) if BBOX_3 else None,
            "bbox_2": list(BBOX_2) if BBOX_2 else None,
            "current_bbox": list(get_current_bbox()) if get_current_bbox() else None,
            "board_bbox": list(BOARD_BBOX) if BOARD_BBOX else None,
        },
    }


@app.post("/api/settings")
@app.put("/api/settings")
def api_update_settings(req: SettingsUpdateRequest):
    global PERF_ANALYSIS_ENABLED, TIMING_PROFILE_CHOICE, RECORD_DATA_ENABLED, CANDIDATE_COUNT, LAB_FEATURES_ENABLED
    
    if req.tmp_debug_enabled is not None:
        set_tmp_debug_enabled(bool(req.tmp_debug_enabled))
    if req.perf_analysis_enabled is not None:
        PERF_ANALYSIS_ENABLED = bool(req.perf_analysis_enabled)
    if req.record_data_enabled is not None:
        RECORD_DATA_ENABLED = bool(req.record_data_enabled)
    if req.candidate_count is not None:
        CANDIDATE_COUNT = int(req.candidate_count)
    if req.lab_features_enabled is not None:
        LAB_FEATURES_ENABLED = bool(req.lab_features_enabled)
    if req.timing_profile is not None:
        choice = str(req.timing_profile).lower()
        if choice in ("safe", "balanced", "fast"):
            TIMING_PROFILE_CHOICE = choice

    _save_settings()
    return {
        "status": "success",
        "settings": {
            "tmp_debug_enabled": bool(get_tmp_debug_enabled()),
            "perf_analysis_enabled": bool(PERF_ANALYSIS_ENABLED),
            "record_data_enabled": bool(RECORD_DATA_ENABLED),
            "timing_profile": str(TIMING_PROFILE_CHOICE),
            "candidate_count": int(CANDIDATE_COUNT),
            "lab_features_enabled": bool(LAB_FEATURES_ENABLED),
            "bbox_4": list(BBOX_4) if BBOX_4 else None,
            "bbox_3": list(BBOX_3) if BBOX_3 else None,
            "bbox_2": list(BBOX_2) if BBOX_2 else None,
            "current_bbox": list(get_current_bbox()) if get_current_bbox() else None,
            "board_bbox": list(BOARD_BBOX) if BOARD_BBOX else None,
        },
    }

@app.get("/api/recognize_board")
def api_recognize_board():
    global BOARD_BBOX
    if not BOARD_BBOX:
        return {"status": "fail", "msg": "请先点击「框选棋盘」设定识别范围！"}
    try:
        started = time.perf_counter()
        board = extract_board_from_memory(BOARD_BBOX)
        filled = sum(1 for r in range(8) for c in range(8) if int(board[r][c]) > 0)
        indestructible = sum(1 for r in range(8) for c in range(8) if int(board[r][c]) == 2)
        durable = sum(1 for r in range(8) for c in range(8) if int(board[r][c]) == 3)
        result = {
            "status": "success",
            "board": board,
            "filled_count": filled,
            "indestructible_count": indestructible,
            "durable_count": durable,
        }
        if PERF_ANALYSIS_ENABLED:
            result["perf"] = {
                "enabled": True,
                "recognize_board_ms": _round_ms((time.perf_counter() - started) * 1000.0),
            }
            _record_perf_log("recognize_board", result, True)
        return result
    except Exception as e:
        result = {"status": "fail", "msg": f"棋盘识别异常: {str(e)}"}
        _record_perf_log("recognize_board", result, PERF_ANALYSIS_ENABLED)
        return result


@app.post("/api/execute_solution")
def api_execute_solution(req: ExecuteRequest):
    global BOARD_BBOX
    current_bbox = get_current_bbox()

    if not current_bbox:
        return {"status": "fail", "msg": "请先点击对应候选数的框选预备区设定识别范围！"}
    if not BOARD_BBOX:
        return {"status": "fail", "msg": "请先点击「框选棋盘」设定识别范围！"}
    if not req.steps:
        return {"status": "fail", "msg": "执行参数缺失：steps 为空。"}

    try:
        result = execute_solution_steps(
            steps=[step.dict() for step in req.steps],
            capture_bbox=current_bbox,
            board_bbox=BOARD_BBOX,
            backend=req.backend,
            dry_run=req.dry_run,
            timing_profile=req.timing_profile,
            enable_profiling=PERF_ANALYSIS_ENABLED,
        )
        _record_perf_log(
            "execute_solution",
            {
                "status": result.get("status"),
                "backend": result.get("backend"),
                "timing_profile": result.get("timing_profile"),
                "perf": result.get("perf"),
            },
            PERF_ANALYSIS_ENABLED,
        )
        return result
    except Exception as e:
        return {"status": "fail", "msg": f"执行器异常: {str(e)}"}


@app.post("/api/agent/start")
def api_agent_start(req: AgentStartRequest):
    global BOARD_BBOX
    current_bbox = get_current_bbox()

    if not current_bbox:
        return {"status": "fail", "msg": "请先点击对应候选数的框选预备区设定识别范围！"}
    if not BOARD_BBOX:
        return {"status": "fail", "msg": "请先点击「框选棋盘」设定识别范围！"}

    if not req.dry_run and not is_running_as_admin():
        return {
            "status": "fail",
            "msg": "代理模式启动失败：当前 Python 非管理员权限，无法控制管理员权限游戏窗口。",
        }

    try:
        hotkey_name, hotkey_vk = _resolve_stop_hotkey(req.stop_hotkey)
    except Exception as e:
        return {"status": "fail", "msg": str(e)}

    with AGENT_LOCK:
        if AGENT_STATE["running"]:
            return {"status": "fail", "msg": "代理模式已在运行中。"}

        AGENT_STATE.update({
            "running": True,
            "phase": "countdown",
            "stop_requested": False,
            "countdown_left": max(0, int(req.countdown_sec)),
            "cycle_count": 0,
            "last_msg": "代理准备启动。",
            "last_error": "",
            "wait_ms": max(50, int(req.wait_ms)),
            "backend": req.backend,
            "dry_run": bool(req.dry_run),
            "timing_profile": req.timing_profile,
            "stop_hotkey": hotkey_name,
            "last_profile": None,
            "thread": None,
        })

        worker = threading.Thread(
            target=_agent_worker,
            args=(
                max(50, int(req.wait_ms)),
                max(0, int(req.countdown_sec)),
                req.backend,
                bool(req.dry_run),
                req.timing_profile,
                hotkey_name,
                hotkey_vk,
            ),
            daemon=True,
        )
        AGENT_STATE["thread"] = worker
        worker.start()

    return {
        "status": "success",
        "msg": f"代理模式已启动，{max(0, int(req.countdown_sec))} 秒后开始。停止热键: {hotkey_name}",
        "stop_hotkey": hotkey_name,
    }


@app.post("/api/agent/stop")
def api_agent_stop():
    with AGENT_LOCK:
        if not AGENT_STATE["running"]:
            return {"status": "fail", "msg": "代理模式当前未运行。"}
        AGENT_STATE["stop_requested"] = True
        AGENT_STATE["last_msg"] = "收到停止请求，准备退出代理循环。"

    return {"status": "success", "msg": "停止请求已发送。"}


@app.get("/api/agent/status")
def api_agent_status():
    return {"status": "success", "agent": _agent_snapshot()}

@app.get("/")
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    print(">> 拉海诺测试台后端 V3 启动：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")