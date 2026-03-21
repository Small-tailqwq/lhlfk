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
import uvicorn
from vision_core import get_screen_bbox, extract_blocks_from_memory, extract_board_from_memory
from executor import execute_solution_steps, is_running_as_admin

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 核心算法 (Numba JIT 保持极致性能) ---
PERMS_4 = np.array(list(itertools.permutations([0, 1, 2, 3])), dtype=np.int8)
NO_SOLUTION_EVAL = -99999999.0
STOP_HOTKEY_VK = {
    "F8": 0x77,
    "F9": 0x78,
    "F10": 0x79,
}

@njit(fastmath=True, nogil=True, cache=True)
def _check_fit(board, piece, r, c):
    for i in range(piece.shape[0]):
        pr, pc = piece[i]
        if pr == -1: break
        nr, nc = r + pr, c + pc
        if nr < 0 or nr >= 8 or nc < 0 or nc >= 8: return False
        if board[nr * 8 + nc] == 1: return False
    return True

@njit(fastmath=True, nogil=True, cache=True)
def _get_clear_score(lines):
    if lines == 0: return 0
    if lines == 1: return 200
    if lines == 2: return 500
    if lines == 3: return 900
    if lines >= 4: return lines * 400
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
            for col in range(8): new_board[row * 8 + col] = 0
    for col in range(8):
        if cols_to_clear[col]:
            cleared_lines += 1
            for row in range(8): new_board[row * 8 + col] = 0
            
    return new_board, _get_clear_score(cleared_lines)

@njit(fastmath=True, nogil=True)
def _evaluate_board(board):
    empty_spaces = 0
    holes = 0
    for r in range(8):
        for c in range(8):
            if board[r * 8 + c] == 0:
                empty_spaces += 1
                is_hole = True
                if r > 0 and board[(r - 1) * 8 + c] == 0: is_hole = False
                elif r < 7 and board[(r + 1) * 8 + c] == 0: is_hole = False
                elif c > 0 and board[r * 8 + c - 1] == 0: is_hole = False
                elif c < 7 and board[r * 8 + c + 1] == 0: is_hole = False
                if is_hole: holes += 1
    return (empty_spaces * 10) - (holes * 500)

@njit(fastmath=True, nogil=True)
def _solve_turn(board, pieces):
    best_eval = -99999999.0
    best_moves = np.zeros((4, 3), dtype=np.int32)
    best_game_score = 0
    
    for perm_idx in range(24):
        order = PERMS_4[perm_idx]
        p1 = pieces[order[0]]
        for r1 in range(8):
            for c1 in range(8):
                if not _check_fit(board, p1, r1, c1): continue
                b1, score1 = _place_and_clear(board, p1, r1, c1)
                
                p2 = pieces[order[1]]
                for r2 in range(8):
                    for c2 in range(8):
                        if not _check_fit(b1, p2, r2, c2): continue
                        b2, score2 = _place_and_clear(b1, p2, r2, c2)
                        
                        p3 = pieces[order[2]]
                        for r3 in range(8):
                            for c3 in range(8):
                                if not _check_fit(b2, p3, r3, c3): continue
                                b3, score3 = _place_and_clear(b2, p3, r3, c3)
                                
                                p4 = pieces[order[3]]
                                for r4 in range(8):
                                    for c4 in range(8):
                                        if not _check_fit(b3, p4, r4, c4): continue
                                        b4, score4 = _place_and_clear(b3, p4, r4, c4)
                                        
                                        total_game_score = score1 + score2 + score3 + score4
                                        final_eval = _evaluate_board(b4) + (total_game_score * 10000)
                                        
                                        if final_eval > best_eval:
                                            best_eval = final_eval
                                            best_game_score = total_game_score
                                            best_moves[0] = [order[0], r1, c1]
                                            best_moves[1] = [order[1], r2, c2]
                                            best_moves[2] = [order[2], r3, c3]
                                            best_moves[3] = [order[3], r4, c4]
                                            
    return best_eval, best_game_score, best_moves

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
    "thread": None,
}


def _solve_with_steps(board: List[List[int]], blocks: List[List[List[int]]]) -> Dict[str, Any]:
    if len(blocks) != 4:
        return {"status": "fail", "msg": "参数校验失败：必须提供 4 个方块"}

    board_1d = np.array(board, dtype=np.int8).flatten()
    if board_1d.size != 64:
        return {"status": "fail", "msg": "参数校验失败：棋盘尺寸必须为 8x8"}

    pieces_arr = np.full((4, 16, 2), -1, dtype=np.int8)
    for i, block_coords in enumerate(blocks):
        for j, (r, c) in enumerate(block_coords):
            if j < 16:
                pieces_arr[i, j, 0] = r
                pieces_arr[i, j, 1] = c

    eval_score, game_score, moves = _solve_turn(board_1d, pieces_arr)
    if eval_score == NO_SOLUTION_EVAL:
        return {"status": "fail", "msg": "Game Over：当前盘面下这4个方块无法全部放置。"}

    steps = []
    current_sim_board = board_1d.copy()
    for move in moves:
        b_idx, r, c = int(move[0]), int(move[1]), int(move[2])
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

    return {
        "status": "success",
        "game_score_gained": int(game_score),
        "evaluation": float(eval_score),
        "steps": steps,
    }


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
                if not CURRENT_BBOX or not BOARD_BBOX:
                    raise RuntimeError("代理执行失败：缺少预备区或棋盘框选。")

                board = extract_board_from_memory(BOARD_BBOX)
                blocks = extract_blocks_from_memory(CURRENT_BBOX)
                solve_result = _solve_with_steps(board, blocks)
                if solve_result.get("status") != "success":
                    raise RuntimeError(solve_result.get("msg", "推导失败"))

                execute_result = execute_solution_steps(
                    steps=solve_result["steps"],
                    capture_bbox=CURRENT_BBOX,
                    board_bbox=BOARD_BBOX,
                    backend=backend,
                    dry_run=dry_run,
                    timing_profile=timing_profile,
                )
                if execute_result.get("status") != "success":
                    raise RuntimeError(execute_result.get("msg", "执行失败"))

                with AGENT_LOCK:
                    AGENT_STATE["cycle_count"] += 1
                    AGENT_STATE["last_error"] = ""
                    AGENT_STATE["last_msg"] = f"第 {AGENT_STATE['cycle_count']} 轮完成。"
            except Exception as e:
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
    return _solve_with_steps(req.board, req.blocks)
    

# 全局存储框选坐标
CURRENT_BBOX = None
BOARD_BBOX = None

@app.get("/api/set_bbox")
def api_set_bbox():
    global CURRENT_BBOX
    bbox = get_screen_bbox()
    if bbox and (bbox[2] - bbox[0] > 10 and bbox[3] - bbox[1] > 10):
        CURRENT_BBOX = bbox
        return {"status": "success", "bbox": bbox, "msg": f"区域已锁定: {bbox}"}
    return {"status": "fail", "msg": "框选无效或被取消"}

@app.get("/api/auto_recognize")
def api_auto_recognize():
    global CURRENT_BBOX
    if not CURRENT_BBOX:
        return {"status": "fail", "msg": "请先点击「1. 框选预备区」设定识别范围！"}
    
    try:
        blocks = extract_blocks_from_memory(CURRENT_BBOX)
        if len(blocks) != 4:
            return {
                "status": "fail",
                "msg": (
                    f"识别数量异常：检测到 {len(blocks)} 个槽位，期望值为 4。"
                    "请确保框选区域干净且包含所有方块。"
                    "已在项目根目录保存调试截图（cv_capture_1.png ~ cv_capture_3.png，自动轮转覆盖）。"
                ),
            }
        return {"status": "success", "blocks": blocks}
    except Exception as e:
        return {"status": "fail", "msg": f"视觉处理异常: {str(e)}"}

@app.get("/api/set_board_bbox")
def api_set_board_bbox():
    global BOARD_BBOX
    bbox = get_screen_bbox()
    if bbox and (bbox[2] - bbox[0] > 10 and bbox[3] - bbox[1] > 10):
        BOARD_BBOX = bbox
        return {"status": "success", "bbox": bbox, "msg": f"棋盘区域已锁定: {bbox}"}
    return {"status": "fail", "msg": "框选无效或被取消"}

@app.get("/api/recognize_board")
def api_recognize_board():
    global BOARD_BBOX
    if not BOARD_BBOX:
        return {"status": "fail", "msg": "请先点击「框选棋盘」设定识别范围！"}
    try:
        board = extract_board_from_memory(BOARD_BBOX)
        filled = sum(board[r][c] for r in range(8) for c in range(8))
        return {"status": "success", "board": board, "filled_count": filled}
    except Exception as e:
        return {"status": "fail", "msg": f"棋盘识别异常: {str(e)}"}


@app.post("/api/execute_solution")
def api_execute_solution(req: ExecuteRequest):
    global CURRENT_BBOX, BOARD_BBOX

    if not CURRENT_BBOX:
        return {"status": "fail", "msg": "请先点击「框选预备区」设定识别范围！"}
    if not BOARD_BBOX:
        return {"status": "fail", "msg": "请先点击「框选棋盘」设定识别范围！"}
    if not req.steps:
        return {"status": "fail", "msg": "执行参数缺失：steps 为空。"}

    try:
        result = execute_solution_steps(
            steps=[step.dict() for step in req.steps],
            capture_bbox=CURRENT_BBOX,
            board_bbox=BOARD_BBOX,
            backend=req.backend,
            dry_run=req.dry_run,
            timing_profile=req.timing_profile,
        )
        return result
    except Exception as e:
        return {"status": "fail", "msg": f"执行器异常: {str(e)}"}


@app.post("/api/agent/start")
def api_agent_start(req: AgentStartRequest):
    global CURRENT_BBOX, BOARD_BBOX

    if not CURRENT_BBOX:
        return {"status": "fail", "msg": "请先点击「框选预备区」设定识别范围！"}
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