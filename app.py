import numpy as np
from numba import njit
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import itertools
import uvicorn
from vision_core import get_screen_bbox, extract_blocks_from_memory, extract_board_from_memory

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 核心算法 (Numba JIT 保持极致性能) ---
PERMS_4 = np.array(list(itertools.permutations([0, 1, 2, 3])), dtype=np.int8)

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

@app.post("/solve")
def solve(req: SolveRequest):
    if len(req.blocks) != 4:
        return {"status": "fail", "msg": "参数校验失败：必须提供 4 个方块"}

    board_1d = np.array(req.board, dtype=np.int8).flatten()
    pieces_arr = np.full((4, 16, 2), -1, dtype=np.int8)
    
    for i, block_coords in enumerate(req.blocks):
        for j, (r, c) in enumerate(block_coords):
            if j < 16:
                pieces_arr[i, j, 0] = r
                pieces_arr[i, j, 1] = c
            
    eval_score, game_score, moves = _solve_turn(board_1d, pieces_arr)
    
    if eval_score == -99999999.0:
        return {"status": "fail", "msg": "Game Over：当前盘面下这4个方块无法全部放置。"}
    
    # 🌟 核心修改：沙盒推演回放生成
    steps = []
    current_sim_board = board_1d.copy()
    
    for move in moves:
        b_idx, r, c = int(move[0]), int(move[1]), int(move[2])
        piece_coords = req.blocks[b_idx]
        
        # 记录放置前的快照
        board_before = current_sim_board.tolist()
        
        # 执行一次推演
        next_board, step_score = _place_and_clear(current_sim_board, pieces_arr[b_idx], r, c)
        
        steps.append({
            "step_order": len(steps) + 1,
            "block_index": b_idx,
            "place_row": r,
            "place_col": c,
            "coords": piece_coords,
            "score_gained": int(step_score),
            "board_before": board_before,     # 消除前的底板
            "board_after": next_board.tolist() # 消除后的底板
        })
        current_sim_board = next_board
        
    return {
        "status": "success",
        "game_score_gained": int(game_score),
        "evaluation": float(eval_score),
        "steps": steps
    }
    

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

@app.get("/")
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    print(">> 拉海诺测试台后端 V3 启动：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")