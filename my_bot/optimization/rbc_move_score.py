from dataclasses import dataclass

import chess.engine
from reconchess.utilities import is_psuedo_legal_castle, capture_square_of_move
import os
import sys
sys.path.append(os.getcwd())
from utilities import simulate_move
# from reconchess.utilitie import capture_square_of_movee

@dataclass
class ScoreConfig:
    capture_king_score: float = 50_000
    checkmate_score: int = 30_000
    into_check_score: float = -40_000
    search_depth: int = 8
    reward_attacker: float = 300
    require_sneak: bool = True
    

def calculate_score(engine: chess.engine.SimpleEngine, 
                    board, move=chess.Move.null(),
                    prev_turn_score=0,
                    score_config: ScoreConfig = ScoreConfig()):

    pov = board.turn

    if move != chess.Move.null() and not is_psuedo_legal_castle(board, move):
        if not board.is_pseudo_legal(move):           
            revised_move = simulate_move(board, move)
            if revised_move is not None:
                return calculate_score(engine, board, revised_move, prev_turn_score, score_config)
            return calculate_score(engine, board, chess.Move.null(), prev_turn_score, score_config)
        if board.is_capture(move):
            if board.piece_at(capture_square_of_move(board, move)).piece_type is chess.KING:
                return score_config.capture_king_score

    next_board = board.copy()
    next_board.push(move)
    next_board.clear_stack()

    if next_board.was_into_check() or not next_board.is_valid():
        return score_config.into_check_score
    try:
        engine_result = engine.analyse(next_board, chess.engine.Limit(depth=score_config.search_depth))
        score = engine_result['score'].pov(pov).score(mate_score=score_config.checkmate_score)
    except:
        return score_config.into_check_score

    king_attackers = next_board.attackers(pov, next_board.king(not pov))
    if king_attackers:

        if not score_config.require_sneak:
            score += score_config.reward_attacker
        elif not next_board.is_capture(move) or any([square != move.to_square for square in king_attackers]):
            score += score_config.reward_attacker

    # score -= prev_turn_score

    return score