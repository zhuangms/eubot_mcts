import imp
import json
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass
from time import sleep, time
from typing import List, Set, Callable, Tuple
import logging
import chess.engine 
import numpy as np
from reconchess import Square
import psutil
import atexit
# from tqdm import tqdm

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/optimization")

from stockfish_1 import STOCKFISH_EXECUTABLE, create_engine, close_stockfish
from rbc_move_score import calculate_score, ScoreConfig

from utilities import simulate_move



def make_cache_key(board: chess.Board, move: chess.Move = chess.Move.null(), prev_turn_score: int = None):
    move = simulate_move(board, move) or chess.Move.null()
    return (board.epd(en_passant="xfen") + ' ' + move.uci() + ' ' +
            (str(prev_turn_score) if prev_turn_score is not None else '-'))

def worker(request_queue, response_queue, score_config, num_threads):
    engine = create_engine()
    atexit.register(lambda: close_stockfish(engine))
    if num_threads:
        engine.configure({'Threads': num_threads})
    while True:
        if not request_queue.empty():
            board, move, prev_turn_score= request_queue.get()
            try:
                score = calculate_score(board=board, move=move, prev_turn_score=prev_turn_score or 0,
                                        engine=engine, score_config=score_config)
            except chess.engine.EngineTerminatedError:
                logging.error('Stockfish engine died while analysing (%s).',
                             make_cache_key(board, move, prev_turn_score))
                response_queue.put({make_cache_key(board, move, prev_turn_score): score_config.into_check_score})
                engine = create_engine()
                atexit.register(lambda: close_stockfish(engine))
                if num_threads:
                    engine.configure({'Threads': num_threads})
            else:
                response_queue.put({make_cache_key(board, move, prev_turn_score): score})
                # logging.debug('worker计算出score为：{}', format(score))
        else:
            sleep(0.001)

def multiprocessing_score(
    score_config: ScoreConfig = ScoreConfig(),
    board_weight_90th_percentile: float = 1000,
    num_workers: int = 2,
    num_threads: int = None,
    load_cache_data: bool = False,
):
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    if load_cache_data:
        logging.debug('Loading cached scores from file.')
        with open('score_cache.json', 'r') as file:
            score_data = json.load(file)
        score_cache = score_data['cache']
    else:
        score_cache = dict()
    workers = [mp.Process(target=worker, args=(request_queue, response_queue, score_config, num_threads)) for _ in range(num_workers)]
    for process in workers:
        process.start()
    score_calc_times = []

    def memo_calc_score(board: chess.Board, move: chess.Move = chess.Move.null(), prev_turn_score: int = None):
        key = make_cache_key(board, move, prev_turn_score)
        if key in score_cache:
            return score_cache[key]
        request_queue.put((board, move, prev_turn_score))
        return None

    def memo_calc_set(requests):

        filtered_requests = [(board, simulate_move(board, move) or chess.Move.null(), prev_turn_score) 
                                for board, move, prev_turn_score in requests]                              
        start = time()
        results = {make_cache_key(board, move, prev_turn_score):
                    memo_calc_score(board, move, prev_turn_score)
                    for board, move, prev_turn_score in filtered_requests}

        num_new = sum(1 for score in results.values() if score is None)
        while any(score is None for score in results.values()):
            response = response_queue.get()
            score_cache.update(response)
            results.update(response)

        duration = time() - start
        if num_new:
            score_calc_times.append((num_new, duration))
        return results

    def weight_board_probability(score):
        return 1 / (1 + np.exp(-2 * np.log(3) / board_weight_90th_percentile * score))

    def calculate_prob(board: chess.Board, valid_actions, prev_turn_score: int = None):
        board_score_reqs = []
        for a_ in valid_actions:
            # board.turn = not our_color
            if(a_ == None):
                board_score_reqs.append((board, chess.Move.null(), prev_turn_score))
            else:
                board_score_reqs.append((board, a_, prev_turn_score))

        board_score_dict = memo_calc_set(board_score_reqs)
        value = []
        for a_ in valid_actions:
            if(a_ == None):
                op_score = board_score_dict[make_cache_key(board, chess.Move.null(), prev_turn_score)]
            else:
                op_score = board_score_dict[make_cache_key(board, a_, prev_turn_score)]
            value.append(weight_board_probability(op_score))
        return value

    def end_game():
        # for process in workers:
        #     pid = process.pid
        #     # 获取所有子进程
        #     children = psutil.Process(pid).children()

        #     # 终止子进程
        #     for child in children:
        #         child.terminate()

        #     # 等待子进程终止
        #     psutil.wait_procs(children)

        [process.terminate() for process in workers]
        [process.join() for process in workers]

    return calculate_prob, end_game