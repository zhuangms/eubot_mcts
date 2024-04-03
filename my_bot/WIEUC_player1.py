# 基于MHSIEC_player.py
import logging
import os, time
import random
import sys
from typing import Optional, List, Tuple, Set
from collections import defaultdict
import chess
path2 = os.path.abspath('..')
#sys.path.append(r'C:\Users\pilot\Desktop\侦察盲棋\reconchess_project_2v\reconchess_project')
sys.path.append(r'C:\Users\18399\Desktop\stockfish\EUBOT\WIEUC\my_bot')
sys.path.append(os.getcwd())
from reconchess import Player, Color, WinReason, GameHistory, Square, LocalGame
from WIEUC_strategy1 import MonteCarloController
from utilities import Color as uColor, MCConfig, DATA_PATH
from weightedIEUC.multiprocessing_strategies import create_strategy
import platform

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s')
class WIEUC(Player):
    def __init__(self):
        self.mc_control = None
        self.is_first_sense = True
        self.is_first_move = True
        self.color = None
        self.op = None
        self.turn_num = None

        self.timeout_control = True
        self.boards: Set[str] = set()
        self._sense_weight, self._end_game = create_strategy()
        self.random_choice = False

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.boards = {board.epd(en_passant='xfen')}
        self.op = opponent_name

        self.turn_num = 0
        logging.debug('Start, color:{}, op: {}'.format(uColor(color), self.op))
        self.sf_path = ""
        if platform.system() == 'Windows':
            self.sf_path = "C:\\Users\\18399\\Desktop\\stockfish\\stock-fish-newest\\stockfish-plus\\src\\stockfish+200122_x86-64-avx2.exe"
        elif platform.system() == 'Linux':
            self.sf_path = "/home/yghuang/stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2"
        self.mc_control = MonteCarloController(LocalGame(),
                                               uColor(color),
                                               MCConfig(),
                                               self.sf_path,
                                               # "E:\\stockfish_win_x64_avx2\\stockfish-windows-2022-x86-64-avx2.exe",
                                               # "E:\\stockfish16\\stockfish-windows-x86-64-avx2.exe",
                                               # "/home/yghuang/stockfish_15.1_linux_x64_avx2/stockfish-ubuntu-20.04-x86-64-avx2",
                                               opponent_name)
        logging.debug('Style:{}, op: {}'.format(self.mc_control.is_attack, self.op))

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        self.turn_num += 1
        logging.debug("Starting turn %d.", self.turn_num)

        if self.turn_num == 1 and self.color == chess.WHITE:
            return

        if captured_my_piece:
            logging.debug('Opponent captured my piece at %s.', chess.SQUARE_NAMES[capture_square])
        else:
            logging.debug("Opponent's move was not a capture.")
        self.mc_control.handle_opponent_move_result(captured_my_piece, capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        logging.debug('choose_sense')
        # 每回合都会增加5s
        logging.error("Time left: {}".format(seconds_left))

        if self.color and self.is_first_sense:
            return None
        if self.timeout_control and seconds_left <= 50 or self.random_choice:
            logging.debug('A little time.')
            self.random_choice = True
            return random.choice(sense_actions)
        elif self.timeout_control and seconds_left <= 100:
            logging.debug('Not much time.')
            self.mc_control.config.mc_samples = 50
        elif self.timeout_control and seconds_left <= 300:
            logging.debug('Time is running out.')
            self.mc_control.config.mc_samples = 150

        self.mc_control.expand_tree()
        if self.mc_control.turn_num >= 3:
            boards = set()
            for node in self.mc_control.tree:
                for ac, cn in node.children.items():
                    boards.add(cn.env.board.epd(en_passant='xfen'))
            self.boards = boards
            self.mc_control.squares_weight = self._sense_weight(\
            self.boards, self.color, sense_actions, move_actions, seconds_left)
        return self.mc_control.get_sense_square()

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # logging.debug('handle_sense_result: {}'.format(sense_result))
        if self.color and self.is_first_sense:
            self.is_first_sense = False
        else:
            self.mc_control.handle_sense_result(sense_result)

        boards = set()
        for node in self.mc_control.tree:
            boards.add(node.env.board.epd(en_passant='xfen'))     
        self.boards = boards

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        logging.debug('choose_move, Time left: {}'.format(seconds_left))
        if seconds_left <= 300:
            self.mc_control.move_quick = True
        if seconds_left <= 50 or self.random_choice:
            self.random_choice = True
            return random.choice(move_actions)
        move_choice = self.mc_control.get_action(move_actions)
        return move_choice if move_choice != chess.Move.null() else None
    
    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        logging.debug('The requested move was %s and the taken move was %s.', requested_move, taken_move)
        if captured_opponent_piece:
            logging.debug('Move %s was a capture!', taken_move)
        self.mc_control.handle_move_result(requested_move, taken_move, capture_square)
        boards = set()
        for node in self.mc_control.tree:
            boards.add(node.env.board.epd(en_passant='xfen'))
        
        self.boards = boards

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        self.mc_control.record_state_num()
        self.mc_control.end_game()
        self._end_game()
        print("end, op name: {}, our color: {}, winner_color: {}".format(self.op,
                                                                         uColor(self.color),
                                                                         uColor(winner_color)))