"""
实现新的感知策略
1. 如果有被吃的子，优先探测
2. 直接扩展当前的所有的节点，遍历所有的可能的感知区域，取出差异最大的感知块，作为我们的感知结果去探测
"""
from cgi import print_directory
import copy
import itertools
import logging
from random import random
import time
import math
import csv
import os
import sys
import chess
import numpy as np
from stockfish import Stockfish
from collections import Counter
from typing import Set
from tqdm import tqdm
import json
import multiprocessing as mp
import chess.engine
from optimization.multiprocessing_score import multiprocessing_score
from weightedIEUC.utilitie.rbc_move_score import ScoreConfig

path2 = os.path.abspath('..')

from reconchess import LocalGame, List, Tuple, Square, Optional
#from train import Color
from utilities import Color, MCConfig, get_sense_pieces


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s')

white_first_four = [['g1f3', 'b1c3'], ['e2e4', 'g1f3']]

black_first_four = [['g8f6', 'b8c6'], ['e7e5', 'g8f6']]

attack_op = ['Testudo',  'trout']

white_first_sense = [43, 33, 26]
black_first_sense = [19, 25, 34]

class GameNode(object):
    def __init__(self, env: LocalGame, prob, value, parent):
        self.env = env
        self.prob = prob
        self.value = value
        self.parent = parent
        self.children = {}

    def expand(self, actions, games, prob, value):
        for i, j, k, l in zip(actions, games, prob, value):
            # logging.debug("expand \n{}".format(str(j.board)))
            if i not in self.children.keys():
                self.children[i] = GameNode(j, k, l, self)


class MonteCarloController(object):
    def __init__(self, env: LocalGame, color: Color, config: MCConfig, fish_path, op_name):
        self.env = env
        self.root = GameNode(self.env, 1, 1, None)
        self.tree = [self.root]
        self.color = color
        self.config = config
        self.op_name = op_name
        self.op_cap_our = False
        self.op_cap_square = None

        self.min_value = 0
        self.fish_path = fish_path
        print("debug: ", fish_path)
        self.stock_fish = Stockfish(fish_path, depth=15,parameters={"Threads": 8, "MCTS": True})
        self._multiprocessing_score, self._end_game = multiprocessing_score()
        self.turn_num = 0
        self.squares_weight = {}

        if self.op_name in attack_op:
            self.is_attack = 1
        else:
            self.is_attack = 0

        self.right_move = None
        self.all_move_num = 0
        self.right_move_num = 0
        self.is_record = False

        # 记录信息
        if self.is_record:
            self.history = '683338.json'
            f = open(os.path.join('sense_err', self.history))
            s = json.load(f)
            if self.color.color:
                self.history_fen = s['fens_after_move']['false']
                self.op_name = [s['black_name']]
            else:
                self.history_fen = s['fens_after_move']['true']
                self.op_name = [s['white_name']]
            f.close()

        self.i_e = []
        self.s_n = []
        self.s_c = []
        self.state_num = self.op_name
        self.state_ie = ['state_ie']
        self.time_usage = ['time_usage']
        self.state_correct = ['state_correct']
        self.pick_right = ['pick_right']
        self.sense_quick = False
        self.move_quick = False
        env.start()

    def record_state_num(self):
        if self.is_record:
            state_num_avg = sum(self.s_n) / len(self.s_n)
            ie_avg = sum(self.i_e) / len(self.i_e)
            sc_avg = sum(self.s_c) / len(self.s_c)
            if self.all_move_num != 0:
                r_m_p = self.right_move_num / self.all_move_num
            else:
                r_m_p = 0

            self.state_num.insert(0, str(round(state_num_avg, 5)))
            self.state_ie.insert(0, str(round(ie_avg, 5)))
            self.state_correct.insert(0, str(round(sc_avg, 5)))
            self.pick_right.insert(0, str(round(r_m_p, 5)))

            s = open('history.csv', 'a+', newline='')
            w = csv.writer(s)
            w.writerow(self.state_num)
            w.writerow(self.state_ie)
            w.writerow(self.state_correct)
            w.writerow(self.pick_right)
            s.close()

    # # 更新pick_right和state_correct
    def check_right_prob(self):
        current_turn = self.tree[0].env.board.fullmove_number

        if len(self.tree) > 50 or self.move_quick:
            self.stock_fish.set_depth(12)
        elif len(self.tree) > 10:
            self.stock_fish.set_depth(15)
        else:
            self.stock_fish.set_depth(20)

        if len(self.tree) < 10 and self.move_quick:
            self.stock_fish.set_depth(15)

        if self.color.color:
            his_fen = self.history_fen[current_turn-2]
        else:
            his_fen = self.history_fen[current_turn-1]

        self.right_move = None
        for i in range(len(self.tree)):
            board = self.tree[i].env.board
            if board.fen() == his_fen:
                self.s_c.append(round(self.tree[i].value, 5))
                self.state_correct.append(str(round(self.tree[i].value, 5)))
                enemy_king_square = board.king(not self.color.color)
                if enemy_king_square is not None:
                    # if there are any ally pieces that can take king, execute one of those moves
                    enemy_king_attackers = board.attackers(self.color.color, enemy_king_square)
                    if enemy_king_attackers:
                        attacker_square = enemy_king_attackers.pop()
                        self.right_move = chess.Move(attacker_square, enemy_king_square)
                        # logging.debug("Kill king")
                        #break
                        # return [chess.Move(attacker_square, enemy_king_square)]
                # stock_fish
            try:
                if board.is_valid():
                    self.stock_fish.set_fen_position(board.fen())
                    self.right_move = self.stock_fish.get_best_move()
            except:
                self.stock_fish = Stockfish(self.fish_path, depth=15)
                #break

    def _normalize(self):
        sum_p = 0
        sum_v = 0
        for i in self.tree:
            sum_p += i.prob
            sum_v += i.value
        for i in range(len(self.tree)):
            self.tree[i].prob /= sum_p
            self.tree[i].value /= sum_v

    def _normalize_value(self, values, min_value, parent_prob: int = 1):
        min = min_value - 1
        sum_ = 0
        for i in range(len(values)):
            values[i] -= min
            sum_ += values[i]
        for i in range(len(values)):       
            values[i] = parent_prob * values[i]/sum_      
        return values

    def _clean(self):
        logging.debug("before clean: {}".format(len(self.tree)))
        # 按照棋盘排序
        tmp = sorted(self.tree, key=lambda x: str(x.env.board))
        # 把相同棋盘分组
        merge = itertools.groupby(tmp, key=lambda k: str(k.env.board))
        out = []
        for i, j in merge:
            j = list(j)
            c = 0
            v = 0
            for _ in j:
                c += _.prob
                v += _.value
            j[0].prob = c
            j[0].value = v
            out.append(j[0])
        self.tree = out
        self._normalize()
        logging.debug("after clean: {}".format(len(self.tree)))
        if len(self.tree) == 1:
            logging.debug("board: \n {}".format(self.tree[0].env.board))

    def _mc_sample(self, move_actions: List[chess.Move]):
        if len(self.tree) > 50 or self.move_quick:
            self.stock_fish.set_depth(12)
        elif len(self.tree) > 10:
            self.stock_fish.set_depth(15)
        else:
            self.stock_fish.set_depth(20)

        if len(self.tree) < 10 and self.move_quick:
            self.stock_fish.set_depth(15)
        if self.config.mc_samples > len(self.tree):
            sample_node = self.tree
            sample = len(self.tree)
        else:
            sample_node = np.random.choice(self.tree,
                                           self.config.mc_samples,
                                           replace=False,
                                           p=[i.prob for i in self.tree])
            sample = self.config.mc_samples
        sample_result = {}
        start = time.time()
        for i in range(sample):
            start_node = sample_node[i]
            value_prob = start_node.value
            board = start_node.env.board
            enemy_king_square = board.king(not self.color.color)
            if enemy_king_square is not None:
                # if there are any ally pieces that can take king, execute one of those moves
                enemy_king_attackers = board.attackers(self.color.color, enemy_king_square)
                if enemy_king_attackers:
                    attacker_square = enemy_king_attackers.pop()
                    kill_king_action = chess.Move(attacker_square, enemy_king_square)
                    if kill_king_action in move_actions and kill_king_action in start_node.env.move_actions():
                        if kill_king_action in sample_result.keys():
                            sample_result[kill_king_action] += value_prob
                        else:
                            sample_result[kill_king_action] = value_prob
                        continue                      
                    # return [chess.Move(attacker_square, enemy_king_square)]

            try:
                if board.is_valid():
                    self.stock_fish.set_fen_position(board.fen())
                    move = self.stock_fish.get_best_move()
                else:
                    continue
            except:
                self.stock_fish = Stockfish(self.fish_path, depth=15)
                continue
            if move is None:
                continue
            else:
                action = chess.Move.from_uci(move)
            if action not in start_node.env.move_actions() or action not in move_actions:
                continue            

            if action in sample_result.keys():
                sample_result[action] += value_prob
            else:
                sample_result[action] = value_prob
            self_king_square = board.king(self.color.color)
            self_king_attackers = board.attackers(not self.color.color, self_king_square)
            if self_king_attackers:
                sample_result[action] += value_prob
        end = time.time()
        # logging.debug(f'sample:{sample}, time:{end - start}')
        return sample_result
 
    def expand_tree(self):
        start = time.time()
        if len(self.tree) > 10:
            self.stock_fish.set_depth(12)
        else:
            self.stock_fish.set_depth(15)        
        for counter_i in tqdm(range(len(self.tree)), disable=False,
                            desc=f'{chess.COLOR_NAMES[self.color.color]} Expanding '
                                f'{len(self.tree)} boards', unit='boards'):
            valid_actions = self.tree[counter_i].env.move_actions()
            valid_actions.append(None)
            valid_pro = [1] * len(valid_actions)
            games = []
            actions = []
            policies = []
            values = []
            for a_, p_ in zip(valid_actions, valid_pro):
                env_t = copy.deepcopy(self.tree[counter_i].env)
                _, _, capture = env_t.move(a_)
                #print(a_, capture, self.op_cap_square)
                if capture == self.op_cap_square:
                    env_t.end_turn()
                    v_ = 0
                    games.append(env_t)
                    actions.append(a_)
                    policies.append(p_)
                    values.append(v_)
            values = self._multiprocessing_score(self.tree[counter_i].env.board, actions, self.tree[counter_i].value)
            # logging.debug('ALL Done, usage: {}'.format(end - start))
            if(len(values) != 0):
                self.min_value = values[0]
                for counter_j in range(len(values)):
                    if(values[counter_j] < self.min_value):
                        self.min_value = values[counter_j]

            values_prob = self._normalize_value(values, self.min_value, self.tree[counter_i].value)
            self.tree[counter_i].expand(actions, games, policies, values_prob)

        end = time.time()
        logging.debug('ALL Done, time: {}, tree: {}'.format(end - start, len(self.tree)))

    def get_sense_square(self, score_config: ScoreConfig = ScoreConfig()):
        best_square = 9
        information_entropy = 10000
        type_len = 0
        self.turn_num += 1
        start_time = time.time()
        if self.turn_num < 4:
            if self.color.color:
                best_square = white_first_sense[self.turn_num - 1]
            else:
                best_square = black_first_sense[self.turn_num - 1]
        else:
            temp = 0
            start = time.time()
            squares_weight = self.squares_weight
            ieList = {}
            weightList = {}
            sum_ie = 0
            sum_weight = 0
            for counter_i in tqdm(range(1, 7), disable=False,
                                desc=f'{chess.COLOR_NAMES[self.color.color]} Calculating '
                                    'information entropy in 36 spots', unit='*6 spots'):
                for counter_j in tqdm(range(1, 7), disable=False, leave=False,
                                desc=f'{chess.COLOR_NAMES[self.color.color]} Calculating '
                                    'information entropy in 6 spots', unit='spot'):
                    type_env = {}
                    for counter_t in range(len(self.tree)):
                        for valid_actions, c in self.tree[counter_t].children.items():
                            square = get_sense_pieces(self.tree[counter_t].children[valid_actions].env.board,
                                                      counter_i, counter_j)
                            if square not in type_env.keys():
                                type_env[square] = []
                            type_env[square].append(c.value)
                    i_e = 0
                    for s, l in type_env.items():
                        type_prob = sum(l)
                        temp = 0
                        for k in range(len(l)):
                            l[k] /= type_prob
                            temp += l[k] * math.log(l[k], 2) * (-1)
                        # if temp >= i_e:
                        #          i_e = temp
                        if len(self.tree) > 250:
                            if temp >= i_e:
                                i_e = temp
                        else:
                            i_e += temp * type_prob
                    if counter_i*8+counter_j not in squares_weight.keys():
                        weight = 1
                    else:
                        if squares_weight[counter_i*8+counter_j] >= score_config.checkmate_score:
                            # 有将军的机会，直接探测国王位置
                            weight = 0
                        else:
                            weight = 1 / (squares_weight[counter_i*8+counter_j] + 1)
                    ieList[counter_i * 8 + counter_j] = i_e
                    weightList[counter_i * 8 + counter_j] = weight
                    sum_ie += i_e
                    sum_weight += weight
                    # c1 = 0
                    # c2 = 1
                    # if c1 * weight + c2 * i_e < information_entropy:
                    #     best_square = counter_i * 8 + counter_j
                    #     information_entropy = c1 * weight + c2 * i_e
            
            if sum_ie != 0 and sum_weight != 0:
                c1 = 0.1
                c2 = 0.9
                for i in range(1, 7):
                    for j in range(1, 7):
                        idx = i * 8 + j
                        # 归一化
                        weightList[idx] = weightList[idx] / sum_weight
                        ieList[idx] = ieList[idx] / sum_ie
                        # 有将军的机会，直接探测国王位置
                        if weightList[idx] == 0:
                            best_square = idx
                            return best_square
                        if c1 * weightList[idx] + c2 * ieList[idx] < information_entropy:
                            best_square = idx
                            information_entropy = c1 * weightList[idx] + c2 * ieList[idx]

            # logging.debug(f'ieList:{ieList}')
            # logging.debug(f'weightList:{weightList}')
            # logging.debug('squares_weight:{}'.format(squares_weight))
            end = time.time()
            logging.debug(f'information_entropy: {information_entropy}, tree: {len(self.tree)}, usage: {end - start}')
        self.time_usage.append(str(time.time()-start_time))
        return best_square

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        success = []
        for t in self.tree:
            for ac, cn in t.children.items():
                is_match = True
                for square, piece in sense_result:
                    if cn.env.board.piece_at(square) != piece:
                        is_match = False
                        break
                if is_match:
                    success.append(cn)
        #logging.debug("ALL Match node: len {}".format(len(success)))
        if len(success) > 0:
            self.tree = success
            self._normalize()
            self._clean()
        if self.is_record:
            self.check_right_prob()
            i_e = 0
            for s in self.tree:
                i_e += s.value * math.log(s.value, 2) * (-1)
            self.state_ie.append(str(round(i_e, 5)))
            self.i_e.append(round(i_e, 5))
            self.s_n.append(len(self.tree))
            self.state_num.append(str(len(self.tree)))
        # self._update_env(sense_result)

    def get_action(self, move_actions: List[chess.Move]):

        if self.turn_num < 3:
            if self.color.color:
                if(self.turn_num == 0):
                    self.turn_num += 1
                best_action = chess.Move.from_uci(white_first_four[self.is_attack][self.turn_num-1])
            else:
                best_action = chess.Move.from_uci(black_first_four[self.is_attack][self.turn_num-1])
        else:
            result = self._mc_sample(move_actions)
            # when self.tree is empty, result is empty.
            if len(result) == 0:
                best_action = np.random.choice(move_actions)
            else:
            #word_counts = Counter(result)
                best_action_prob = max(result.items(), key=lambda x:x[1])
                best_action = best_action_prob[0]

            if self.is_record:
                if self.right_move is not None:
                    self.all_move_num += 1
                    if str(best_action) == str(self.right_move):
                        self.right_move_num += 1

        #.debug("choose action: {}".format(best_action))
        return best_action

    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        #logging.debug("opponent_move_result: {}, {}".format(captured_my_piece, capture_square))
        self.op_cap_our = captured_my_piece
        self.op_cap_square = capture_square

    def handle_move_result(self, request_move, taken_move, capture):
        #logging.info("cut by move result, R: {}, T: {}, C:{}".format(request_move, taken_move, capture))
        right = []
        for i in range(len(self.tree)):
            node = self.tree[i]
            if request_move in node.env.move_actions():
                _, t, c = node.env.move(request_move)
                node.env.end_turn()
            else:
                node.env.move(np.random.choice(node.env.move_actions()))
                node.env.end_turn()
            # 上一步生成动作时可能涉及棋盘抽样，某些动作在部分棋盘中非法
                continue
            if t == taken_move and c == capture:
                enemy_king_square = node.env.board.king(not node.env.turn)
                if enemy_king_square is not None:
                    right.append(node)

        #logging.debug("After CUT, left len {}".format(len(right)))
        if len(right) > 0:
            self.tree = right
            self._normalize()
            self._clean()
    
    def end_game(self):
        self._end_game()