"""
实现新的感知策略
1. 如果有被吃的子，优先探测
2. 直接扩展当前的所有的节点，遍历所有的可能的感知区域，取出差异最大的感知块，作为我们的感知结果去探测
"""
from cgi import print_directory
import copy
import itertools
import logging
import time
import math
import csv
import os
import sys
import chess
import numpy as np
from stockfish import Stockfish
from collections import Counter
from tqdm import tqdm
import json

import multiprocessing as mp
import chess.engine
from optimization.multiprocessing_score import multiprocessing_score

path2 = os.path.abspath('..')

sys.path.append(r'C:\Users\pilot\Desktop\侦察盲棋\reconchess_project_2v\reconchess_project')

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
        # 敌方是否捕获我方棋子
        self.op_cap_our = False
        # 敌方捕获我方棋子的位置
        self.op_cap_square = None

        self.min_value = 0
        self.stock_fish = Stockfish(fish_path, depth=15)
        self._multiprocessing_score, self._end_game = multiprocessing_score()
        self.turn_num = 0

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
            self.history = '675364.json'
            f = open(self.history)
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
        
            s = open('record.csv', 'a+', newline='')
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
            # print(self.tree[i].value)
            board = self.tree[i].env.board
            # print(board.fen(), his_fen)
            if board.fen() == his_fen:
                print(board)
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
                if self.stock_fish.is_fen_valid(board.fen()):
                    self.stock_fish.set_fen_position(board.fen())
                else:
                    continue
                # print(self.stock_fish.get_evaluation())
                self.right_move = self.stock_fish.get_best_move()
                # break

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
        tmp = sorted(self.tree, key=lambda x: str(x.env.board))
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
        for i in range(sample):
            start = time.time()
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
                    # if kill_king_action not in start_node.env.move_actions() or kill_king_action not in move_actions:
                    #     # 采取的动作是黑方的，节点回合是白方的
                    #     logging.debug('color:{}'.format(chess.COLOR_NAMES[start_node.env.turn]))
                    #     logging.debug('board:{}'.format(board.fen()))
                    #     logging.debug('node_move_actions:{}'.format(start_node.env.move_actions()))
                    #     logging.debug('move_actions:{}'.format(move_actions))
                    #     logging.debug('king_action_error:{}'.format(kill_king_action))
                    # 此动作既是实际棋盘的合法动作，又是当前预估棋盘的合法动作
                    if kill_king_action in move_actions and kill_king_action in start_node.env.move_actions():
                        if kill_king_action in sample_result.keys():
                            sample_result[kill_king_action] += value_prob
                        else:
                            sample_result[kill_king_action] = value_prob
                        continue
                        #logging.debug("Kill king")
                    # return [chess.Move(attacker_square, enemy_king_square)]
            if self.stock_fish.is_fen_valid(start_node.env.board.fen()):
                self.stock_fish.set_fen_position(start_node.env.board.fen())
            else:
                continue
            move = self.stock_fish.get_best_move()
            if move is None:
                continue
            else:
                action = chess.Move.from_uci(move)
            if action not in start_node.env.move_actions() or action not in move_actions:
                # logging.debug('color:{}'.format(chess.COLOR_NAMES[start_node.env.turn]))
                # logging.debug('board:{}'.format(board.fen()))
                # logging.debug('node_move_actions:{}'.format(start_node.env.move_actions()))
                # logging.debug('move_actions:{}'.format(move_actions))
                # logging.debug('stockfish_error:{}'.format(action))
                continue
            end = time.time()
            if action in sample_result.keys():
                sample_result[action] += value_prob
            else:
                sample_result[action] = value_prob
            self_king_square = board.king(self.color.color)
            self_king_attackers = board.attackers(not self.color.color, self_king_square)
            if self_king_attackers:
                sample_result[action] += value_prob

        return sample_result
 
    
    def get_sense_square(self):
        start = time.time()
        e_k_s_a_list = []
        if len(self.tree) > 10:
            self.stock_fish.set_depth(12)
        else:
            self.stock_fish.set_depth(15)
        for counter_i in tqdm(range(len(self.tree)), disable=False,
                            desc=f'{chess.COLOR_NAMES[self.color.color]} Expanding '
                                f'{len(self.tree)} boards', unit='boards'):
            # logging.debug(self.tree[counter_i].env.turn)
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
            end = time.time()
            if(len(values) != 0):
                self.min_value = values[0]
                for counter_j in range(len(values)):
                    if(values[counter_j] < self.min_value):
                        self.min_value = values[counter_j]

            values_prob = self._normalize_value(values, self.min_value, self.tree[counter_i].value)
            self.tree[counter_i].expand(actions, games, policies, values_prob)
            # 确保所有棋盘局面的当前棋手均为我方（一般为黑方）？
            # self.tree[counter_i].env.end_turn()
        best_square = 9
        information_entropy = 10000
        type_len = 0
        self.turn_num += 1
        if self.turn_num < 4:
            if self.color.color:
                best_square = white_first_sense[self.turn_num - 1]
            else:
                best_square = black_first_sense[self.turn_num - 1]
        else:
            temp = 0
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
                    if i_e < information_entropy:
                        best_square = counter_i * 8 + counter_j
                        information_entropy = i_e
        end = time.time()
        logging.debug('ALL Done, best result: {}, information_entropy: {}, usage: {}'.format(best_square,
                                                                                   information_entropy,
                                                                                   end - start))
        self.time_usage.append(str(end-start))
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
            # 上一部生成动作时可能涉及棋盘抽样，某些动作在部分棋盘中非法
                node.env.move(np.random.choice(node.env.move_actions()))
                node.env.end_turn()
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