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
path2 = os.path.abspath('..')
sys.path.append(r'C:\Users\pilot\Desktop\侦察盲棋\reconchess_project_2v\reconchess_project')
from reconchess import LocalGame, List, Tuple, Square, Optional
from utilities import Color, MCConfig, get_sense_pieces
# 第一/二优代码
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

        self.max_value = 0
        self.stock_fish = Stockfish(fish_path, depth=15)

        self.time_usage = []

        if self.op_name in attack_op:
            self.is_attack = 1
        else:
            self.is_attack = 0

        self.right_move = None
        self.all_move_num = 0
        self.right_move_num = 0
        self.is_record = True
        self.i_e = []
        self.s_c = []
        self.state_num = [self.op_name]
        self.state_ie = ['state_ie']
        self.state_correct = ['state_correct']
        self.pick_right = ['pick_right']

        self.sense_quick = False
        self.move_quick = False
        env.start()



    def record_state_num(self):
        ie_avg = sum(self.i_e) / len(self.i_e)
        self.state_ie.insert(0, str(ie_avg))
        s = open('record.csv', 'w+', newline='')
        w = csv.writer(s)
        w.writerow(self.state_num)
        w.writerow(self.state_ie)
        w.writerow(self.time_usage)
        # w.writerow(self.state_correct)
        # w.writerow(self.pick_right)
        s.close()


    def _normalize(self):
        # normalize
        sum_ = 0
        sum_v = 0
        for i in self.tree:
            sum_ += i.prob
            sum_v += i.value
        for i in range(len(self.tree)):
            self.tree[i].prob /= sum_
            self.tree[i].value /= sum_v

    def _normalize_value(self, values, max_value, parent_prob: int = 1):
        # normalize
        max = max_value+1
        sum_ = 0
        for i in range(len(values)):
            values[i] -= max
            sum_ += values[i]

        for i in range(len(values)):
            values[i] = parent_prob * values[i]/sum_

        return values

    def _clean(self):
        # logging.debug("before clean: {}".format(len(self.tree)))
        # 从移除更改为合并
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
        # logging.debug("after clean: {}".format(len(self.tree)))
        self.state_num.append(str(len(self.tree)))
        # if len(self.tree) == 1:
        #     logging.debug("board: \n {}".format(self.tree[0].env.board))

    def _mc_sample(self):
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
                enemy_king_attackers = board.attackers(self.color.color, enemy_king_square)
                if enemy_king_attackers:
                    attacker_square = enemy_king_attackers.pop()
                    if chess.Move(attacker_square, enemy_king_square) in sample_result.keys():
                        sample_result[chess.Move(attacker_square, enemy_king_square)] += value_prob
                    else:
                        sample_result[chess.Move(attacker_square, enemy_king_square)] = value_prob
                    #logging.debug("Kill king")
                    continue
                    # return [chess.Move(attacker_square, enemy_king_square)]
            # stock_fish
            self.stock_fish.set_fen_position(start_node.env.board.fen())
            #print(self.stock_fish.get_evaluation())
            move = self.stock_fish.get_best_move()
            #print(move)
            # logging.debug("stockfish move: {}".format(move))
            if move is None:
                action = np.random.choice(start_node.env.move_actions())
            else:
                action = chess.Move.from_uci(move)
            end = time.time()
            #logging.debug('Sample {}, usage: {}'.format(i, end - start))

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
        e_k_s = []
        if len(self.tree) > 10:
            self.stock_fish.set_depth(12)
        else:
            self.stock_fish.set_depth(15)
        # 对于每一个树, 扩展节点
        for i in range(len(self.tree)):
            a = self.tree[i].env.move_actions()
            a.append(None)
            p = [1] * len(a)
            games = []
            actions = []
            policies = []
            values = []
            self.max_value = 0
            for a_, p_ in zip(a, p):
                env_t = copy.deepcopy(self.tree[i].env)
                _, _, capture = env_t.move(a_)
                #print(a_, capture, self.op_cap_square)
                if capture == self.op_cap_square:
                    env_t.end_turn()
                    v_ = 0
                    enemy_king_square = env_t.board.king(not env_t.turn)
                    if enemy_king_square is not None:
                        # if there are any ally pieces that can take king, execute one of those moves
                        enemy_king_attackers = env_t.board.attackers(env_t.turn, enemy_king_square)
                        # stockfish couldn't end the game
                        if enemy_king_attackers:
                            e_k_s.append(enemy_king_square)
                    if len(self.tree) > 50 or self.sense_quick:
                        # logging.debug("jump")
                        v_ = 1
                        self.max_value = v_
                    else:
                        if enemy_king_square is not None:
                            if enemy_king_attackers:
                                v_ = 0
                            else:
                                self.stock_fish.set_fen_position(env_t.board.fen())
                                t_v_ = self.stock_fish.get_evaluation()
                                v_ = t_v_['value']
                                if(v_ > self.max_value):
                                    self.max_value = v_

                    # if enemy_king_square is not None:
                    #     if enemy_king_attackers:
                    #         v_ = 0
                    #     else:
                    #         self.stock_fish.set_fen_position(env_t.board.fen())
                    #         t_v_ = self.stock_fish.get_evaluation()
                    #         v_ = t_v_['value']
                    #         if(v_ > self.max_value):
                    #             self.max_value = v_

                    #logging.debug('Move: {} Value: {}'.format(a_, v_,))
                    games.append(env_t)
                    actions.append(a_)
                    policies.append(p_)
                    values.append(v_)

            # Take the maximum value of the action that can end the game
            for j in range(len(values)):
                if(values[j] == 0):
                    values[j] = self.max_value

            values_prob = self._normalize_value(values, self.max_value, self.tree[i].value)
            # expand tree
            self.tree[i].expand(actions, games, policies, values_prob)
        if len(e_k_s) > 0:
            word_counts = Counter(e_k_s)
            enemy_king_square = word_counts.most_common(1)[0][0]

        end = time.time()

        best_square = 9
        information_entropy = 10000
        type_len = 0

        current_turn = self.tree[0].env.board.fullmove_number
        if current_turn < 4:
            if self.color.color:
                best_square = white_first_sense[current_turn - 1]
            else:
                best_square = black_first_sense[current_turn - 1]

        elif len(e_k_s) >= len(self.tree)*2 and len(self.tree) < 250:
            best_square = enemy_king_square
            if best_square < 8:
                best_square += 8
            if best_square > 55:
                best_square -= 8
            if best_square % 8 == 0:
                best_square += 1
            if best_square % 7 == 0:
                best_square -= 1
            # logging.debug("try to kill king, sense: {}".format(best_square))

        else:
            for i in range(1, 7):
                for j in range(1, 7):
                    # calculate the information_entropy of this sense_square
                    type_env = {}
                    square_list = []
                    for t in range(len(self.tree)):
                        for a, c in self.tree[t].children.items():
                            square = get_sense_pieces(self.tree[t].children[a].env.board,
                                                      i, j)
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
                        best_square = i * 8 + j
                        information_entropy = i_e


        # end = time.time()
        logging.debug('ALL Done, best result: {}, information_entropy: {}, usage: {}'.format(best_square,
                                                                                   information_entropy,
                                                                                   end - start))
        self.time_usage.append(end - start)
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
        self.tree = success
        # normalize
        self._normalize()
        # clean
        self._clean()

        #self.check_right_prob()

        if self.is_record:
            i_e = 0
            for s in self.tree:
                i_e += s.value * math.log(s.value, 2) * (-1)
            self.state_ie.append(str(i_e))
            self.i_e.append(i_e)
        # # 更新节点地图
        # self._update_env(sense_result)

    def get_action(self):
        current_turn = self.tree[0].env.board.fullmove_number
        if current_turn < 3:
            if self.color.color:
                best_action = chess.Move.from_uci(white_first_four[self.is_attack][current_turn-1])
            else:
                best_action = chess.Move.from_uci(black_first_four[self.is_attack][current_turn-1])
        else:
            result = self._mc_sample()
            # when self.tree is empty, result is empty.
            if len(result) == 0:
                best_action = chess.Move.null()
            else:
            #word_counts = Counter(result)
                best_action_prob = max(result.items(), key=lambda x:x[1])
                best_action = best_action_prob[0]

            self.all_move_num += 1
            if best_action == self.right_move:
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
            _, t, c = node.env.move(request_move)
            if t == taken_move and c == capture:
                enemy_king_square = node.env.board.king(not node.env.turn)
                if enemy_king_square is not None:
                    node.env.end_turn()
                    right.append(node)
        #logging.debug("After CUT, left len {}".format(len(right)))
        self.tree = right
        self._normalize()
        self._clean()
