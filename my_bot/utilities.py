from configparser import ConfigParser
from math import sqrt

import chess
import numpy as np
# import torch

from reconchess import LocalGame

from reconchess.utilities import is_psuedo_legal_castle, slide_move, is_illegal_castle

"""
初始棋盘
            r n b q k b n r
            p p p p p p p p
            . . . . . . . .
            . . . . . . . .
            . . . . . . . .
            . . . . . . . .
            P P P P P P P P
            R N B Q K B N R
"""
"""################################ DEFINITIONS ######################################"""

chessboard = {'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5, 'p': 6,
              'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12,
              '.': 0}
PIECE_TYPE = 6
PIECE_CAMP = 2
PIECE_NONE = 1
COLORS = [WHITE, BLACK] = [True, False]
DATA_PATH = '../data/'
MODEL_PATH = '../model/'
SENSE_SQUARE_B = [9, 12, 15, 33, 36, 39, 57, 60, 63]
SENSE_SQUARE_XY_B = [(1, 1), (1, 4), (1, 7), (4, 1), (4, 4), (4, 7), (7, 1), (7, 4), (7, 7)]
SENSE_SQUARE_W = [25, 28, 31, 49, 52, 55, 1, 4, 7]
SENSE_SQUARE_XY_W = [(3, 1), (3, 4), (3, 7), (6, 1), (6, 4), (6, 7), (0, 1), (0, 4), (0, 7)]
"""################################# USEFUL FUNCTION #################################"""


# 模拟一个移动行为，得到最终的移动
def simulate_move(board, move:chess.Move):
    if move == chess.Move.null():
        return None
    
    # 合法的移动就不改变了
    if move in board.generate_pseudo_legal_moves() or is_psuedo_legal_castle(board, move):
        return move
    
    # 不合法的王车易位，返回none
    if is_illegal_castle(board, move):
        return None
    
    # 如果是平移，移到能达到的最远位置
    piece = board.piece_at(move.from_square)
    if piece.piece_type in [chess.PAWN, chess.ROOK, chess.BISHOP, chess.QUEEN]:
        move = slide_move(board, move)
    
    return move if move in board.generate_pseudo_legal_moves() else None


# 返回感知九宫格
def get_sense_pieces(board: chess.Board, x, y):

    sense = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            sense.append(board.piece_at((x + i) * 8 + (y + j)))
    
    return tuple(sense)

def convert_chessboard_array(board: chess.Board):
    """
    convert chessboard to a number array, using chessboard define before
    :param board:
    :return:
    """
    c = [i.split(' ') for i in str(board).split('\n')]
    for i, a in enumerate(c):
        for j, b in enumerate(a):
            c[i][j] = chessboard[b]
    return c


def convert_chessboard_numpy(boards: list, cur_player: bool):
    """
    convert chessboard to numpy array, using one-hot to encode the feature
    layer 0:None
    layer 1-6: chess piece type
    layer 7-8: encode the color(our, op)
    other: to be continue
    from the alphaGo feature
    :param cur_player:
    :param boards: board list
    :return:
    """
    batch = len(boards)
    c_in = PIECE_CAMP + PIECE_NONE + PIECE_TYPE
    res = np.zeros((batch, c_in, 8, 8))
    for b in range(batch):
        for i in range(8):
            for j in range(8):
                piece = boards[b].piece_at(i * 8 + j)
                if piece is not None:
                    res[b][piece.piece_type][i][j] = 1
                    if piece.color == cur_player:
                        res[b][7][i][j] = 1
                    else:
                        res[b][8][i][j] = 1
                else:
                    res[b][0][i][j] = 1
    return res


def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                           [(l1, t) for t in range(8)] + \
                           [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                           [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array


def get_action_moves():
    label_array = create_uci_labels()
    all_moves_array = []
    for i in range(len(label_array)):
        all_moves_array.append(chess.Move.from_uci(label_array[i]))
    all_moves_array.append(None)
    return all_moves_array


def convert_action_to_index(move: chess.Move, all_moves_array):
    """
    Convert action to a index
    把action映射为policy向量中的index, label数量1968个，policy1968维
    player_chess里面的calc_policy 中定义policy = np.zeros(self.labels_n)， label_n维，Label_n是Label的数量
    config.py里面的create_uci_labels定义label
    """
    """
    :param move:chess.Move,all_moves_array是所有动作的列表，元素类型是chess.Move，由labels_array元素uci string转化而来
    :return:action_index
    """
    action_index = all_moves_array.index(move)
    return action_index


def create_policy_vector(move: chess.Move, all_moves_array):
    # 采取的动作转化成policy向量，policy向量1968维
    index = convert_action_to_index(move, all_moves_array)
    policy = np.zeros(len(all_moves_array))
    policy[index] = 0.8
    for i in range(len(all_moves_array)):
        if policy[i] == 0:
            policy[i] = (1 - policy[index]) / (len(all_moves_array) - 1)
    return policy


# def save_model(model, path):
#     torch.save(model, path)


# def load_model(path):
#     return torch.load(path)


# def save_model_parameter(model, path):
#     torch.save(model.state_dict(), path)


# def load_model_parameter(model, path):
#     model.load_state_dict(torch.load(path))
#     return model


def get_nearest_square(s, color, old, round_):
    if color:
        sense_square = SENSE_SQUARE_W
        sense_square_xy = SENSE_SQUARE_XY_W
    else:
        sense_square = SENSE_SQUARE_B
        sense_square_xy = SENSE_SQUARE_XY_B

    if round_ < 5:
        sense_square = sense_square[:6]
        sense_square_xy = sense_square_xy[:6]

    r = s // 8
    c = s % 8

    def dis(a, b):
        return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

    distance = [dis((r, c), i) for i in sense_square_xy]
    sense = sense_square[distance.index(min(distance))]
    if sense == old:
        distance.remove(min(distance))
        return sense_square[distance.index(min(distance))]
    else:
        return sense


"""################################# CONFIG #################################"""


class Config:
    """
    Config 基类，只要继承此类，在init中定义好变量名称，可以实现自动的从文件中
    读取变量填充，以及自动写入
    """

    def __init__(self):
        self.section = "config"
        # the total actions that the game have
        # it's used for both net and train, so put it here
        # moves
        self.actions = get_action_moves()
        self.n_actions = len(self.actions)

    def read(self, path):
        """
        实现从配置文件中找到所有变量，并赋值
        :param path:
        :return:
        """
        cfg = ConfigParser()
        cfg.read(path)
        members = \
            [attr for attr in dir(self)
             if not callable(getattr(self, attr)) and not attr.startswith("__")]
        members.remove('section')
        for m in members:
            if cfg.has_option(self.section, m):
                setattr(self, m, cfg.get(self.section, m))

    def write(self, path):
        """
        读取类中定义的变量，将其保存
        :param path:
        :return:
        """
        cfg = ConfigParser()
        cfg.add_section(self.section)
        # get all members of the class, which not function and built-in
        members = \
            [attr for attr in dir(self)
             if not callable(getattr(self, attr)) and not attr.startswith("__")]
        # remove the section
        members.remove('section')
        for m in members:
            cfg.set(self.section, m, str(getattr(self, m)))
        cfg.write(open(path, 'w'))


class NetConfig(Config):
    def __init__(self):
        super().__init__()
        # net config
        self.section = 'net'
        # the number of feature
        self.in_channel = 19
        # board_size, should not change
        self.board_size = 8
        # the filter number used for the first layer
        self.con_filter_number = 256
        # the kernel size of first layer
        self.con_kernel_size = 3
        # layers of residual blocks
        self.res_number = 7
        # residual blocks filter number
        self.res_con_filter_number = 256
        # residual blocks kernel size
        self.res_con_kernel_size = 3
        # the fc size of the first fc layer of value head
        self.value_fc_size = 256
        # the filter number of the policy head
        self.policy_con_f_num = 2
        # the filter number of the value head
        self.value_con_f_num = 1


class TrainConfig(Config):
    def __init__(self):
        super().__init__()
        # train section
        self.section = 'train'
        # using to calc the u value
        self.c_puct = 1.5
        # to add dirichlet noise,
        # (1- noise_eps)p + noise_eps * dirichlet(noise_alpha)
        self.noise_eps = 0.25
        self.noise_alpha = 0.3
        # how much threads a mcts controller should use in search
        self.search_threads = 16
        # total threads
        self.simulation_num_per_move = 400
        # virtual loss to avoid search the same branch many times
        self.virtual_loss = 3
        # the temperature used for exploration decrease rate
        self.tau_decay_rate = 0.99
        # the number of data to collect in a training epoch
        self.collect_rounds = 200
        # the batch_size used to train
        self.batch_size = 200
        # collect_processing
        self.collect_processing = 6
        # is_alpha_go
        self.is_alpha_go = True


class MCConfig(Config):
    def __init__(self):
        super().__init__()
        # 每次mc执行多少次
        self.threads_per_mc = 200
        # 执行多少次mc
        self.mc_samples = 250
        self.c_puct = 5
        # sense时的总次数限制
        self.limited_sense_rounds = 2500
        # sense的每次限制
        self.limited_sense_per = 20


class Color:
    def __init__(self, color: bool):
        self.color = color

    def __str__(self):
        if self.color:
            return 'WHITE'
        else:
            return 'BLACK'

    def flip(self):
        if self.color:
            self.color = BLACK
        else:
            self.color = WHITE
        return self


class Mission:
    def __init__(self, env_list, exclude_action_list, n_threads, color: Color):
        self.color = color
        self.n_threads = n_threads
        self.env_list = env_list
        self.ex_ac_list = exclude_action_list

    def number(self):
        assert len(self.env_list) == len(self.ex_ac_list)
        return len(self.env_list)
