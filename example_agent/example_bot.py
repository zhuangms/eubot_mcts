"""
要实现的bot的说明，整合了要自己实现的部分的描述
以及对应的规则信息
"""
from typing import Optional, List, Tuple

import chess
import sys

sys.path.append("E:\\PROJECTS\\python\\reconchess_project\\")
from reconchess import Player, Color, WinReason, GameHistory, Square
from train import convert_chessboard_array


class ExampleBot(Player):
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        """
        在游戏开始时，会调用此函数
        :param color: 本代理的颜色，白色为True，黑色为False
        :param board: 初始的棋盘情况
            r n b q k b n r
            p p p p p p p p
            . . . . . . . .
            . . . . . . . .
            . . . . . . . .
            . . . . . . . .
            P P P P P P P P
            R N B Q K B N R
        :param opponent_name:对手的名字，str
        :return:
        """
        convert_chessboard_array(board)

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        """
        capture 吃子 piece 棋子
        自己回合开始阶段，系统告知上一局对方是否captured我方的棋子，如果有，给出对应的square
        :param captured_my_piece: 是否被吃
        :param capture_square: 被吃掉的piece所处的square
        :return:
        """
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        """
        自己回合第二阶段，感知选取
        :param sense_actions: 给出所有当前能选的感知选项，一般为所有棋盘上的区域，貌似不变
        :param move_actions: 所有可以选择的动作，此数组和下一步choose move收到的相同
        :param seconds_left: 时间剩余
        :return:
        """
        pass

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        """
        获取感知结果 以选择的id为中心，3*3的一个窗口
        :param sense_result: 结果为一个列表，为可以感知的返回值，对应的Square ID，以及其中对应的棋子
            例子：[(60, None), (61, None), (62, None), (52, Piece.from_symbol('N')),
                  (53, None), (54, None), (44, None), (45, None), (46, None)]
                  如果选择的是边缘，那么返回的可能不是3*3
        :return:
        """
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        """
        选择动作阶段
        :param move_actions: 所有可选的动作。来自game
            和上面的感知阶段返回的结果相同，因此可能并不完全，没有包含感知信息。
        :param seconds_left: 剩余的时间
        :return:
        """
        print(len(move_actions))
        print(move_actions)

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        """
        接受移动的结果信息，move的规则如下:
        Move phase: the player chooses any chess move, or chooses to “pass.”
        If the move is a pawn promotion and the player does not specify a piece to promote to,
        then a queen promotion is assumed.
        国际象棋中兵到达敌方的底线时，会进行升变，可以变为己方的后、马、象或车，在没有指定的情况下，默认为后
        Then, given that move, one of three conditions holds:

            i)   The move is legal on the game board. 合法动作
            ii)  The moving piece is a queen, bishop, or rook and the move is illegal on the game board
                 because one or more opponent pieces block the path of the moving piece. Then, the move
                 is modified so that the destination square is the location of the first obstructing opponent
                 piece, and that opponent piece is captured. (Note: this rule does not apply to a castling king).
                 如果移动的棋子是女王，主教或车，并且移动是非法的，因为一个或多个对手棋子阻挡了移动棋子的路径。
                 那么移动被修改，使得目的地方块是第一个阻挡对手棋子的位置，并且该对手棋子被吃。也即改为吃掉对方第一个
                 挡路的棋子
            iii) The moving piece is a pawn, moving two squares forward on that pawn’s first move, and the move
                 is illegal because an opponent’s piece blocks the path of the pawn. Then, the move is modified
                 so that the pawn moves only one square if that modified move is legal, otherwise the player’s
                 move is illegal.
                 移动的棋子是一个小兵的时候，在棋子的移动中选择向前移动两个方格，此时是非法的，因为对手的棋子阻挡了棋子的
                 路径。 那么动作会被修改为仅仅移动一格，如果此时是合法的，那么就执行，否则玩家的移动是非法的。
            iv） If any of (i)-(iii) do not hold, the move is considered illegal (or the player chose to pass
                 which has the same result). 除上面之外的所有情况都被任务是不合法的

        The results of the move are then determined: if condition (iv) holds, then no changes are made to the
        board, the player is notified that their move choice was illegal (or the pass is acknowledged), and the
        player’s turn is over. Otherwise the move is made on the game board. If the move was modified because of
        (ii) or (iii), then the modified move is made, and the current player is notified of the modified move in
        the move results. If the move results in a capture, the current player is notified that they captured a
        piece and which square the capture occurred, but not the type of opponent piece captured (the opponent will
        be notified of the capture square on their turn start phase). If the move captures the opponent’s king,
        the game ends and both players are notified. The current player’s turn is now over and play proceeds to
        the opponent.
        之后就会确认动作的结果：
        a.  如果条件（iv）成立，则不对棋盘进行任何更改，通知玩家他们的选择是非法的（或传递被确认），并且玩家的回合结束。也即
            此次移动结束，但不进行任何修改，此次动作选择报废，游戏继续进行。反之执行合法动作
        b.  如果因为符合i,ii,iii的条件，对玩家的移动进行了修改，那么执行动作，并通知玩家修改后的动作
        c.  如果动作会导致吃子，那么通知玩家吃子发生的位置，但是不会告知对方的棋子的类型。同时对方在其动作回合的起始阶段
            得知这一消息。也即吃子是公共事件，但是仅仅是吃子发生的位置信息，被吃的棋子类型是不完全的。
        d.  一方的王被吃了，游戏结束，双方得知消息

        :param requested_move: 请求的动作
        :param taken_move: 被修改的动作
        :param captured_opponent_piece: 是否吃了对方的子
        :param capture_square: 在哪吃的子
        :return:
        """
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        """
        处理游戏结束事件
        :param winner_color: 赢家
        :param win_reason: 赢得原因
        :param game_history: 游戏历史信息
        :return:
        """
        pass
