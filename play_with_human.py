"""
play with human local
"""

import datetime

import chess

from reconchess import LocalGame, Optional, Square, List, Tuple, Player, Color, WinReason, GameHistory, \
    play_sense, play_move
from reconchess.scripts.rc_play import UIPlayer


class LocalChessGame(LocalGame):
    def __init__(self):
        super().__init__()

    def sense(self, square: Optional[Square]) -> List[Tuple[Square, Optional[chess.Piece]]]:
        if self._is_finished:
            return []

        if square is None:
            # don't sense anything
            sense_result = []
        else:
            sense_result = []
            for i in range(64):
                sense_result.append((i, self.board.piece_at(i)))
            # print(self.board)

        return sense_result


class ChessUIPlayer(UIPlayer):
    def __init__(self):
        super().__init__()

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        self.window.perspective = chess.WHITE
        return 0


def play_local_chess_game(white_player: Player, black_player: Player, game: LocalGame = None,
                          seconds_per_player: float = 900) -> Tuple[Optional[Color], Optional[WinReason], GameHistory]:
    players = [black_player, white_player]

    if game is None:
        game = LocalGame(seconds_per_player=seconds_per_player)

    white_name = white_player.__class__.__name__
    black_name = black_player.__class__.__name__
    game.store_players(white_name, black_name)

    white_player.handle_game_start(chess.WHITE, game.board.copy(), black_name)
    black_player.handle_game_start(chess.BLACK, game.board.copy(), white_name)
    game.start()

    while not game.is_over():
        player = players[game.turn]
        sense_actions = game.sense_actions()
        move_actions = game.move_actions()

        play_sense(game, player, sense_actions, move_actions)

        play_move(game, player, move_actions)

    game.end()
    winner_color = game.get_winner_color()
    win_reason = game.get_win_reason()
    game_history = game.get_game_history()

    white_player.handle_game_end(winner_color, win_reason, game_history)
    black_player.handle_game_end(winner_color, win_reason, game_history)

    return winner_color, win_reason, game_history


def main():
    white_player, black_player = ChessUIPlayer, ChessUIPlayer
    # play game
    winner_color, win_reason, game_history = play_local_chess_game(white_player(), black_player(), LocalChessGame())

    # game result
    winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    game_history.save('./history/{}_{}.json'.format(timestamp, winner))


if __name__ == '__main__':
    main()
