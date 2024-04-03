"""
based on reconchess.scripts.rc_play_on_server
增加将自己的代理连接到服务器进行对战的功能
"""

import argparse
import datetime
import random

import chess

from reconchess import play_remote_game
from reconchess.scripts.rc_connect import RBCServer, ask_for_username, ask_for_password
from reconchess.scripts.rc_play import UIPlayer, load_player
# import sys,os
# sys.path.append(os.getcwd())

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # color
    parser.add_argument('--color', default='random', choices=['white', 'black', 'random'],
                        help='The color you want to play as.')
    # the local bot and server bot
    parser.add_argument('--local-bot', default='C:\\Users\\18399\\Desktop\\stockfish\\EUBOT\\WIEUC\\my_bot\\IEUC_player.py', help='bot path which connect to server')
    parser.add_argument('--server-url', default='https://rbc.jhuapl.edu', help='URL of the server.')
    parser.add_argument('--server-bot', default=None, help='server bot name to connected')
    # input user and password
    parser.add_argument('--username', default="MY_wubot", help='Username for login. Enter with prompt if not specified.')
    parser.add_argument('--password', default="225516459wdv..", help='Password for login. Enter with prompt if not specified.')
    # parse
    args = parser.parse_args()

    # get auth
    username = ask_for_username() if args.username is None else args.username
    password = ask_for_password() if args.password is None else args.password
    auth = username, password

    # server and get active user
    server = RBCServer(args.server_url, auth)
    user_names = server.get_active_users()
    if auth[0] in user_names:
        user_names.remove(auth[0])

    if len(user_names) == 0:
        print('No active users.')
        quit()

    # choose server bot
    a = len(user_names)
    if args.server_bot is None:
        for j, username in enumerate(user_names):
            print('[{}] {}'.format(j, username))
        i = int(input('Choose opponent: '))
    else:
        i = -1
        for j, username in enumerate(user_names):
            if username == args.server_bot:
                i = j
    if i < 0 or i >= a:
        print('wrong server bot, it\'s not running')
        quit()

    # choose color
    color = chess.WHITE
    if args.color == 'black' or (args.color == 'random' and random.uniform(0, 1) < 0.5):
        color = chess.BLACK

    # send invitation to create game
    game_id = server.send_invitation(user_names[i], color)

    # choose player, if it's given, load it. else load UIPlayer
    if args.local_bot is None:
        player = UIPlayer
    else:
        _, player = load_player(args.local_bot)

    # play game
    winner_color, win_reason, game_history = play_remote_game(args.server_url, game_id, auth, player())

    # game result
    winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    game_history.save('{}_{}_{}.json'.format(timestamp, game_id, winner))


if __name__ == '__main__':
    main()
