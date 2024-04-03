# 代码结构
- WIEUC
    - example_agent 示例代理
        - attacker_bot_2 增加了套路的攻击者
        - example_bot 增加了文字描述的示例
        - 其他为原生示例代理
    - reconchess 侦查盲棋源代码
    - my_bot 我方智能体
        - MHSIEC 初始版本，扩展节点阶段用了多进程加速
        - IEUC 初始版本，扩展节点阶段速度稍慢
        - WIEUC_player1 基于MHSIEC的加权信息熵版本
        - WIEUC_player2 基于IEUC的加权信息熵版本
    - strangefish 智能体strangefish源码
    - play_with_server.py 和服务器对打的辅助py
        - usage
        ```shell script
        usage: play_with_server.py [-h] [--color {white,black,random}]
                           [--local-bot LOCAL_BOT] [--server-url SERVER_URL]
                           [--server-bot SERVER_BOT] [--username USERNAME]
                           [--password PASSWORD]
        ```
    - play_with_server.py 和真人对打的辅助py
    - record.csv 记录运行时的一些数据，包括每回合的树节点数量、信息熵、侦查所用时间
    - vpdb.py 生成.vscode/launch.json运行配置
    - Linux下运行，可能需要杀掉相关僵尸进程，保证和服务器智能体正常对局
        ```
        $ ps -ef | grep python | grep yghuang
        $ pkill -9 -u yghuang python
        $ pgrep -u yghuang rc-connect | xargs kill -9
        ```
   