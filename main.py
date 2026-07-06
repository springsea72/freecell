import sys

from game_model import FreeCellGame


def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

def main():
    configure_console_encoding()
    game = FreeCellGame()

    while True:
        print("\n======= 当前牌局状态 =======")
        game.display()

        moves = game.generate_legal_moves()
        if not moves:
            print("没有任何合法动作，游戏结束！")
            break

        print("\n可选动作：")
        for idx, move in enumerate(moves):
            print(f"{idx}: {move}")

        try:
            cmd = input("\n请输入动作编号（或输入 q 退出）：")
            if cmd.lower() == 'q':
                print("退出游戏。")
                break

            move_idx = int(cmd)
            if move_idx < 0 or move_idx >= len(moves):
                print("无效编号，请重新输入。")
                continue

            move = moves[move_idx]
            success = game.apply_move(move)
            if not success:
                print("执行失败，可能是非法状态。")

        except Exception as e:
            print("发生错误：", e)

if __name__ == "__main__":
    main()
