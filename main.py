from game_model import FreeCellGame, MoveType, Move

def main():
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
            success = apply_move(game, move)
            if not success:
                print("执行失败，可能是非法状态。")

        except Exception as e:
            print("发生错误：", e)


def apply_move(game: FreeCellGame, move: Move) -> bool:
    t = move.move_type
    f = move.from_idx
    t_ = move.to_idx

    if t == MoveType.COL_TO_COL:
        return game.move_card_between_columns(f, t_)
    elif t == MoveType.COL_TO_FREE:
        return game.move_card_to_free_cell(f)
    elif t == MoveType.FREE_TO_COL:
        return game.move_card_from_free_cell(f, t_)
    elif t == MoveType.COL_TO_HOME:
        return game.move_card_to_home(f)
    return False


if __name__ == "__main__":
    main()
