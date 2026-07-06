# gui.py
import tkinter as tk
from game_model import FreeCellGame, Card, Move, MoveType
from tkinter import messagebox


CARD_WIDTH = 60
CARD_HEIGHT = 80
SPACING_X = 80
SPACING_Y = 30
START_Y = 100
BG_COLOR = "#357960"

class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("空当接龙 FreeCell")

        # 主体框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        # 游戏画布
        self.canvas = tk.Canvas(self.main_frame, width=1000, height=600, bg=BG_COLOR)
        self.canvas.pack()

        # 保持按钮始终在底部显示
        self.restart_button = tk.Button(self.root, text="重开一局", command=self.restart_game)
        self.restart_button.pack(pady=10)

        # 初始化游戏逻辑和界面状态
        self.game = FreeCellGame()
        self.card_widgets = {}  # Card对象 → Canvas对象
        self.selected_card = None
        self.start_pos = None
        self.victory = False  # 是否胜利标志
        self.history = []

        # 绑定鼠标和键盘事件
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind_all("<Control-z>", self.undo)

        self.render()




    def render(self):
        self.canvas.delete("all")
        self.card_widgets.clear()

        CARD_SPACING_Y = 27  # 每张牌垂直错开距离（建议 20~35 之间）

        # 空当 & home 显示
        for i in range(4):
            x = 20 + i * SPACING_X
            self.draw_slot(x, 20, f"F{i}", self.game.free_cells[i])

        for i, suit in enumerate(self.game.home_cells):
            x = 400 + i * SPACING_X
            cards = self.game.home_cells[suit]
            top = cards[-1] if cards else None
            self.draw_slot(x, 20, suit.value, top)

        # 列显示：错位显示每张牌
        for col_idx, column in enumerate(self.game.columns):
            x = 20 + col_idx * SPACING_X
            for row_idx, card in enumerate(column):
                y = START_Y + row_idx * CARD_SPACING_Y
                self.draw_card(card, x, y)


    def draw_slot(self, x, y, label, card: Card):
        self.canvas.create_rectangle(x, y, x + CARD_WIDTH, y + CARD_HEIGHT, outline="white")
        self.canvas.create_text(x + 30, y - 10, text=label, fill="white")
        if card:
            self.draw_card(card, x, y)

    def draw_card(self, card: Card, x, y):
        # 背景矩形（牌）
        rect = self.canvas.create_rectangle(
            x, y, x + CARD_WIDTH, y + CARD_HEIGHT,
            fill="white" #注释掉后面，改为全用白底  if card.color() == "red" else "black"
        )

        # 文本显示在牌的顶部，稍微靠右
        text = self.canvas.create_text(
            x + 10, y + 12,
            text=str(card),
            anchor="nw",  # 左上角对齐
            fill="#C41E3A" if card.color() == "red" else "black",
            font=("Arial", 12, "bold")
        )

        self.card_widgets[card] = (rect, text)


    def on_click(self, event):
        self.selected_card = None
        self.selected_sequence = None
        self.selected_col_idx = None
        self.start_pos = None

        # 检查列牌
        for col_idx, column in enumerate(self.game.columns):
            x = 20 + col_idx * SPACING_X
            for row_idx in reversed(range(len(column))):  # 🔁 从底部向上遍历
                card = column[row_idx]
                y = START_Y + row_idx * 27
                if x <= event.x <= x + CARD_WIDTH and y <= event.y <= y + CARD_HEIGHT:
                    sequence = self.game.get_valid_sequence_from_column(col_idx, row_idx)

                    # 情况 1：该牌是列的最后一张（单牌），允许拖动
                    if row_idx == len(column) - 1:
                        self.selected_card = card
                        self.selected_sequence = [card]
                        self.selected_col_idx = col_idx
                        self.start_pos = (event.x, event.y)
                        rect, text = self.card_widgets[card]
                        self.canvas.tag_raise(rect)
                        self.canvas.tag_raise(text)
                        return

                    # 情况 2：该牌是合法连续序列起点，且长度 ≥ 2，允许整列拖动
                    if sequence and sequence[0] == card and len(sequence) >= 2:
                        self.selected_card = card
                        self.selected_sequence = sequence
                        self.selected_col_idx = col_idx
                        self.start_pos = (event.x, event.y)
                        for c in sequence:
                            rect, text = self.card_widgets[c]
                            self.canvas.tag_raise(rect)
                            self.canvas.tag_raise(text)
                        return

        # 空当位点击（保持原样）
        for i, card in enumerate(self.game.free_cells):
            if card is not None:
                rect, text = self.card_widgets[card]
                coords = self.canvas.coords(rect)
                x1, y1, x2, y2 = coords
                if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                    self.selected_card = card
                    self.selected_sequence = [card]
                    self.start_pos = (event.x, event.y)
                    return


    def on_drag(self, event):
        if not self.selected_sequence or not self.start_pos:
            return
        dx = event.x - self.start_pos[0]
        dy = event.y - self.start_pos[1]
        for card in self.selected_sequence:
            rect, text = self.card_widgets[card]
            self.canvas.move(rect, dx, dy)
            self.canvas.move(text, dx, dy)
        self.start_pos = (event.x, event.y)


    def get_max_movable_sequence_length(self):
        return self.game.max_movable_sequence_length()

    def apply_gui_move(self, move: Move) -> bool:
        snapshot = self.game.clone()
        if self.game.apply_move(move):
            self.history.append(snapshot)
            return True
        return False


    def on_release(self, event):
        if not self.selected_card or not self.selected_sequence:
            return

        drop_x, drop_y = event.x, event.y
        moved = False

        from_col_idx = self.selected_col_idx
        from_free_idx = self.find_card_in_free_cells(self.selected_card)
        to_col_idx = self.get_column_index_at(drop_x, drop_y)

        if from_col_idx is not None:
            if to_col_idx is not None:
                moved = self.apply_gui_move(
                    Move(
                        MoveType.COL_TO_COL,
                        from_col_idx,
                        to_col_idx,
                        count=len(self.selected_sequence),
                    )
                )

            if not moved and len(self.selected_sequence) == 1:
                free_idx = self.get_free_cell_index_at(drop_x, drop_y)
                if free_idx is not None:
                    moved = self.apply_gui_move(Move(MoveType.COL_TO_FREE, from_col_idx, free_idx))

            if not moved and len(self.selected_sequence) == 1 and self.is_home_cell_area(drop_x, drop_y):
                moved = self.apply_gui_move(Move(MoveType.COL_TO_HOME, from_col_idx))

        elif from_free_idx is not None and len(self.selected_sequence) == 1:
            if to_col_idx is not None:
                moved = self.apply_gui_move(Move(MoveType.FREE_TO_COL, from_free_idx, to_col_idx))

            if not moved and self.is_home_cell_area(drop_x, drop_y):
                moved = self.apply_gui_move(Move(MoveType.FREE_TO_HOME, from_free_idx))

        self.selected_card = None
        self.selected_sequence = None
        self.render()
        self.check_victory()


    def find_card_column(self, card: Card) -> int:
        for col_idx, col in enumerate(self.game.columns):
            if col and col[-1] == card:
                return col_idx
        return -1

    def get_column_index_at(self, x, y) -> int:
        # 判断是否落在任一列区域上
        for col_idx in range(8):
            col_x = 20 + col_idx * SPACING_X
            if col_x <= x <= col_x + CARD_WIDTH and y >= START_Y:
                return col_idx
        return None

    def get_free_cell_index_at(self, x, y) -> int:
        for i in range(4):
            fx = 20 + i * SPACING_X
            if fx <= x <= fx + CARD_WIDTH and 20 <= y <= 20 + CARD_HEIGHT:
                return i
        return None

    def is_home_cell_area(self, x, y) -> bool:
        return 400 <= x <= 400 + 4 * SPACING_X and 20 <= y <= 20 + CARD_HEIGHT

    def find_card_in_free_cells(self, card: Card) -> int:
        for i, c in enumerate(self.game.free_cells):
            if c == card:
                return i
        return None

    def auto_move_to_home(self):
        self.game.auto_move_to_home()
        self.render()

    def on_right_click(self, event):
        # 判断是否真的会发生归堆再记录历史
        snapshot = self.game.clone()
        self.auto_move_to_home()
        if snapshot != self.game:
            self.history.append(snapshot)
        self.check_victory()



    def check_victory(self):
        if self.game.is_won():
            self.victory = True
            self.canvas.create_text(
                500, 300,
                text="🎉 胜利！你赢了！ 🎉",
                fill="yellow",
                font=("Arial", 24, "bold"),
                tags="victory_text"
            )


    def restart_game(self):
        if not self.victory:
            # 弹出确认对话框
            confirm = messagebox.askyesno("确认", "是否要开启一局新游戏？")
            if not confirm:
                return  # 用户选择否，取消重开

        self.victory = False  # 重置胜利状态
        self.history.clear()
        self.game = FreeCellGame()
        self.selected_card = None
        self.render()


    def undo(self, event=None):
        if self.history:
            self.game = self.history.pop()
            self.selected_card = None
            self.render()
        else:
            print("⚠️ 没有可撤销的历史操作")




if __name__ == "__main__":
    root = tk.Tk()
    app = GameGUI(root)
    root.mainloop()
