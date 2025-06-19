# gui.py
import tkinter as tk
from game_model import FreeCellGame, Card, Move, MoveType

CARD_WIDTH = 60
CARD_HEIGHT = 80
SPACING_X = 80
SPACING_Y = 30
START_Y = 100
BG_COLOR = "#357960"

class GameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("空当接龙 FreeCell AI")
        self.canvas = tk.Canvas(root, width=1000, height=600, bg=BG_COLOR)
        self.canvas.pack()

        self.game = FreeCellGame()
        self.card_widgets = {}  # Card对象 → Canvas对象
        self.selected_card = None
        self.start_pos = None

        self.render()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.restart_button = tk.Button(self.root, text="重新开始", command=self.restart_game)
        self.restart_button.pack(pady=10)

        self.history = []
        # 绑定 Ctrl+Z
        self.root.bind_all("<Control-z>", self.undo)



    def render(self):
        self.canvas.delete("all")
        self.card_widgets.clear()

        CARD_SPACING_Y = 25  # 每张牌垂直错开距离（建议 20~35 之间）

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
            fill="white" if card.color() == "red" else "black"
        )

        # 文本显示在牌的顶部，稍微靠右
        text = self.canvas.create_text(
            x + 10, y + 12,
            text=str(card),
            anchor="nw",  # 左上角对齐
            fill="black" if card.color() == "red" else "white",
            font=("Arial", 12, "bold")
        )

        self.card_widgets[card] = (rect, text)

    def on_click(self, event):
        # 检查列中最上方的牌是否被点击
        for col_idx, column in enumerate(self.game.columns):
            if not column:
                continue
            top_card = column[-1]
            rect, text = self.card_widgets[top_card]
            coords = self.canvas.coords(rect)
            x1, y1, x2, y2 = coords
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.selected_card = top_card
                self.start_pos = (event.x, event.y)
                self.canvas.tag_raise(rect)
                self.canvas.tag_raise(text)
                return  # 点击成功，立即返回

        # 检查空当位中的牌是否被点击
        for i, card in enumerate(self.game.free_cells):
            if card is not None:
                rect, text = self.card_widgets[card]
                coords = self.canvas.coords(rect)
                x1, y1, x2, y2 = coords
                if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                    self.selected_card = card
                    self.start_pos = (event.x, event.y)
                    self.canvas.tag_raise(rect)
                    self.canvas.tag_raise(text)
                    return  # 点击成功，立即返回

        # 如果不是列牌，也检查空当位是否选中
        for i, card in enumerate(self.game.free_cells):
            if card is not None:
                fx = 20 + i * SPACING_X
                fy = 20
                if fx <= event.x <= fx + CARD_WIDTH and fy <= event.y <= fy + CARD_HEIGHT:
                    self.selected_card = card
                    self.start_pos = (event.x, event.y)
                    return

    def on_drag(self, event):
        if not self.selected_card:
            return
        dx = event.x - self.start_pos[0]
        dy = event.y - self.start_pos[1]
        rect, text = self.card_widgets[self.selected_card]
        self.canvas.move(rect, dx, dy)
        self.canvas.move(text, dx, dy)
        self.start_pos = (event.x, event.y)

    def on_release(self, event):
        if self.selected_card:
            self.history.append(self.game.clone())

        if not self.selected_card:
            return

        drop_x, drop_y = event.x, event.y
        moved = False

        # 判断是否从列来
        from_col_idx = self.find_card_column(self.selected_card)
        if from_col_idx != -1:
            # 1. 列 → 列
            to_col_idx = self.get_column_index_at(drop_x, drop_y)
            if to_col_idx is not None and to_col_idx != from_col_idx:
                if self.game.can_move_to_column(self.selected_card, self.game.columns[to_col_idx]):
                    self.game.columns[to_col_idx].append(self.game.columns[from_col_idx].pop())
                    moved = True

            # 2. 列 → 空当
            if not moved:
                free_idx = self.get_free_cell_index_at(drop_x, drop_y)
                if free_idx is not None and self.game.free_cells[free_idx] is None:
                    self.game.free_cells[free_idx] = self.game.columns[from_col_idx].pop()
                    moved = True

            # 3. 列 → home
            if not moved:
                if self.is_home_cell_area(drop_x, drop_y):
                    if self.game.can_move_to_home(self.selected_card):
                        self.game.home_cells[self.selected_card.suit].append(self.game.columns[from_col_idx].pop())
                        moved = True

        # ✅ 4. 空当 → 列（新增）
        if not moved:
            from_free_idx = self.find_card_in_free_cells(self.selected_card)
            to_col_idx = self.get_column_index_at(drop_x, drop_y)
            if from_free_idx is not None and to_col_idx is not None:
                if self.game.can_move_to_column(self.selected_card, self.game.columns[to_col_idx]):
                    self.game.columns[to_col_idx].append(self.selected_card)
                    self.game.free_cells[from_free_idx] = None
                    moved = True

        # 最后刷新界面
        self.selected_card = None
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
        moved = True
        while moved:
            moved = False

            # 检查每列的顶部牌
            for i, column in enumerate(self.game.columns):
                if column:
                    top_card = column[-1]
                    if self.game.can_move_to_home(top_card):
                        self.game.home_cells[top_card.suit].append(column.pop())
                        moved = True

            # 检查空当位的牌
            for i, card in enumerate(self.game.free_cells):
                if card and self.game.can_move_to_home(card):
                    self.game.home_cells[card.suit].append(card)
                    self.game.free_cells[i] = None
                    moved = True

        self.render()

    def on_right_click(self, event):
        self.history.append(self.game.clone())

        self.auto_move_to_home()
        self.check_victory()


    def check_victory(self):
        total_home_cards = sum(len(stack) for stack in self.game.home_cells.values())
        if total_home_cards == 52:
            self.canvas.create_text(
                500, 300,
                text="🎉 胜利！你赢了！ 🎉",
                fill="yellow",
                font=("Arial", 32, "bold")
            )

    def restart_game(self):
        self.history.clear()  # 重开时清空历史

        self.game = FreeCellGame()  # 重新初始化游戏
        self.selected_card = None   # 清除当前拖动状态
        self.render()               # 重绘界面

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
