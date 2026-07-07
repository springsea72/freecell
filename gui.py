# gui.py
import threading
import tkinter as tk
from game_model import FreeCellGame, Card, Move, MoveType
from solver import solve
from tkinter import messagebox


CARD_WIDTH = 60
CARD_HEIGHT = 80
SPACING_X = 80
SPACING_Y = 30
START_Y = 100
BG_COLOR = "#357960"
GUI_SOLVE_MAX_NODES = 5000
GUI_SOLVE_MAX_DEPTH = 200
PLAYBACK_INTERVAL_MS = 200

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
        self.playback_moves = []
        self.playback_index = 0
        self.playback_running = False
        self.playback_paused = False
        self.solve_running = False
        self.playback_after_id = None
        self.playback_generation = 0
        self.status_var = tk.StringVar(value="空闲")
        self.playback_speed_var = tk.IntVar(value=PLAYBACK_INTERVAL_MS)

        self.playback_controls_frame = tk.Frame(self.root)
        self.playback_controls_frame.pack(pady=4)
        self.solve_button = tk.Button(self.playback_controls_frame, text="求解播放", command=self.solve_and_play)
        self.solve_button.pack(side=tk.LEFT, padx=4)
        self.pause_button = tk.Button(self.playback_controls_frame, text="暂停/继续", command=self.toggle_playback_pause)
        self.pause_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = tk.Button(self.playback_controls_frame, text="停止播放", command=self.stop_playback)
        self.stop_button.pack(side=tk.LEFT, padx=4)
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack()
        self.params_label = tk.Label(
            self.root,
            text=f"max_nodes={GUI_SOLVE_MAX_NODES} max_depth={GUI_SOLVE_MAX_DEPTH}",
        )
        self.params_label.pack()
        self.speed_scale = tk.Scale(
            self.root,
            from_=50,
            to=1000,
            resolution=50,
            orient=tk.HORIZONTAL,
            label="播放间隔(ms)",
            variable=self.playback_speed_var,
        )
        self.speed_scale.pack()
        self.update_playback_controls()

        # 绑定鼠标和键盘事件
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind_all("<Control-z>", self.undo)

        self.render()




    def set_status(self, text):
        if hasattr(self, "status_var"):
            self.status_var.set(text)

    def is_autoplay_active(self):
        return self.solve_running or self.playback_running

    def get_playback_interval(self):
        if not hasattr(self, "playback_speed_var"):
            return PLAYBACK_INTERVAL_MS
        try:
            interval = int(self.playback_speed_var.get())
        except Exception:
            return PLAYBACK_INTERVAL_MS
        return max(50, min(1000, interval))

    def format_solve_result_status(self, result):
        label = "solved" if result.solved else "failed"
        path_length = len(result.moves)
        if result.solved:
            summary = f"已找到解法：{path_length} 步"
        else:
            summary = f"未找到解法：{result.reason}"
        return (
            f"{label}: {summary}，reason={result.reason}，"
            f"explored_nodes={result.explored_nodes}，"
            f"generated_nodes={result.generated_nodes}，"
            f"max_frontier={result.max_frontier}，"
            f"path length={path_length}"
        )

    def block_manual_input_if_autoplay_active(self):
        if not self.is_autoplay_active():
            return False
        self.set_status("自动求解/播放中，手动操作已禁用")
        return True

    def update_playback_controls(self):
        if hasattr(self, "solve_button"):
            self.solve_button.config(state=tk.DISABLED if self.is_autoplay_active() else tk.NORMAL)
        if hasattr(self, "pause_button"):
            self.pause_button.config(state=tk.NORMAL if self.playback_running else tk.DISABLED)
        if hasattr(self, "stop_button"):
            self.stop_button.config(state=tk.NORMAL if self.is_autoplay_active() else tk.DISABLED)

    def solve_and_play(self):
        if self.is_autoplay_active():
            return

        game_snapshot = self.game.clone()
        self.solve_running = True
        self.playback_generation += 1
        token = self.playback_generation
        self.set_status("求解中...")
        self.update_playback_controls()

        thread = threading.Thread(
            target=self._solve_worker,
            args=(game_snapshot, token),
            daemon=True,
        )
        thread.start()

    def _solve_worker(self, game_snapshot, token):
        try:
            result = solve(
                game_snapshot,
                max_nodes=GUI_SOLVE_MAX_NODES,
                max_depth=GUI_SOLVE_MAX_DEPTH,
            )
        except Exception as exc:
            message = str(exc)
            self.root.after(0, lambda: self.on_solve_error(message, token))
            return

        self.root.after(0, lambda: self.on_solve_finished(result, token))

    def on_solve_error(self, message, token):
        if token != self.playback_generation:
            return
        self.solve_running = False
        self.set_status(f"求解失败：{message}")
        self.update_playback_controls()

    def on_solve_finished(self, result, token):
        if token != self.playback_generation:
            return

        self.solve_running = False
        status = self.format_solve_result_status(result)
        if result.solved:
            self.start_playback(result.moves)
            self.set_status(status)
        else:
            self.set_status(status)
            self.update_playback_controls()

    def start_playback(self, moves):
        self.cancel_playback_after()
        self.playback_generation += 1
        self.playback_moves = list(moves)
        self.playback_index = 0
        self.playback_paused = False

        if not self.playback_moves:
            self.playback_running = False
            self.set_status("播放完成")
            self.update_playback_controls()
            return

        self.playback_running = True
        self.set_status(f"播放中：0/{len(self.playback_moves)}")
        self.update_playback_controls()
        self.schedule_next_playback_step(self.playback_generation)

    def schedule_next_playback_step(self, token=None):
        if token is None:
            token = self.playback_generation
        if not self.playback_running or self.playback_paused:
            return
        self.cancel_playback_after()
        self.playback_after_id = self.root.after(
            self.get_playback_interval(),
            lambda: self.playback_step(token),
        )

    def cancel_playback_after(self):
        if self.playback_after_id is None:
            return
        try:
            self.root.after_cancel(self.playback_after_id)
        except Exception:
            pass
        self.playback_after_id = None

    def playback_step(self, token=None):
        if token is None:
            token = self.playback_generation
        self.playback_after_id = None
        if token != self.playback_generation or not self.playback_running or self.playback_paused:
            return

        total = len(self.playback_moves)
        if self.playback_index >= total:
            self.finish_playback()
            return

        move = self.playback_moves[self.playback_index]
        snapshot = self.game.clone()
        if not self.game.apply_move(move):
            self.playback_running = False
            self.playback_paused = False
            self.set_status(f"播放失败：{self.playback_index + 1}/{total}")
            self.update_playback_controls()
            return

        self.history.append(snapshot)
        self.playback_index += 1
        self.render()
        self.check_victory()

        if self.playback_index >= total:
            self.finish_playback()
        else:
            self.set_status(f"播放中：{self.playback_index}/{total}")
            self.schedule_next_playback_step(token)

    def finish_playback(self):
        self.playback_running = False
        self.playback_paused = False
        self.playback_after_id = None
        self.set_status("播放完成")
        self.update_playback_controls()

    def toggle_playback_pause(self):
        if not self.playback_running:
            return
        if self.playback_paused:
            self.playback_paused = False
            self.set_status(f"播放中：{self.playback_index}/{len(self.playback_moves)}")
            self.schedule_next_playback_step(self.playback_generation)
        else:
            self.playback_paused = True
            self.cancel_playback_after()
            self.set_status("已暂停")
        self.update_playback_controls()

    def stop_playback(self):
        if self.playback_after_id is not None:
            self.cancel_playback_after()
        self.playback_generation += 1
        self.solve_running = False
        self.playback_running = False
        self.playback_paused = False
        self.playback_moves = []
        self.playback_index = 0
        self.set_status("空闲")
        self.update_playback_controls()


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
        if self.block_manual_input_if_autoplay_active():
            return

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
        if self.block_manual_input_if_autoplay_active():
            return

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
        if self.block_manual_input_if_autoplay_active():
            return

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
        if self.block_manual_input_if_autoplay_active():
            return

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
        self.stop_playback()
        self.history.clear()
        self.game = FreeCellGame()
        self.selected_card = None
        self.render()


    def undo(self, event=None):
        stopped_autoplay = self.is_autoplay_active()
        if stopped_autoplay:
            self.stop_playback()

        if self.history:
            self.game = self.history.pop()
            self.selected_card = None
            self.render()
            if stopped_autoplay:
                self.set_status("已撤销，播放已停止")
        else:
            if stopped_autoplay:
                self.set_status("播放已停止，无可撤销")
            print("没有可撤销的历史操作")




if __name__ == "__main__":
    root = tk.Tk()
    app = GameGUI(root)
    root.mainloop()
