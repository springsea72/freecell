import io
import unittest
from contextlib import redirect_stdout

import gui
from game_model import Card, FreeCellGame, Move, MoveType, Suit
from solver import SolveResult


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


def one_move_game():
    game = FreeCellGame(deal=empty_deal())
    game.columns[0] = [card(Suit.SPADES, 1)]
    return game


def two_move_game():
    game = FreeCellGame(deal=empty_deal())
    game.columns[0] = [card(Suit.SPADES, 2), card(Suit.SPADES, 1)]
    return game


class FakeRoot:
    def __init__(self):
        self.callbacks = {}
        self.delays = {}
        self.cancelled = set()
        self.next_id = 0

    def after(self, delay, callback):
        self.next_id += 1
        after_id = f"after-{self.next_id}"
        self.callbacks[after_id] = callback
        self.delays[after_id] = delay
        return after_id

    def after_cancel(self, after_id):
        self.cancelled.add(after_id)

    def run(self, after_id):
        if after_id not in self.cancelled:
            self.callbacks[after_id]()


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeButton:
    def __init__(self):
        self.kwargs = {}

    def config(self, **kwargs):
        self.kwargs.update(kwargs)


class FakeEvent:
    x = 20
    y = 100


def make_app(game):
    app = gui.GameGUI.__new__(gui.GameGUI)
    app.root = FakeRoot()
    app.game = game
    app.history = []
    app.playback_moves = []
    app.playback_index = 0
    app.playback_running = False
    app.playback_paused = False
    app.solve_running = False
    app.playback_after_id = None
    app.playback_generation = 0
    app.status_var = FakeVar()
    app.playback_speed_var = FakeVar(gui.PLAYBACK_INTERVAL_MS)
    app.solve_button = FakeButton()
    app.pause_button = FakeButton()
    app.stop_button = FakeButton()
    app.render_calls = 0
    app.victory_checks = 0

    def render():
        app.render_calls += 1

    def check_victory():
        app.victory_checks += 1

    app.render = render
    app.check_victory = check_victory
    return app


class GuiAutoplayTests(unittest.TestCase):
    def test_playback_legal_path_updates_state_and_status(self):
        app = make_app(one_move_game())
        move = Move(MoveType.COL_TO_HOME, 0)

        app.start_playback([move])
        app.root.run(app.playback_after_id)

        self.assertFalse(app.playback_running)
        self.assertEqual(1, app.playback_index)
        self.assertEqual([card(Suit.SPADES, 1)], app.game.home_cells[Suit.SPADES])
        self.assertEqual(1, len(app.history))
        self.assertEqual(1, app.render_calls)
        self.assertEqual(1, app.victory_checks)

    def test_stop_playback_prevents_scheduled_callback_from_consuming_move(self):
        app = make_app(two_move_game())
        first_move = Move(MoveType.COL_TO_HOME, 0)
        second_move = Move(MoveType.COL_TO_HOME, 0)

        app.start_playback([first_move, second_move])
        scheduled = app.playback_after_id
        app.stop_playback()
        app.root.run(scheduled)

        self.assertFalse(app.playback_running)
        self.assertEqual(0, app.playback_index)
        self.assertEqual([], app.game.home_cells[Suit.SPADES])
        self.assertEqual([], app.history)

    def test_solve_failure_does_not_modify_game_state(self):
        app = make_app(one_move_game())
        app.solve_running = True
        token = app.playback_generation
        before = app.game.state_key()

        app.on_solve_finished(
            SolveResult(False, [], 1, 0, 0, "max_nodes_exceeded"),
            token,
        )

        self.assertFalse(app.solve_running)
        self.assertFalse(app.playback_running)
        self.assertEqual(before, app.game.state_key())
        self.assertEqual([], app.history)
        self.assertIn("max_nodes_exceeded", app.status_var.get())

    def test_playback_uses_game_apply_move(self):
        app = make_app(one_move_game())
        move = Move(MoveType.COL_TO_HOME, 0)
        calls = []
        original_apply_move = app.game.apply_move

        def spy_apply_move(candidate):
            calls.append(candidate)
            return original_apply_move(candidate)

        app.game.apply_move = spy_apply_move

        app.start_playback([move])
        app.root.run(app.playback_after_id)

        self.assertEqual([move], calls)

    def test_undo_during_playback_stops_playback_and_invalidates_callback(self):
        app = make_app(two_move_game())
        first_move = Move(MoveType.COL_TO_HOME, 0)
        second_move = Move(MoveType.COL_TO_HOME, 0)

        app.start_playback([first_move, second_move])
        app.root.run(app.playback_after_id)
        stale_after_id = app.playback_after_id

        self.assertEqual([card(Suit.SPADES, 1)], app.game.home_cells[Suit.SPADES])
        self.assertEqual(1, app.playback_index)

        app.undo()
        app.root.run(stale_after_id)

        self.assertEqual([card(Suit.SPADES, 2), card(Suit.SPADES, 1)], app.game.columns[0])
        self.assertEqual([], app.game.home_cells[Suit.SPADES])
        self.assertFalse(app.playback_running)
        self.assertFalse(app.playback_paused)
        self.assertFalse(app.solve_running)
        self.assertEqual([], app.playback_moves)
        self.assertEqual(0, app.playback_index)
        self.assertEqual([], app.history)

    def test_undo_during_solve_stops_pending_solve_result(self):
        app = make_app(one_move_game())
        app.solve_running = True
        token = app.playback_generation
        before = app.game.state_key()

        with redirect_stdout(io.StringIO()):
            app.undo()
        app.on_solve_finished(
            SolveResult(True, [Move(MoveType.COL_TO_HOME, 0)], 1, 1, 1, "won"),
            token,
        )

        self.assertFalse(app.solve_running)
        self.assertFalse(app.playback_running)
        self.assertFalse(app.playback_paused)
        self.assertEqual([], app.playback_moves)
        self.assertEqual(0, app.playback_index)
        self.assertEqual(before, app.game.state_key())
        self.assertEqual([], app.history)

    def test_manual_input_is_ignored_while_autoplay_active(self):
        app = make_app(one_move_game())
        app.playback_running = True
        card_to_drag = app.game.columns[0][-1]
        app.selected_card = card_to_drag
        app.selected_sequence = [card_to_drag]
        app.selected_col_idx = 0
        app.start_pos = (0, 0)
        before = app.game.state_key()

        app.on_click(FakeEvent())
        app.on_drag(FakeEvent())
        app.on_release(FakeEvent())
        app.on_right_click(FakeEvent())

        self.assertEqual(before, app.game.state_key())
        self.assertEqual([], app.history)
        self.assertTrue(app.playback_running)

    def test_schedule_next_playback_step_uses_current_speed_value(self):
        app = make_app(one_move_game())
        app.playback_running = True
        app.playback_speed_var.set(350)

        app.schedule_next_playback_step()

        self.assertEqual(350, app.root.delays[app.playback_after_id])

    def test_solve_success_status_text_contains_stats(self):
        app = make_app(one_move_game())
        app.solve_running = True
        token = app.playback_generation
        result = SolveResult(
            True,
            [Move(MoveType.COL_TO_HOME, 0)],
            explored_nodes=7,
            generated_nodes=8,
            max_frontier=9,
            reason="won",
        )

        app.on_solve_finished(result, token)

        status = app.status_var.get()
        self.assertIn("solved", status)
        self.assertIn("reason=won", status)
        self.assertIn("explored_nodes=7", status)
        self.assertIn("generated_nodes=8", status)
        self.assertIn("max_frontier=9", status)
        self.assertIn("path length=1", status)

    def test_solve_failure_status_text_contains_stats(self):
        app = make_app(one_move_game())
        app.solve_running = True
        token = app.playback_generation
        result = SolveResult(
            False,
            [],
            explored_nodes=5000,
            generated_nodes=4999,
            max_frontier=120,
            reason="max_nodes_exceeded",
        )

        app.on_solve_finished(result, token)

        status = app.status_var.get()
        self.assertIn("failed", status)
        self.assertIn("reason=max_nodes_exceeded", status)
        self.assertIn("explored_nodes=5000", status)
        self.assertIn("generated_nodes=4999", status)
        self.assertIn("max_frontier=120", status)
        self.assertIn("path length=0", status)

    def test_manual_input_allowed_after_stop_or_finish(self):
        stopped = make_app(one_move_game())
        stopped.playback_running = True
        stopped.stop_playback()
        stopped.on_right_click(FakeEvent())

        self.assertEqual([card(Suit.SPADES, 1)], stopped.game.home_cells[Suit.SPADES])

        finished = make_app(one_move_game())
        finished.playback_running = True
        finished.finish_playback()
        finished.on_right_click(FakeEvent())

        self.assertEqual([card(Suit.SPADES, 1)], finished.game.home_cells[Suit.SPADES])


if __name__ == "__main__":
    unittest.main()
