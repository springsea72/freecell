import unittest

from game_model import Card, FreeCellGame, Move, MoveType, Suit


def card(suit, value):
    return Card(suit, value)


def empty_deal():
    return [[] for _ in range(8)]


class FreeCellGameTests(unittest.TestCase):
    def test_same_seed_reproduces_deal(self):
        first = FreeCellGame(seed=123)
        second = FreeCellGame(seed=123)

        self.assertEqual(first.state_key(), second.state_key())

    def test_initial_deal_has_52_unique_cards(self):
        game = FreeCellGame(seed=1)
        cards = [card for column in game.columns for card in column]

        self.assertEqual(52, len(cards))
        self.assertEqual(52, len(set(cards)))

    def test_generated_moves_are_executable_on_clone(self):
        game = FreeCellGame(seed=1)
        moves = game.generate_legal_moves()

        self.assertGreater(len(moves), 0)
        for move in moves:
            with self.subTest(move=move):
                clone = game.clone()
                self.assertTrue(clone.apply_move(move))

    def test_illegal_move_does_not_change_state(self):
        game = FreeCellGame(seed=1)
        before = game.state_key()

        self.assertFalse(game.apply_move(Move(MoveType.COL_TO_COL, 0, 0)))
        self.assertEqual(before, game.state_key())

    def test_sequence_move_and_max_movable_length(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 9), card(Suit.HEARTS, 8)],
                [card(Suit.DIAMONDS, 10)],
                [card(Suit.CLUBS, 2)],
                [card(Suit.DIAMONDS, 3)],
                [card(Suit.SPADES, 4)],
                [card(Suit.HEARTS, 5)],
                [card(Suit.CLUBS, 6)],
                [card(Suit.DIAMONDS, 7)],
            ]
        )
        game.free_cells = [
            card(Suit.SPADES, 1),
            card(Suit.HEARTS, 1),
            card(Suit.CLUBS, 1),
            card(Suit.DIAMONDS, 1),
        ]

        blocked_move = Move(MoveType.COL_TO_COL, 0, 1, count=2)
        self.assertEqual(1, game.max_movable_sequence_length())
        self.assertFalse(game.can_apply_move(blocked_move))

        game.free_cells[0] = None
        allowed_move = Move(MoveType.COL_TO_COL, 0, 1, count=2)

        self.assertEqual(2, game.max_movable_sequence_length())
        self.assertIn(allowed_move, game.generate_legal_moves())
        self.assertTrue(game.apply_move(allowed_move))
        self.assertEqual(
            [card(Suit.DIAMONDS, 10), card(Suit.SPADES, 9), card(Suit.HEARTS, 8)],
            game.columns[1],
        )

    def test_sequence_move_to_only_empty_target_column_is_illegal_without_buffer(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 9), card(Suit.HEARTS, 8)],
                [],
                [card(Suit.CLUBS, 2)],
                [card(Suit.DIAMONDS, 3)],
                [card(Suit.SPADES, 4)],
                [card(Suit.HEARTS, 5)],
                [card(Suit.CLUBS, 6)],
                [card(Suit.DIAMONDS, 7)],
            ]
        )
        game.free_cells = [
            card(Suit.SPADES, 1),
            card(Suit.HEARTS, 1),
            card(Suit.CLUBS, 1),
            card(Suit.DIAMONDS, 1),
        ]
        move = Move(MoveType.COL_TO_COL, 0, 1, count=2)
        before = game.state_key()

        self.assertFalse(game.can_apply_move(move))
        self.assertNotIn(move, game.generate_legal_moves())
        self.assertFalse(game.apply_move(move))
        self.assertEqual(before, game.state_key())

    def test_sequence_move_to_empty_column_can_use_other_empty_column_as_buffer(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 9), card(Suit.HEARTS, 8)],
                [],
                [],
                [card(Suit.DIAMONDS, 3)],
                [card(Suit.SPADES, 4)],
                [card(Suit.HEARTS, 5)],
                [card(Suit.CLUBS, 6)],
                [card(Suit.DIAMONDS, 7)],
            ]
        )
        game.free_cells = [
            card(Suit.SPADES, 1),
            card(Suit.HEARTS, 1),
            card(Suit.CLUBS, 1),
            card(Suit.DIAMONDS, 1),
        ]
        move = Move(MoveType.COL_TO_COL, 0, 1, count=2)

        self.assertTrue(game.can_apply_move(move))
        self.assertIn(move, game.generate_legal_moves())
        self.assertTrue(game.apply_move(move))
        self.assertEqual([card(Suit.SPADES, 9), card(Suit.HEARTS, 8)], game.columns[1])

    def test_free_cell_to_home(self):
        game = FreeCellGame(deal=empty_deal())
        game.free_cells[0] = card(Suit.SPADES, 1)

        self.assertTrue(game.apply_move(Move(MoveType.FREE_TO_HOME, 0)))
        self.assertIsNone(game.free_cells[0])
        self.assertEqual([card(Suit.SPADES, 1)], game.home_cells[Suit.SPADES])

    def test_state_key_is_hashable_and_changes_after_move(self):
        game = FreeCellGame(seed=2)
        before = game.state_key()

        hash(before)
        self.assertTrue(game.apply_move(game.generate_legal_moves()[0]))
        self.assertNotEqual(before, game.state_key())

    def test_won_state(self):
        game = FreeCellGame(deal=empty_deal())
        for suit in Suit:
            game.home_cells[suit] = [card(suit, value) for value in range(1, 14)]

        self.assertTrue(game.is_won())
        self.assertTrue(game.is_terminal())

    def test_is_terminal_when_no_legal_moves_and_not_won(self):
        game = FreeCellGame(
            deal=[
                [card(Suit.SPADES, 10)],
                [card(Suit.CLUBS, 10)],
                [card(Suit.SPADES, 11)],
                [card(Suit.CLUBS, 11)],
                [card(Suit.SPADES, 12)],
                [card(Suit.CLUBS, 12)],
                [card(Suit.SPADES, 13)],
                [card(Suit.CLUBS, 13)],
            ]
        )
        game.free_cells = [
            card(Suit.SPADES, 2),
            card(Suit.CLUBS, 3),
            card(Suit.SPADES, 4),
            card(Suit.CLUBS, 5),
        ]

        self.assertFalse(game.is_won())
        self.assertEqual([], game.generate_legal_moves())
        self.assertTrue(game.is_terminal())

    def test_auto_move_to_home_moves_columns_and_free_cells(self):
        game = FreeCellGame(deal=empty_deal())
        game.columns[0] = [card(Suit.SPADES, 2), card(Suit.SPADES, 1)]
        game.free_cells[0] = card(Suit.HEARTS, 1)
        game.free_cells[1] = card(Suit.HEARTS, 2)

        self.assertTrue(game.auto_move_to_home())
        self.assertEqual(
            [card(Suit.SPADES, 1), card(Suit.SPADES, 2)],
            game.home_cells[Suit.SPADES],
        )
        self.assertEqual(
            [card(Suit.HEARTS, 1), card(Suit.HEARTS, 2)],
            game.home_cells[Suit.HEARTS],
        )


if __name__ == "__main__":
    unittest.main()
