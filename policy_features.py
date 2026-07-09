FEATURE_VERSION = 2
SUPPORTED_SAMPLE_VERSION = 1

SUITS = ("SPADES", "HEARTS", "CLUBS", "DIAMONDS")
MOVE_TYPES = ("FREE_TO_HOME", "COL_TO_HOME", "FREE_TO_COL", "COL_TO_COL", "COL_TO_FREE")

CARDS = [(suit, value) for suit in SUITS for value in range(1, 14)]
CARD_INDEX = {card: idx for idx, card in enumerate(CARDS)}

STATE_HOME_NEXT_START = 252
STATE_MOVABLE_SUFFIX_START = STATE_HOME_NEXT_START + len(SUITS)
STATE_BURIED_LOW_START = STATE_MOVABLE_SUFFIX_START + 8
STATE_FEATURE_SIZE = STATE_BURIED_LOW_START + 8

MOVE_INCREASES_HOME_IDX = 38
MOVE_RELEASES_LOW_IDX = MOVE_INCREASES_HOME_IDX + 1
MOVE_SOURCE_EMPTIED_IDX = MOVE_RELEASES_LOW_IDX + 1
MOVE_OCCUPIES_FREE_IDX = MOVE_SOURCE_EMPTIED_IDX + 1
MOVE_RELEASES_FREE_IDX = MOVE_OCCUPIES_FREE_IDX + 1
MOVE_TO_EMPTY_COLUMN_IDX = MOVE_RELEASES_FREE_IDX + 1
MOVE_REDUCES_BUFFER_IDX = MOVE_TO_EMPTY_COLUMN_IDX + 1
MOVE_TARGET_NEAR_HOME_IDX = MOVE_REDUCES_BUFFER_IDX + 1
MOVE_FEATURE_SIZE = MOVE_TARGET_NEAR_HOME_IDX + 1


def validate_sample(sample: dict) -> None:
    if sample.get("version") != SUPPORTED_SAMPLE_VERSION:
        raise ValueError(f"unsupported sample version: {sample.get('version')}")

    legal_moves = sample.get("legal_moves")
    if not legal_moves:
        raise ValueError("sample has no legal moves")

    action_index = sample.get("action_index")
    if not isinstance(action_index, int) or action_index < 0 or action_index >= len(legal_moves):
        raise ValueError(f"action_index out of range: {action_index}")

    action = sample.get("action")
    if action != legal_moves[action_index]:
        raise ValueError("action does not match legal_moves[action_index]")

    state = sample.get("state")
    if not isinstance(state, dict):
        raise ValueError("sample state must be a dict")
    _validate_state_shape(state)


def state_features_from_sample(sample: dict) -> list[float]:
    state = sample["state"]
    return state_to_features(state, remaining_moves=sample.get("remaining_moves", 0))


def move_features_from_sample(sample: dict, move: dict) -> list[float]:
    return move_to_features(sample["state"], move)


def features_from_sample(sample: dict) -> tuple[list[float], list[list[float]], int]:
    validate_sample(sample)
    state_features = state_features_from_sample(sample)
    move_features = [move_features_from_sample(sample, move) for move in sample["legal_moves"]]
    return state_features, move_features, sample["action_index"]


def features_from_state_and_moves(
    state: dict,
    legal_moves: list[dict],
    remaining_moves: int = 0,
) -> tuple[list[float], list[list[float]]]:
    _validate_state_shape(state)
    if not legal_moves:
        raise ValueError("legal_moves must not be empty")
    return (
        state_to_features(state, remaining_moves=remaining_moves),
        [move_to_features(state, move) for move in legal_moves],
    )


def state_to_features(state: dict, remaining_moves: int = 0) -> list[float]:
    _validate_state_shape(state)
    locations = _card_locations(state)
    features: list[float] = []

    for suit, value in CARDS:
        zone, index, depth, is_accessible = locations.get((suit, value), (0.0, 0.0, 0.0, 0.0))
        features.extend([zone, index, depth, is_accessible])

    home_cells = state["home_cells"]
    home_progress = [len(home_cells.get(suit, [])) / 13.0 for suit in SUITS]
    features.extend(home_progress)

    free_cells = state["free_cells"]
    features.extend([1.0 if card is not None else 0.0 for card in free_cells])

    columns = state["columns"]
    features.extend([min(len(column), 52) / 52.0 for column in columns])
    features.extend([_card_value(column[-1]) / 13.0 if column else 0.0 for column in columns])
    features.extend([_card_color_value(column[-1]) if column else 0.0 for column in columns])
    features.extend([1.0 if not column else 0.0 for column in columns])

    total_home = sum(len(home_cells.get(suit, [])) for suit in SUITS)
    empty_columns = sum(1 for column in columns if not column)
    occupied_free = sum(1 for card in free_cells if card is not None)
    features.extend(
        [
            total_home / 52.0,
            empty_columns / 8.0,
            occupied_free / 4.0,
            min(max(remaining_moves or 0, 0), 200) / 200.0,
        ]
    )
    features.extend(_next_home_value(home_cells, suit) / 13.0 for suit in SUITS)
    features.extend(min(_movable_suffix_length(column), 13) / 13.0 for column in columns)
    features.extend(1.0 if _has_buried_low_card(column) else 0.0 for column in columns)

    if len(features) != STATE_FEATURE_SIZE:
        raise AssertionError(f"state feature size changed: {len(features)}")
    return features


def move_to_features(state: dict, move: dict) -> list[float]:
    _validate_state_shape(state)
    features: list[float] = []
    move_type = move.get("move_type")
    from_idx = move.get("from_idx")
    to_idx = move.get("to_idx")
    count = move.get("count", 1) or 1

    features.extend(_one_hot(MOVE_TYPES, move_type))
    features.extend(_index_one_hot(from_idx, 8))
    features.extend(_index_one_hot(to_idx, 8))
    features.append(1.0 if to_idx is None else 0.0)
    features.append(min(max(count, 1), 13) / 13.0)

    source_is_free = move_type in ("FREE_TO_HOME", "FREE_TO_COL")
    source_is_col = move_type in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL")
    dest_is_free = move_type == "COL_TO_FREE"
    dest_is_col = move_type in ("FREE_TO_COL", "COL_TO_COL")
    is_home_move = move_type in ("FREE_TO_HOME", "COL_TO_HOME")
    features.extend(
        [
            1.0 if is_home_move else 0.0,
            1.0 if source_is_free else 0.0,
            1.0 if dest_is_free else 0.0,
            1.0 if dest_is_col else 0.0,
            1.0 if source_is_col else 0.0,
            1.0 if count > 1 else 0.0,
        ]
    )

    moving_card = _moving_card(state, move)
    destination_top = _destination_top_card(state, move)
    features.append(_card_value(moving_card) / 13.0 if moving_card else 0.0)
    features.append(_card_color_value(moving_card) if moving_card else 0.0)
    features.extend(_one_hot(SUITS, moving_card.get("suit") if moving_card else None))
    features.append(_card_value(destination_top) / 13.0 if destination_top else 0.0)
    features.append(_card_color_value(destination_top) if destination_top else 0.0)
    features.append(1.0 if dest_is_col and destination_top is None else 0.0)
    features.extend(
        [
            1.0 if is_home_move and moving_card else 0.0,
            1.0 if _move_releases_low_card(state, move) else 0.0,
            1.0 if _move_empties_source_column(state, move) else 0.0,
            1.0 if dest_is_free and moving_card else 0.0,
            1.0 if source_is_free and moving_card else 0.0,
            1.0 if dest_is_col and destination_top is None else 0.0,
            1.0 if _move_reduces_buffer_space(state, move) else 0.0,
            1.0 if _move_targets_near_home_card(state, move, moving_card) else 0.0,
        ]
    )

    if len(features) != MOVE_FEATURE_SIZE:
        raise AssertionError(f"move feature size changed: {len(features)}")
    return features


def _validate_state_shape(state: dict) -> None:
    columns = state.get("columns")
    free_cells = state.get("free_cells")
    home_cells = state.get("home_cells")
    if not isinstance(columns, list) or len(columns) != 8:
        raise ValueError("state.columns must contain 8 columns")
    if not isinstance(free_cells, list) or len(free_cells) != 4:
        raise ValueError("state.free_cells must contain 4 cells")
    if not isinstance(home_cells, dict):
        raise ValueError("state.home_cells must be a dict")
    for suit in SUITS:
        if suit not in home_cells:
            raise ValueError(f"state.home_cells is missing {suit}")


def _card_locations(state: dict) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    locations = {}
    for col_idx, column in enumerate(state["columns"]):
        height = max(len(column), 1)
        for depth, card in enumerate(column):
            locations[_card_key(card)] = (
                1.0 / 3.0,
                col_idx / 7.0,
                depth / max(height - 1, 1),
                1.0 if depth == len(column) - 1 else 0.0,
            )

    for free_idx, card in enumerate(state["free_cells"]):
        if card is not None:
            locations[_card_key(card)] = (2.0 / 3.0, free_idx / 3.0, 0.0, 1.0)

    for suit_idx, suit in enumerate(SUITS):
        stack = state["home_cells"].get(suit, [])
        height = max(len(stack), 1)
        for depth, card in enumerate(stack):
            locations[_card_key(card)] = (
                1.0,
                suit_idx / 3.0,
                depth / max(height - 1, 1),
                1.0 if depth == len(stack) - 1 else 0.0,
            )
    return locations


def _moving_card(state: dict, move: dict):
    move_type = move.get("move_type")
    from_idx = move.get("from_idx")
    count = move.get("count", 1) or 1
    if move_type in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL"):
        column = _safe_get(state["columns"], from_idx, [])
        if not column or len(column) < count:
            return None
        return column[-count]
    if move_type in ("FREE_TO_HOME", "FREE_TO_COL"):
        return _safe_get(state["free_cells"], from_idx)
    return None


def _destination_top_card(state: dict, move: dict):
    if move.get("move_type") not in ("FREE_TO_COL", "COL_TO_COL"):
        return None
    column = _safe_get(state["columns"], move.get("to_idx"), [])
    return column[-1] if column else None


def _next_home_value(home_cells: dict, suit: str) -> int:
    stack = home_cells.get(suit, [])
    top_value = _card_value(stack[-1]) if stack else 0
    next_value = top_value + 1
    return next_value if next_value <= 13 else 0


def _movable_suffix_length(column: list[dict]) -> int:
    if not column:
        return 0
    length = 1
    for idx in range(len(column) - 2, -1, -1):
        lower_card = column[idx]
        upper_card = column[idx + 1]
        if _card_color_value(lower_card) == _card_color_value(upper_card):
            break
        if _card_value(lower_card) != _card_value(upper_card) + 1:
            break
        length += 1
    return length


def _has_buried_low_card(column: list[dict]) -> bool:
    return any(_is_low_card(card) for card in column[:-1])


def _is_low_card(card: dict) -> bool:
    value = _card_value(card)
    return 1 <= value <= 3


def _move_empties_source_column(state: dict, move: dict) -> bool:
    if move.get("move_type") not in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL"):
        return False
    count = move.get("count", 1) or 1
    column = _safe_get(state["columns"], move.get("from_idx"), [])
    return bool(column) and len(column) == count


def _move_releases_low_card(state: dict, move: dict) -> bool:
    if move.get("move_type") not in ("COL_TO_HOME", "COL_TO_FREE", "COL_TO_COL"):
        return False
    count = move.get("count", 1) or 1
    column = _safe_get(state["columns"], move.get("from_idx"), [])
    remaining = len(column) - count
    if remaining <= 0:
        return False
    return _is_low_card(column[remaining - 1])


def _move_reduces_buffer_space(state: dict, move: dict) -> bool:
    before = _available_buffer_count(state)
    after = before
    move_type = move.get("move_type")

    if move_type == "COL_TO_FREE":
        after -= 1
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "FREE_TO_COL":
        after += 1
        if _destination_top_card(state, move) is None:
            after -= 1
    elif move_type == "COL_TO_COL":
        if _destination_top_card(state, move) is None:
            after -= 1
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "COL_TO_HOME":
        if _move_empties_source_column(state, move):
            after += 1
    elif move_type == "FREE_TO_HOME":
        after += 1

    return after < before


def _available_buffer_count(state: dict) -> int:
    return sum(1 for card in state["free_cells"] if card is None) + sum(1 for column in state["columns"] if not column)


def _move_targets_near_home_card(state: dict, move: dict, moving_card) -> bool:
    if move.get("move_type") != "COL_TO_COL" or moving_card is None:
        return False
    next_value = _next_home_value(state["home_cells"], moving_card.get("suit"))
    if next_value <= 0:
        return False
    return _card_value(moving_card) <= next_value + 2


def _card_key(card: dict) -> tuple[str, int]:
    return card.get("suit"), int(card.get("value", 0))


def _card_value(card: dict) -> int:
    if card is None:
        return 0
    return int(card.get("value", 0))


def _card_color_value(card: dict) -> float:
    suit = card.get("suit") if card else None
    if suit in ("HEARTS", "DIAMONDS"):
        return 1.0
    if suit in ("SPADES", "CLUBS"):
        return -1.0
    return 0.0


def _one_hot(values, selected) -> list[float]:
    return [1.0 if value == selected else 0.0 for value in values]


def _index_one_hot(index, size: int) -> list[float]:
    output = [0.0] * size
    if isinstance(index, int) and 0 <= index < size:
        output[index] = 1.0
    return output


def _safe_get(items, index, default=None):
    if not isinstance(index, int) or index < 0 or index >= len(items):
        return default
    return items[index]
