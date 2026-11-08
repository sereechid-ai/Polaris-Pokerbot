# Imports
 
import os
import time
import gzip
import itertools
import json
from collections import Counter
 
import traceback
 
import numpy as np
 
import uvicorn
 
from fastapi import FastAPI, Request
 
from fastapi.responses import JSONResponse
 
from evaluator import category_of, card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py*
 
from game import Action, AgentConfig, HandResult, Observation, Street
 
# Build omaha_score's lookup table now, while the server starts, instead of
 
# during your first decision (where it would eat into your time bank).
 
# Do any other slow setup (loading a model, ...) here too: the server must
 
# answer within 15 seconds of starting.
 
warmup()
 
# Lookup tables built from the solver CSV by build_tables.py.
_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_tables.json.gz")
with gzip.open(_TABLE_PATH, "rt") as _f:
    TABLES = json.load(_f)
 
# rename as your team/bot's name
 
PARTICIPANT = "TeamPolaris"
 
AGENT_NAME = "PolarisAgent"
 
def new_half(config: AgentConfig, memory: dict) -> None:
 
    memory["hands_played"] = 0
 
    memory["_time_used"] = 0.0
 
    memory["starting_stack"] = config.starting_stack
 
    memory["big_blind"] = config.big_blind
 
    memory["opp"] = {
 
        "hands": 0,
 
        "vpip": 0,
 
        "pfr": 0,
 
        "postflop_raises": 0,
 
        "postflop_calls": 0,
 
        "postflop_checks": 0,
 
        "faced_our_raise": 0,
 
        "folded_to_our_raise": 0,
 
        "folds": 0,
 
        "showdowns": 0,
 
        "showdown_wins": 0,
 
        "sd_category_counts": [0] * 9,
 
        "value_raises": 0,
 
        "weak_calldowns": 0,
 
        "bluff_raises": 0,
 
        "vpip_weak_hole": 0,
 
        "vpip_strong_hole": 0,
 
        "sd_hands_vpip": 0,
 
        "checkdowns": 0,
 
        "checkdown_eligible": 0,
 
    }
 
    memory["hand"] = {
 
        "vpip": False,
 
        "pfr": False,
 
        "saw_flop": False,
 
        "actions": [],
 
        "seen": 0,
 
        "raise_pending": False,
 
    }
 
 
RANK_LETTERS = "6789TJQKA"
ACTION_NAMES = ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT")
 
 
RIVER_EQUITY_X = [0.0, 0.0522, 0.1484, 0.238, 0.3293, 0.4164, 0.4874, 0.5794, 0.6555, 0.7571, 0.8569, 0.9383, 0.9939, 1.0]
RIVER_STRENGTH_Y = [0.0, 0.0081, 0.0513, 0.1024, 0.1789, 0.2455, 0.3299, 0.5079, 0.6019, 0.6893, 0.7827, 0.8868, 0.9487, 1.0]
 
 
def _straight_high(ranks):
    unique = sorted(set(ranks))
    if len(unique) < 5:
        return -1
    if unique[-1] - unique[0] == 4:
        return unique[-1]
    if unique == [0, 1, 2, 3, 8]:  # A-6-7-8-9
        return 3
    return -1
 
 
def _five_card_value(cards):
    # Short-deck order used by the dataset: flush > full house, trips > straight.
    ranks = [card_rank(c) for c in cards]
    is_flush = len({card_suit(c) for c in cards}) == 1
    straight = _straight_high(ranks)
    counts = Counter(ranks)
    ordered = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = [counts[r] for r in ordered]
    if is_flush and straight >= 0:
        return (8, [straight])
    if shape[0] == 4:
        return (7, ordered)
    if is_flush:
        return (6, sorted(ranks, reverse=True))
    if shape[:2] == [3, 2]:
        return (5, ordered)
    if shape[0] == 3:
        return (4, ordered)
    if straight >= 0:
        return (3, [straight])
    if shape[:2] == [2, 2]:
        return (2, ordered)
    if shape[0] == 2:
        return (1, ordered)
    return (0, sorted(ranks, reverse=True))
 
 
def made_hand_label(hole, board):
    category, ordered = max(
        _five_card_value(h + b)
        for h in itertools.combinations(hole, 2)
        for b in itertools.combinations(board, 3)
    )
    board_ranks = sorted({card_rank(c) for c in board}, reverse=True)
    if category >= 3:
        return {3: "straight", 4: "trips", 5: "full_house", 6: "flush"}.get(category, "quads_plus")
    if category == 2:
        return "two_pair_top" if ordered[0] >= board_ranks[0] else "two_pair_low"
    if category == 1:
        if ordered[0] > board_ranks[0]:
            return "overpair"
        if ordered[0] == board_ranks[0]:
            return "top_pair"
        if len(board_ranks) > 1 and ordered[0] == board_ranks[1]:
            return "second_pair"
        return "low_pair"
    return "high_card"
 
 
def flush_draw_label(hole, board):
    suit_counts = Counter(card_suit(c) for c in board)
    label = "none"
    if max(suit_counts.values()) > 2:
        return label
    board_set = set(board)
    for suit, count in suit_counts.items():
        hole_ranks = [card_rank(c) for c in hole if card_suit(c) == suit]
        if count != 2 or len(hole_ranks) < 2:
            continue
        label = "non_nut" if label == "none" else label
        board_ranks = {card_rank(c) for c in board_set if card_suit(c) == suit}
        if max(hole_ranks) == max(r for r in range(9) if r not in board_ranks):
            label = "nut"
    return label
 
 
def straight_draw_label(hole, board):
    hole_ranks = [card_rank(c) for c in hole]
    board_ranks = [card_rank(c) for c in board]
    outs = 0
    for new_rank in range(9):
        if any(
            _straight_high(list(h) + list(b) + [new_rank]) >= 0
            for h in itertools.combinations(hole_ranks, 2)
            for b in itertools.combinations(board_ranks, 2)
        ):
            outs += 1
    return "none" if outs == 0 else ("weak" if outs <= 2 else "strong")
 
 
def table_hand_key(obs, river_equity=None):
    hole, board = tuple(obs.hole_cards), tuple(obs.board)
    if not board:
        ordered = sorted(hole, key=lambda c: (-card_rank(c), card_suit(c)))
        suits, text = {}, ""
        for card in ordered:
            suit = card_suit(card)
            if suit not in suits:
                suits[suit] = "wxyz"[len(suits)]
            text += "6789TJQKA"[card_rank(card)] + suits[suit]
        bucket = TABLES["preflop_buckets"].get(text)
        if bucket is None:
            bucket = TABLES["preflop_rank_buckets"].get(text[0::2])
        if bucket is None:
            # Unseen rank pattern: use the closest pattern that was seen.
            ranks = [RANK_LETTERS.index(r) for r in text[0::2]]
            nearest = min(
                TABLES["preflop_rank_buckets"],
                key=lambda p: sum(abs(RANK_LETTERS.index(x) - y) for x, y in zip(p, ranks)),
            )
            bucket = TABLES["preflop_rank_buckets"][nearest]
        return str(bucket)
 
    paired = int(len({card_rank(c) for c in board}) < len(board))
    max_suit = max(Counter(card_suit(c) for c in board).values())
    if len(board) == 5:
        strength = float(np.interp(river_equity, RIVER_EQUITY_X, RIVER_STRENGTH_Y))
        return str(4 * int(round(strength * 39)) + 2 * paired + int(max_suit >= 3))
    return ",".join([
        made_hand_label(hole, board), flush_draw_label(hole, board),
        straight_draw_label(hole, board), str(paired), str(max_suit - 1),
    ])
 
 
def _nearby_numeric_hands(hand, street, max_steps, other_board_classes=False):
    """Yield solver buckets closest in strength to a preflop or river bucket."""
    bucket = int(hand)
    step = 4 if street == "RIVER" else 1
    limit = 160 if street == "RIVER" else 128
    for distance in range(1, max_steps + 1):
        for candidate in (bucket - distance * step, bucket + distance * step):
            if 0 <= candidate < limit:
                yield str(candidate)
    if other_board_classes and street == "RIVER":
        strength_base = bucket - bucket % 4
        for distance in range(0, max_steps + 1):
            for base in (strength_base - 4 * distance, strength_base + 4 * distance):
                for board_class in range(4):
                    candidate = base + board_class
                    if 0 <= candidate < limit and candidate != bucket:
                        yield str(candidate)
 
 
def lookup_with_level(obs, river_equity=None):
    """Return (solver action mix, name of the table level that answered)."""
    street = {0: "PREFLOP", 3: "FLOP", 4: "TURN", 5: "RIVER"}[len(obs.board)]
    position = "button" if obs.is_button else "big_blind"
    facing = min(obs.to_call, 1)
    raises = obs.raises_this_street
    hand = table_hand_key(obs, river_equity)
    numeric = street in ("PREFLOP", "RIVER")
 
    def base(h):
        return f"{street}|{h}|{position}"
 
    def attempts():
        yield "exact", f"{base(hand)}|{raises}|{obs.pot}|{obs.to_call}|{obs.my_stack}|{obs.opp_stack}"
        yield "spot", f"{base(hand)}|{raises}|{obs.to_call}"
        if numeric:
            for h in _nearby_numeric_hands(hand, street, 3):
                yield "spot", f"{base(h)}|{raises}|{obs.to_call}"
        yield "by_raises", f"{base(hand)}|{raises}"
        if numeric:
            for h in _nearby_numeric_hands(hand, street, 6):
                yield "by_raises", f"{base(h)}|{raises}"
        yield "coarse", f"{base(hand)}|{facing}"
        if numeric:
            for h in _nearby_numeric_hands(hand, street, 20, other_board_classes=True):
                yield "coarse", f"{base(h)}|{facing}"
        else:
            made = hand.split(",")[0]
            yield "made", f"{street}|{made}|{position}|{raises}|{facing}"
            yield "made_coarse", f"{street}|{made}|{position}|{facing}"
        yield "street", f"{street}|{position}|{facing}"
 
    for level, key in attempts():
        mix = TABLES[level].get(key)
        if mix is not None:
            return dict(zip(ACTION_NAMES, mix)), level
    return None, None
 
 
def lookup_strategy(obs, river_equity=None):
    """Return the solver's action mix for this spot, or None if unknown."""
    return lookup_with_level(obs, river_equity)[0]
 
 
# Section 4 settings. The solver's mix is played unchanged unless the opponent's
# postflop fold-to-raise rate is clearly different from the solver's own.
SOLVER_FOLD_TO_RAISE = 0.448
MIN_RAISES_FACED = 60
SHRINKAGE_PRIOR = 100
MAX_MIX_SHIFT = 0.10
WEAK_MADE_HANDS = ("high_card", "low_pair", "second_pair")
 
 
def update_opponent_stats(obs, memory):
    """Count opponent actions that have appeared since our last decision."""
    opp, hand = memory["opp"], memory["hand"]
    history = obs.action_history
    if hand["raise_pending"] and len(history) > hand["seen"]:
        opp["faced_our_raise"] += 1
        hand["raise_pending"] = False
    for item in history[hand["seen"]:]:
        if item.is_me:
            continue
        is_raise = item.action in (Action.RAISE_HALF, Action.RAISE_POT)
        if item.street == Street.PREFLOP:
            if item.amount > 0 and not hand["vpip"]:
                hand["vpip"] = True
                opp["vpip"] += 1
            if is_raise and not hand["pfr"]:
                hand["pfr"] = True
                opp["pfr"] += 1
        elif is_raise:
            opp["postflop_raises"] += 1
        elif item.action == Action.CHECK_CALL:
            opp["postflop_calls" if item.amount > 0 else "postflop_checks"] += 1
    hand["seen"] = len(history)
 
 
def fold_to_raise_edge(opp):
    """Shrunken gap between observed and solver fold rates; 0 without clear evidence."""
    faced = opp["faced_our_raise"]
    if faced < MIN_RAISES_FACED:
        return 0.0
    gap = opp["folded_to_our_raise"] / faced - SOLVER_FOLD_TO_RAISE
    standard_error = (SOLVER_FOLD_TO_RAISE * (1 - SOLVER_FOLD_TO_RAISE) / faced) ** 0.5
    if abs(gap) < 2 * standard_error:
        return 0.0
    return gap * faced / (faced + SHRINKAGE_PRIOR)
 
 
def adjust_strategy(mix, obs, memory, river_equity):
    """Nudge a postflop solver mix, staying within the actions the solver uses."""
    if mix is None or not obs.board:
        return mix
    edge = fold_to_raise_edge(memory["opp"])
    if edge == 0.0:
        return mix
 
    adjusted = dict(mix)
    raise_total = adjusted["RAISE_HALF"] + adjusted["RAISE_POT"]
    if raise_total <= 0:
        return mix
    shift = min(abs(edge), MAX_MIX_SHIFT)
 
    if edge > 0:
        # They overfold: turn some calls and checks into raises.
        moved = adjusted["CHECK_CALL"] * shift
        adjusted["CHECK_CALL"] -= moved
        adjusted["RAISE_HALF"] += moved * adjusted["RAISE_HALF"] / raise_total
        adjusted["RAISE_POT"] += moved * mix["RAISE_POT"] / raise_total
    else:
        # They underfold: bluff less with weak hands, keep value raises.
        if len(obs.board) == 5:
            weak = river_equity is not None and river_equity < 0.5
        else:
            weak = made_hand_label(tuple(obs.hole_cards), tuple(obs.board)) in WEAK_MADE_HANDS
        if not weak:
            return mix
        moved = raise_total * shift
        adjusted["RAISE_HALF"] -= adjusted["RAISE_HALF"] * shift
        adjusted["RAISE_POT"] -= adjusted["RAISE_POT"] * shift
        adjusted["CHECK_CALL"] += moved
    return adjusted
 
 
def action_from_table(probabilities: dict, obs: Observation) -> Action:
    """Sample a legal action from a lookup-table strategy."""
    weights = {}
    illegal_raise_weight = 0.0
    for name in ACTION_NAMES:
        action = Action[name]
        weight = max(float(probabilities.get(name) or 0.0), 0.0)
        if action in obs.legal_actions:
            weights[action] = weight
        elif name.startswith("RAISE"):
            illegal_raise_weight += weight
 
    # If a raise size isn't allowed, use the other size, or else call.
    for fallback in (Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL):
        if fallback in weights:
            weights[fallback] += illegal_raise_weight
            break
 
    # Folding is never better than a free check.
    if obs.to_call == 0 and Action.FOLD in weights and Action.CHECK_CALL in weights:
        weights[Action.CHECK_CALL] += weights.pop(Action.FOLD)
 
    total = sum(weights.values())
    if total <= 0:
        return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
 
    legal = list(weights)
    rng = np.random.default_rng(obs.hand_index * 1000003 + sum(obs.hole_cards) * 101 + len(obs.action_history))
    return legal[int(rng.choice(len(legal), p=[weights[a] / total for a in legal]))]
 
 
def choose_action(obs: Observation, memory: dict) -> Action:
    started = time.perf_counter()
    try:
        action = _choose_action(obs, memory)
        memory["hand"]["raise_pending"] = (
            obs.street != Street.PREFLOP
            and action in (Action.RAISE_HALF, Action.RAISE_POT)
        )
        return action
    finally:
        memory["_time_used"] = memory.get("_time_used", 0.0) + time.perf_counter() - started
 
 
def _choose_action(obs: Observation, memory: dict) -> Action:
 
    #** -------------------- 1. Extract game state -------------------- **
 
    state = {
 
        "hand_index": obs.hand_index,
 
        "street": obs.street,
 
        "hole_cards": tuple(obs.hole_cards),
 
        "board": tuple(obs.board),
 
        "pot": obs.pot,
 
        "to_call": obs.to_call,
 
        "my_bet": obs.my_bet,
 
        "opp_bet": obs.opp_bet,
 
        "my_stack": obs.my_stack,
 
        "opp_stack": obs.opp_stack,
 
        "is_button": obs.is_button,
 
        "raises_this_street": obs.raises_this_street,
 
        "raise_to": dict(obs.raise_to),
 
        "min_raise_to": obs.min_raise_to,
 
        "legal_actions": tuple(obs.legal_actions),
 
        "action_history": tuple(obs.action_history),
 
        "big_blind": obs.big_blind,
 
    }
 
    state.update({
 
        "position": "button" if obs.is_button else "big_blind",
 
        "cards_remaining": 5 - len(obs.board),
 
        "board_cards": len(obs.board),
 
        "effective_stack": min(obs.my_stack, obs.opp_stack),
 
        "spr": (
 
            min(obs.my_stack, obs.opp_stack) / obs.pot
 
            if obs.pot > 0 else None
 
        ),
 
        "pot_odds": (
 
            obs.to_call / (obs.pot + obs.to_call)
 
            if obs.to_call > 0 else 0.0
 
        ),
 
        "facing_bet": obs.to_call > 0,
 
        "is_preflop": obs.street == Street.PREFLOP,
 
        "is_flop": obs.street == Street.FLOP,
 
        "is_turn": obs.street == Street.TURN,
 
        "is_river": obs.street == Street.RIVER,
 
    })
 
    state["all_known_cards"] = state["hole_cards"] + state["board"]
 
    state["known_card_count"] = len(state["all_known_cards"])
 
    state["unseen_card_count"] = 36 - state["known_card_count"]
 
    state["history_length"] = len(state["action_history"])
 
    state["my_actions"] = tuple(
 
        action for action in state["action_history"]
 
        if action.is_me
 
    )
 
    state["opp_actions"] = tuple(
 
        action for action in state["action_history"]
 
        if not action.is_me
 
    )
 
    state["my_action_count"] = len(state["my_actions"])
 
    state["opp_action_count"] = len(state["opp_actions"])
 
    #** -------------------- 2. Evaluate hand -------------------- **
 
    hole = state["hole_cards"]
 
    board = state["board"]
 
    if state["is_preflop"]:
        state["equity"] = None
        state["equity_samples"] = 0
 
    #** Postflop: Monte Carlo equity estimation **
 
    else:
 
        known_cards = set(hole + board)
 
        unseen = np.array(
 
            [c for c in range(36) if c not in known_cards],
 
            dtype=np.int8
 
        )
 
        cards_to_come = 5 - len(board)
 
        # Only the river table uses equity. Spend less if the time bank runs low.
        if not state["is_river"]:
            samples = 0
        elif memory.get("_time_used", 0.0) < 5.0:
            samples = 150
        else:
            samples = 40
 
        draw_size = 5 + cards_to_come
 
        rng = np.random.default_rng(
 
            obs.hand_index * 1000003
 
            + sum(hole) * 101
 
            + sum(board) * 17
 
            + len(board)
 
        )
 
        random_keys = rng.random((samples, len(unseen)))
 
        sampled_indices = np.argpartition(
 
            random_keys,
 
            draw_size - 1,
 
            axis=1
 
        )[:, :draw_size]
 
        draws = unseen[sampled_indices]
 
        wins = 0
 
        ties = 0
 
        losses = 0
 
        my_score = omaha_score(hole, board) if samples else 0
 
        for i in range(samples):
 
            opponent_hole = tuple(
 
                int(c) for c in draws[i, :5]
 
            )
 
            completed_board = board + tuple(
 
                int(c) for c in draws[i, 5:]
 
            )
 
 
            opponent_score = omaha_score(
 
                opponent_hole,
 
                completed_board
 
            )
 
            if my_score > opponent_score:
 
                wins += 1
 
            elif my_score == opponent_score:
 
                ties += 1
 
            else:
 
                losses += 1
 
        equity = (wins + 0.5 * ties) / samples if samples else None
 
        state["hand_abstraction"] = None
 
        state["equity"] = equity
 
        state["equity_wins"] = wins
 
        state["equity_ties"] = ties
 
        state["equity_losses"] = losses
 
        state["equity_samples"] = samples
 
    #** -------------------- 4. Adjust strategy -------------------- **
 
    update_opponent_stats(obs, memory)
    solver_mix = lookup_strategy(obs, state["equity"])
    state["strategy_probabilities"] = solver_mix
    state["adjusted_probabilities"] = adjust_strategy(solver_mix, obs, memory, state["equity"])
 
    #** -------------------- 5. Select legal action -------------------- **
 
    probabilities = state["adjusted_probabilities"]
    if probabilities is not None:
        return action_from_table(probabilities, obs)
 
    # No table entry at any level of detail.
    if obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.FOLD
 
def on_hand_end(result: HandResult, memory: dict) -> None:
 
    """Called once for every finished hand. Optional: learn from what happened.
 
    To save time, the judge sends finished hands along with your next
 
    decision request, so this always runs BEFORE choose_action() sees the
 
    next hand. (The last hands of a half arrive at the end of the half.)
 
    result has: hand_index, my_net (chips won or lost), pot,
 
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
 
    only after a showdown, else None), winner ('me', 'opp' or 'split').
 
    """
 
    memory["hands_played"] += 1
 
    opp, hand = memory["opp"], memory["hand"]
    opp["hands"] += 1
    opponent_folded = not result.went_to_showdown and result.winner == "me"
    if opponent_folded:
        opp["folds"] += 1
    if hand["raise_pending"]:
        opp["faced_our_raise"] += 1
        if opponent_folded:
            opp["folded_to_our_raise"] += 1
    if result.went_to_showdown:
        opp["showdowns"] += 1
        opp["showdown_wins"] += int(result.winner == "opp")
        if result.opp_hole is not None and len(result.board) == 5:
            opp["sd_category_counts"][category_of(omaha_score(result.opp_hole, result.board))] += 1

 # ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
 
    memory["hand"] = {"vpip": False, "pfr": False, "saw_flop": False, "actions": [], "seen": 0, "raise_pending": False}
