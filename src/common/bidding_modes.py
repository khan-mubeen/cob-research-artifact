"""which bidding mode a process runs, and the reverse-auction constants

BIDDING_MODE selects the mode. sealed is the default so an unset environment
behaves exactly as the sealed-bid deployment always has

the round count and the platform minimum decrement live here because both sides
of the protocol, supervisor and agents, must read the same values
"""

import os


SEALED = "sealed"
REVERSE_AUCTION = "reverse_auction"
VALID_MODES = (SEALED, REVERSE_AUCTION)

# the cap on rounds, from the reverse auction rules in agent_logic_design.md.
# an auction now also closes early when a round brings no improvement, so this is
# a maximum rather than a fixed count
AUCTION_ROUNDS = 3

# fixed eur amount, configurable for later experiments
DEFAULT_PLATFORM_MINIMUM_DECREMENT = 5.0


def bidding_mode():
    """return the mode this process runs

    an unknown value raises instead of falling back, a typo must not silently
    run the wrong experiment"""
    mode = os.getenv("BIDDING_MODE", "").strip().lower() or SEALED
    if mode not in VALID_MODES:
        raise RuntimeError(
            f"BIDDING_MODE must be one of {VALID_MODES}, got {mode!r}"
        )
    return mode


def reverse_auction_enabled():
    return bidding_mode() == REVERSE_AUCTION


def auction_max_rounds():
    """the round cap, AUCTION_ROUNDS unless the environment raises or lowers it"""
    raw = os.getenv("AUCTION_MAX_ROUNDS", "").strip()
    if not raw:
        return AUCTION_ROUNDS
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"AUCTION_MAX_ROUNDS must be an integer, got {raw!r}")
    if value < 1:
        raise RuntimeError(f"AUCTION_MAX_ROUNDS must be at least 1, got {value}")
    return value


def platform_minimum_decrement():
    """the eur step every round after the first must undercut the best bid by"""
    raw = os.getenv("PLATFORM_MINIMUM_DECREMENT", "").strip()
    if not raw:
        return DEFAULT_PLATFORM_MINIMUM_DECREMENT
    try:
        value = float(raw)
    except ValueError:
        raise RuntimeError(f"PLATFORM_MINIMUM_DECREMENT must be a number, got {raw!r}")
    if value <= 0:
        raise RuntimeError(f"PLATFORM_MINIMUM_DECREMENT must be positive, got {value}")
    return value


def is_auction_rfq(rfq):
    """an rfq belongs to the reverse auction when it carries a round number

    sealed rfqs never carry one, so this also detects a supervisor and an agent
    running different modes against the same broker"""
    return "round_number" in rfq
