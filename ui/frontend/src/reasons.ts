// backend reason codes said in words, in one place
//
// the codes come from src/common/bidding_formulas.py and
// src/common/reverse_auction.py. the wording follows src/api/infeasibility.py
// DECLINE_WORDS so a decline reads the same in the log, in the round table and
// inside the backend's own infeasibility summary
//
// short  fits a narrow table cell
// phrase completes "M08 stayed out, it ..." in a log line

interface Reason {
    short: string;
    phrase: string;
}

const REASONS: Record<string, Reason> = {
    material_not_supplied: { short: "wrong material", phrase: "does not carry that material" },
    product_not_made: { short: "wrong product", phrase: "does not make that product" },
    region_not_serviced: { short: "region not served", phrase: "does not deliver to that city" },
    defect_probability_too_high: { short: "defect rate high", phrase: "has a defect rate above the limit" },
    damage_probability_too_high: { short: "damage rate high", phrase: "has a damage rate above the limit" },
    reputation_too_low: { short: "reputation low", phrase: "has a reputation below the minimum" },
    insufficient_stock_and_restock: { short: "not enough stock", phrase: "has not enough stock even with a restock" },
    completion_exceeds_deadline: { short: "past deadline", phrase: "could not finish before the deadline" },
    workload_ratio_above_one: { short: "fully booked", phrase: "is already fully booked" },
    no_daily_capacity: { short: "no capacity", phrase: "has no daily capacity left" },
    slm_no_bid: { short: "model declined", phrase: "was turned down by the model with no rule reason" },
    withdrawn_in_earlier_round: { short: "withdrawn", phrase: "had already dropped out of the auction" },
    next_bid_below_minimum_bid: { short: "at its floor", phrase: "cannot go lower without breaking its price floor" },
    no_current_best_bid: { short: "nothing to beat", phrase: "had no announced best bid to undercut" },
    never_entered: { short: "never entered", phrase: "never entered the auction" },
    withdrawn: { short: "withdrawn", phrase: "dropped out of the auction" },
};

/** an agent's own guard fires under a prefix, and a crash carries its text */
function unwrap(code: string): { code: string; detail: string | null } {
    if (code.startsWith("dynamic_guard:")) return { code: code.slice(14), detail: null };
    if (code.startsWith("decision_failed:")) return { code: "decision_failed", detail: code.slice(16).trim() };
    return { code, detail: null };
}

/** last resort for a code nobody has words for yet: never show the raw
 * underscores, but never invent a meaning either */
function plain(code: string): string {
    return code.replaceAll("_", " ");
}

export function reasonShort(code: string | null | undefined): string {
    if (!code) return "declined";
    const { code: key, detail } = unwrap(code);
    if (key === "decision_failed") return "agent error";
    void detail;
    return REASONS[key]?.short ?? plain(key);
}

export function reasonPhrase(code: string | null | undefined): string {
    if (!code) return "gave no reason";
    const { code: key, detail } = unwrap(code);
    if (key === "decision_failed") return `hit an error while deciding${detail ? ` (${detail})` : ""}`;
    return REASONS[key]?.phrase ?? plain(key);
}
