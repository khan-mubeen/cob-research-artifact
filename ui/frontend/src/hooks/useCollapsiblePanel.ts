import { useCallback, useState } from "react";

/** a side panel that remembers whether it was left open
 *
 * the same localStorage habit the column widths already have, so a presenter
 * who collapsed the log keeps it collapsed across a reload */
export function useCollapsiblePanel(key: string, initial = false) {
    const [collapsed, setCollapsed] = useState(() => {
        try {
            const stored = localStorage.getItem(key);
            return stored === null ? initial : stored === "1";
        } catch {
            // private mode or a full quota: the panel still works, it just
            // will not be remembered
            return initial;
        }
    });

    const toggle = useCallback(() => {
        setCollapsed((was) => {
            const next = !was;
            try {
                localStorage.setItem(key, next ? "1" : "0");
            } catch {
                // same as above, remembering is the optional part
            }
            return next;
        });
    }, [key]);

    return { collapsed, toggle };
}
