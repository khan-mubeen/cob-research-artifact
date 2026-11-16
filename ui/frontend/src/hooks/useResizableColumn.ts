import { useCallback, useEffect, useRef, useState } from "react";

export function useResizableColumn(key: string, initial: number, min: number, max: number, side: "left" | "right") {
    const [width, setWidth] = useState(() => {
        try {
            const saved = Number(localStorage.getItem(key));
            return Number.isFinite(saved) && saved >= min && saved <= max ? saved : initial;
        } catch {
            return initial;
        }
    });
    const widthRef = useRef(width);
    const [dragging, setDragging] = useState(false);

    useEffect(() => {
        widthRef.current = width;
        try {
            localStorage.setItem(key, String(width));
        } catch {
            // private mode or a full quota: the column still works, it just
            // will not be remembered
        }
    }, [width, key]);

    const onMouseDown = useCallback(
        (e: React.MouseEvent) => {
            e.preventDefault();
            const startX = e.clientX;
            const startWidth = widthRef.current;
            setDragging(true);

            const onMove = (ev: MouseEvent) => {
                const delta = ev.clientX - startX;
                const next = side === "left" ? startWidth + delta : startWidth - delta;
                setWidth(Math.min(max, Math.max(min, next)));
            };
            const onUp = () => {
                setDragging(false);
                window.removeEventListener("mousemove", onMove);
                window.removeEventListener("mouseup", onUp);
            };
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup", onUp);
        },
        [side, min, max],
    );

    return { width, dragging, onMouseDown };
}
