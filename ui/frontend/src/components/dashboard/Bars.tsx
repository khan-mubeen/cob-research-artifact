interface BarItem {
    label: string;
    value: number;
    color?: string;
}

interface Props {
    items: BarItem[];
    max?: number;
    unit?: string;
}

const DEFAULT_COLOR = "#3b82f6";

export default function Bars({ items, max = 100, unit = "%" }: Props) {
    return (
        <div className="bar-chart">
            {items.map((item) => (
                <div className="bar-row" key={item.label}>
                    <span className="bar-label">{item.label}</span>
                    <div className="bar-track">
                        <div
                            className="bar-fill"
                            style={{
                                width: `${Math.max(0, Math.min(100, (item.value / max) * 100))}%`,
                                background: item.color ?? DEFAULT_COLOR,
                            }}
                        />
                    </div>
                    <span className="bar-value">
                        {item.value.toFixed(1)}
                        {unit}
                    </span>
                </div>
            ))}
        </div>
    );
}
