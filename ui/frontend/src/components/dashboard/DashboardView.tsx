import { useEffect, useState } from "react";
import ReverseAuctionDashboard from "./ReverseAuctionDashboard";
import Epoch2Dashboard from "./Epoch2Dashboard";
import type { DashboardData } from "../../types";

type SubTab = "reverse_auction" | "epoch2";

export default function DashboardView() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [subTab, setSubTab] = useState<SubTab>("reverse_auction");

    useEffect(() => {
        fetch("/data/dashboard.json")
            .then((r) => r.json())
            .then(setData)
            .catch(() => setData(null));
    }, []);

    if (!data) {
        return (
            <div className="panel">
                <p className="empty">Loading evaluation data…</p>
            </div>
        );
    }

    return (
        <div className="dashboard-view">
            <div className="subtabs">
                <button className={subTab === "reverse_auction" ? "active" : ""} onClick={() => setSubTab("reverse_auction")}>
                    Reverse Auction (v4)
                </button>
                <button className={subTab === "epoch2" ? "active" : ""} onClick={() => setSubTab("epoch2")}>
                    Epoch-2 Production
                </button>
            </div>
            {subTab === "reverse_auction" ? (
                <ReverseAuctionDashboard entries={data.reverse_auction} />
            ) : (
                <Epoch2Dashboard entries={data.epoch2_production} />
            )}
        </div>
    );
}
