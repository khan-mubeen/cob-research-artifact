import { useState } from "react";
import LiveAuctionView from "./components/LiveAuctionView";
import DashboardView from "./components/dashboard/DashboardView";

type Tab = "live" | "dashboard";

export default function App() {
    const [tab, setTab] = useState<Tab>("live");

    return (
        <div className="app">
            <header className="app-header">
                <h1>Chain of Bid</h1>
                <nav className="tabs">
                    <button className={tab === "live" ? "active" : ""} onClick={() => setTab("live")}>
                        Live Auction
                    </button>
                    <button className={tab === "dashboard" ? "active" : ""} onClick={() => setTab("dashboard")}>
                        Evaluation Dashboard
                    </button>
                </nav>
            </header>

            <main className="app-main">
                {tab === "live" ? <LiveAuctionView /> : <DashboardView />}
            </main>
        </div>
    );
}
