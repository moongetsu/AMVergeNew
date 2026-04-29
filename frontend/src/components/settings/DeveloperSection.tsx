import { useEffect, useState, useRef } from "react";

type DeveloperSectionProps = {
  // We can pass logs here if we store them globally, 
  // or just use this for real-time capturing later.
};

export default function DeveloperSection({}: DeveloperSectionProps) {
  const [logs, setLogs] = useState<{ id: number; message: string; type: "info" | "error" | "warn" }[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // For now, let's just listen to a custom event or similar for logs
  useEffect(() => {
    const handleNewLog = (event: any) => {
      const { message, type } = event.detail;
      setLogs((prev) => [...prev, { id: Date.now(), message, type }]);
    };

    window.addEventListener("amverge:dev_log", handleNewLog);
    return () => window.removeEventListener("amverge:dev_log", handleNewLog);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <section className="settings-section">
      <h3>Developer Console</h3>
      <p style={{ fontSize: "0.85rem", opacity: 0.6, marginBottom: "16px" }}>
        Live installation logs and internal process monitoring.
      </p>

      <div 
        ref={scrollRef}
        style={{ 
          background: "#000", 
          border: "1px solid rgba(255,255,255,0.1)", 
          borderRadius: "8px", 
          padding: "16px", 
          height: "300px", 
          overflowY: "auto",
          fontFamily: "'Courier New', Courier, monospace",
          fontSize: "0.8rem",
          display: "flex",
          flexDirection: "column",
          gap: "4px"
        }}
      >
        {logs.length === 0 && (
          <span style={{ opacity: 0.3 }}>No logs recorded in this session.</span>
        )}
        {logs.map((log) => (
          <div key={log.id} style={{ 
            color: log.type === "error" ? "#ff5555" : log.type === "warn" ? "#ffaa00" : "#aaffaa",
            whiteSpace: "pre-wrap",
            wordBreak: "break-all"
          }}>
            <span style={{ opacity: 0.4, marginRight: "8px" }}>[{new Date(log.id).toLocaleTimeString()}]</span>
            {log.message}
          </div>
        ))}
      </div>

      <div style={{ marginTop: "12px", display: "flex", justifyContent: "flex-end" }}>
        <button 
          className="buttons" 
          onClick={() => setLogs([])}
          style={{ fontSize: "0.75rem", padding: "4px 12px", height: "auto" }}
        >
          Clear Console
        </button>
      </div>
    </section>
  );
}
