import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { type GeneralSettings } from "../../settings/generalSettings";

type GpuInfo = {
  available: boolean;
  name: string;
  backend: string;
  torch_cuda: boolean;
  nvidia_smi_found: boolean;
  details: string;
  torch_version?: string;
  cuda_version?: string;
};

type SceneDetectionSectionProps = {
  generalSettings: GeneralSettings;
  setGeneralSettings: React.Dispatch<React.SetStateAction<GeneralSettings>>;
};

export default function SceneDetectionSection({
  generalSettings,
  setGeneralSettings,
}: SceneDetectionSectionProps) {
  const [gpuInfo, setGpuInfo] = useState<GpuInfo | null>(null);
  const [loadingGpu, setLoadingGpu] = useState(true);
  const [installing, setInstalling] = useState(false);
  const [installResult, setInstallResult] = useState<{ success: boolean; message: string } | null>(null);

  const fetchGpuInfo = async () => {
    setLoadingGpu(true);
    try {
      const result = await invoke<string>("get_gpu_status");
      setGpuInfo(JSON.parse(result));
    } catch (e) {
      console.error("Failed to fetch GPU status", e);
    } finally {
      setLoadingGpu(false);
    }
  };

  useEffect(() => {
    fetchGpuInfo();
  }, []);

  const handleInstallCuda = async () => {
    if (!window.confirm("This will uninstall your current PyTorch and install the CUDA-enabled version. It may take several minutes. Continue?")) {
      return;
    }
    
    setInstalling(true);
    setInstallResult(null);

    const log = (message: string, type: "info" | "error" | "warn" = "info") => {
      window.dispatchEvent(new CustomEvent("amverge:dev_log", { 
        detail: { message, type } 
      }));
    };

    log("Starting PyTorch CUDA installation...", "info");
    log("This process will uninstall existing torch packages first.", "warn");
    
    try {
      const result = await invoke<string>("install_cuda_pytorch");
      const parsed = JSON.parse(result);
      setInstallResult(parsed);
      
      if (parsed.success) {
        log("Installation completed successfully!", "info");
        log(parsed.message, "info");
        await fetchGpuInfo();
      } else {
        log("Installation failed.", "error");
        log(parsed.message, "error");
      }
    } catch (e) {
      const errMsg = String(e);
      log("Critical error during installation: " + errMsg, "error");
      setInstallResult({ success: false, message: errMsg });
    } finally {
      setInstalling(false);
    }
  };

  const showInstallButton = gpuInfo && !gpuInfo.torch_cuda && gpuInfo.nvidia_smi_found;

  return (
    <section className="settings-section">
      <h3>Scene Detection</h3>
      {/* Primary Config Row */}
      <div className="settings-row">
        <label className="settings-label">Detection Method</label>
        <div className="settings-control">
          <select
            className="settings-select"
            value={generalSettings.sceneDetectionMethod}
            onChange={(e) =>
              setGeneralSettings((prev) => ({
                ...prev,
                sceneDetectionMethod: e.target.value as "amverge" | "transnetv2" | "omnishotcut" | "hybrid",
              }))
            }
            style={{ width: "240px" }}
          >
            <option value="amverge">AMVerge (Fast)</option>
            <option value="transnetv2">TransNet V2 (Accurate AI)</option>
            <option value="omnishotcut">OmniShotCut (Elite Transformer)</option>
            <option value="hybrid">Hybrid (Heuristic + AI Consensus)</option>
          </select>
        </div>
      </div>

      {/* Threshold / Sensitivity Slider - Shown for AI methods */}
      {generalSettings.sceneDetectionMethod !== "amverge" && (
        <div className="settings-row" style={{ marginTop: "-8px", marginBottom: "12px" }}>
          <label className="settings-label" style={{ fontSize: "0.85rem", opacity: 0.8 }}>Sensitivity (Threshold)</label>
          <div className="settings-control" style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={generalSettings.sceneDetectionThreshold}
              onChange={(e) => setGeneralSettings(prev => ({ ...prev, sceneDetectionThreshold: parseFloat(e.target.value) }))}
              className="settings-slider"
              style={{ width: "180px", accentColor: "#4CAF50" }}
            />
            <span style={{ 
              fontSize: "0.9rem", 
              fontWeight: "bold", 
              color: generalSettings.sceneDetectionThreshold < 0.3 ? "#ff5252" : (generalSettings.sceneDetectionThreshold > 0.7 ? "#ffeb3b" : "#4CAF50"),
              minWidth: "40px" 
            }}>
              {Math.round((1 - generalSettings.sceneDetectionThreshold) * 100)}%
            </span>
          </div>
        </div>
      )}
      <p style={{ fontSize: "0.85rem", opacity: 0.6, marginLeft: "24px", marginBottom: "24px", marginTop: "-4px", lineHeight: "1.4" }}>
        {generalSettings.sceneDetectionMethod === "amverge" 
          ? "Standard algorithm. Splits scenes quickly using keyframe analysis without needing AI models."
          : generalSettings.sceneDetectionMethod === "transnetv2"
          ? "Deep learning model for pixel-perfect anime scene detection. Lower sensitivity to catch more subtle cuts."
          : generalSettings.sceneDetectionMethod === "omnishotcut"
          ? "Shot-Query Transformer with relational reasoning. Highly recommended for complex anime transitions."
          : "Maximum accuracy mode. Combines heuristics with AI consensus for mission-critical scene splitting."}
        {generalSettings.sceneDetectionMethod !== "amverge" && (
          <span style={{ display: "block", marginTop: "4px", color: generalSettings.sceneDetectionThreshold < 0.3 ? "#ff5252" : "inherit", opacity: generalSettings.sceneDetectionThreshold < 0.3 ? 1 : 0.8 }}>
             {generalSettings.sceneDetectionThreshold < 0.3 ? "⚠️ High sensitivity may cause many false positives in high-motion scenes." : "Balance sensitivity based on the video complexity."}
          </span>
        )}
      </p>

      {/* Hardware Status Panel - Shown for AI methods */}
      {(generalSettings.sceneDetectionMethod !== "amverge") && (
        <div style={{ 
          marginTop: "12px",
          padding: "20px", 
          background: "rgba(255,255,255,0.03)", 
          border: "1px solid rgba(255,255,255,0.1)", 
          borderRadius: "12px",
          display: "flex",
          flexDirection: "column",
          gap: "20px"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "12px" }}>
            <span style={{ fontSize: "0.8rem", opacity: 0.5, textTransform: "uppercase", letterSpacing: "1px", fontWeight: "bold" }}>
              Hardware Configuration
            </span>
            {loadingGpu && <span style={{ fontSize: "0.8rem", opacity: 0.5 }}>Refreshing...</span>}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
            {/* Active Device */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.8rem", opacity: 0.5, fontWeight: "bold" }}>ACTIVE DEVICE</span>
              <span style={{ 
                fontSize: "1.2rem", 
                fontWeight: "bold",
                color: gpuInfo?.available ? "#4CAF50" : "#ff9800",
                lineHeight: "1.2"
              }}>
                {loadingGpu ? "..." : (gpuInfo?.name || "CPU Only")}
              </span>
            </div>

            {/* AI Backend */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.8rem", opacity: 0.5, fontWeight: "bold" }}>AI BACKEND</span>
              <span style={{ fontSize: "1.1rem", color: "#fff" }}>
                {loadingGpu ? "..." : (gpuInfo?.backend || "None")}
              </span>
            </div>

            {/* PyTorch Version */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.8rem", opacity: 0.5, fontWeight: "bold" }}>PYTORCH VERSION</span>
              <span style={{ fontSize: "1rem", color: "#fff", opacity: 0.9 }}>
                {loadingGpu ? "..." : (gpuInfo?.torch_version || "Not found")}
              </span>
            </div>

            {/* CUDA Status */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.8rem", opacity: 0.5, fontWeight: "bold" }}>CUDA SUPPORT</span>
              <span style={{ 
                fontSize: "1rem", 
                color: gpuInfo?.torch_cuda ? "#4CAF50" : "#ff9800",
                fontWeight: "bold"
              }}>
                {loadingGpu ? "..." : (gpuInfo?.torch_cuda ? "ENABLED" : "DISABLED")}
              </span>
            </div>
          </div>

          {/* Action / Warning Area */}
          {showInstallButton && (
            <div style={{ 
              marginTop: "8px",
              padding: "20px", 
              backgroundColor: "rgba(255, 152, 0, 0.08)", 
              borderRadius: "8px",
              border: "1px solid rgba(255, 152, 0, 0.2)",
              display: "flex",
              flexDirection: "column",
              gap: "16px"
            }}>
              <div>
                <div style={{ fontWeight: "bold", color: "#ff9800", marginBottom: "6px", fontSize: "1.1rem" }}>
                  NVIDIA GPU Detected (Unused)
                </div>
                <div style={{ fontSize: "0.9rem", opacity: 0.9, lineHeight: "1.5" }}>
                  An NVIDIA GPU was found, but your software is running on the CPU. 
                  Install CUDA support for up to 10x faster processing.
                </div>
              </div>
              <button 
                className="buttons" 
                onClick={handleInstallCuda}
                disabled={installing}
                style={{ 
                  margin: 0, 
                  width: "100%",
                  height: "44px",
                  fontSize: "1rem",
                  backgroundColor: "#ff9800", 
                  color: "#000",
                  fontWeight: "bold",
                  borderRadius: "4px"
                }}
              >
                {installing ? "Installing PyTorch (This takes 2-5 mins)..." : "Install CUDA PyTorch"}
              </button>
            </div>
          )}

          {installResult && (
            <div style={{ 
              padding: "12px 16px", 
              borderRadius: "6px",
              fontSize: "0.9rem",
              backgroundColor: installResult.success ? "rgba(76, 175, 80, 0.1)" : "rgba(244, 67, 54, 0.1)",
              color: installResult.success ? "#4CAF50" : "#f44336",
              border: `1px solid ${installResult.success ? "#4CAF50" : "#f44336"}`,
              lineHeight: "1.5"
            }}>
              {installResult.message}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
