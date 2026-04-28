import { useState, useRef, startTransition, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ClipItem, EpisodeEntry } from "../types/domain"
import { fileNameFromPath, truncateFileName, detectScenes } from "../utils/episodeUtils";
type ImportExportProps = {
  abortedRef: React.RefObject<boolean>;
  clips: ClipItem[];
  setFocusedClip: React.Dispatch<React.SetStateAction<string | null>>;
  setSelectedClips: React.Dispatch<React.SetStateAction<Set<string>>>;
  setVideoIsHEVC: React.Dispatch<React.SetStateAction<boolean | null>>;
  setImportedVideoPath: React.Dispatch<React.SetStateAction<string | null>>;
  setClips: React.Dispatch<React.SetStateAction<ClipItem[]>>;
  setEpisodes: React.Dispatch<React.SetStateAction<EpisodeEntry[]>>;
  setSelectedEpisodeId: React.Dispatch<React.SetStateAction<string | null>>;
  setOpenedEpisodeId: React.Dispatch<React.SetStateAction<string | null>>;
  selectedFolderId: string | null;
  EXPORT_DIR_STORAGE_KEY: string;
  exportDir: string | null;
  setExportDir: React.Dispatch<React.SetStateAction<string | null>>;
  setProgress: React.Dispatch<React.SetStateAction<number>>;
  setProgressMsg: React.Dispatch<React.SetStateAction<string>>;
  episodesPath: string | null;
  exportFormat: "mp4" | "mkv" | "mov" | "avi";
};

export default function useImportExport({
  setProgress,
  setProgressMsg,
  setSelectedClips,
  setFocusedClip,
  setImportedVideoPath,
  setVideoIsHEVC,
  setEpisodes,
  setSelectedEpisodeId,
  setOpenedEpisodeId,
  setClips,
  episodesPath,
  selectedFolderId,
  abortedRef,
  exportDir,
  setExportDir,
  exportFormat,
  clips,
}: ImportExportProps) {
  const [loading, setLoading] = useState(false);
  const [importToken, setImportToken] = useState(() => Date.now().toString());
  const importGenRef = useRef(0);
  const [batchTotal, setBatchTotal] = useState(0);
  const [batchDone, setBatchDone] = useState(0);
  const [batchCurrentFile, setBatchCurrentFile] = useState("");

  const handleImport = useCallback(async (file: string | null) => {
    // This opens the file dialog to select a video file
    if (!file) return;

    const episodeId = crypto.randomUUID();
    const gen = ++importGenRef.current;

    try {
      setProgress(0);
      setProgressMsg("Starting...");
      setLoading(true);
      setSelectedClips(new Set());
      setFocusedClip(null);
      setImportedVideoPath(file);
      setVideoIsHEVC(null);
      setImportToken(Date.now().toString());

      const formatted = await detectScenes(file, episodeId, episodesPath);

      // A newer import started while we were waiting - discard stale results.
      if (importGenRef.current !== gen) return;

      const inferredName = formatted[0]?.originalName || fileNameFromPath(file);

      const episodeEntry: EpisodeEntry = {
        id: episodeId,
        displayName: inferredName,
        videoPath: file,
        folderId: selectedFolderId,
        importedAt: Date.now(),
        clips: formatted,
      };

      setEpisodes((prev) => [episodeEntry, ...prev]);
      setSelectedEpisodeId(episodeId);
      setOpenedEpisodeId(episodeId);
      startTransition(() => {
        setClips(formatted);
      });
    } catch (err) {
      if (importGenRef.current !== gen) return;
      console.error("Detection failed:", err);
    } finally {
      if (importGenRef.current === gen) setLoading(false);
    }
  }, [setProgress, setProgressMsg, setSelectedClips, setFocusedClip, setImportedVideoPath, setVideoIsHEVC, episodesPath, selectedFolderId, setEpisodes, setSelectedEpisodeId, setOpenedEpisodeId, setClips]);

  const handleBatchImport = useCallback(async (files: string[]) => {
    const gen = ++importGenRef.current;
    abortedRef.current = false;

    const completedEpisodes: EpisodeEntry[] = [];

    try {
      setProgress(0);
      setProgressMsg("Starting...");
      setLoading(true);
      setSelectedClips(new Set());
      setFocusedClip(null);
      setVideoIsHEVC(null);
      setBatchTotal(files.length);
      setBatchDone(0);
      setBatchCurrentFile("");

      for (let i = 0; i < files.length; i++) {
        if (abortedRef.current) break;
        if (importGenRef.current !== gen) return;

        const file = files[i];
        const episodeId = crypto.randomUUID();
        const fileName = fileNameFromPath(file);

        setBatchDone(i);
        setBatchCurrentFile(truncateFileName(fileName));
        setProgress(0);
        setProgressMsg("Starting...");

        try {
          const formatted = await detectScenes(file, episodeId, episodesPath);

          if (abortedRef.current || importGenRef.current !== gen) {
            // Aborted or superseded mid-flight — clean up this episode's cache
            invoke("delete_episode_cache", {
              episodeCacheId: episodeId,
              customPath: episodesPath,
            }).catch(() => {});
            break;
          }

          const inferredName = formatted[0]?.originalName || fileNameFromPath(file);

          const episodeEntry: EpisodeEntry = {
            id: episodeId,
            displayName: inferredName,
            videoPath: file,
            folderId: selectedFolderId,
            importedAt: Date.now(),
            clips: formatted,
          };

          completedEpisodes.push(episodeEntry);
          setEpisodes((prev) => [episodeEntry, ...prev]);
        } catch (err) {
          if (abortedRef.current) {
            invoke("delete_episode_cache", {
              episodeCacheId: episodeId,
              customPath: episodesPath,
            }).catch(() => {});
            break;
          }
          console.error(`Detection failed for ${fileName}:`, err);
          invoke("delete_episode_cache", {
            episodeCacheId: episodeId,
            customPath: episodesPath,
          }).catch(() => {});
        }
      }

      // Open the first completed episode
      if (completedEpisodes.length > 0 && importGenRef.current === gen) {
        const first = completedEpisodes[0];
        setSelectedEpisodeId(first.id);
        setOpenedEpisodeId(first.id);
        setImportedVideoPath(first.videoPath);
        setImportToken(Date.now().toString());
        startTransition(() => {
          setClips(first.clips);
        });
      }
    } finally {
      if (importGenRef.current === gen) {
        setLoading(false);
        setBatchTotal(0);
        setBatchDone(0);
        setBatchCurrentFile("");
      }
    }
  }, [abortedRef, setProgress, setProgressMsg, setSelectedClips, setFocusedClip, setVideoIsHEVC, episodesPath, selectedFolderId, setEpisodes, setSelectedEpisodeId, setOpenedEpisodeId, setImportedVideoPath, setClips]);

  const onImportClick = useCallback(async () => {
    const files = await open({
      multiple: true,
      filters: [
        {
          name: "Video",
          extensions: ["mp4", "mkv", "mov", "avi"]
        }
      ]
    });

    if (!files) return;

    // open() with multiple:true returns string[] | null
    const fileList = Array.isArray(files) ? files : [files];
    if (fileList.length === 0) return;

    if (fileList.length === 1) {
      handleImport(fileList[0]);
    } else {
      handleBatchImport(fileList);
    }
  }, [handleImport, handleBatchImport]);

  const handleExport = useCallback(async(selectedClips: Set<string>, mergeEnabled: boolean, mergeFileName?: string) => {
    if (selectedClips.size === 0) return;

    const selected = clips.filter((c: ClipItem) => selectedClips.has(c.id));
    if (selected.length === 0) return;

    // If no export directory is set, prompt the user to pick one first
    let dir = exportDir;
    if (!dir) {
        const picked = await open({ directory: true, multiple: false });
        if (!picked) return;
        dir = picked as string;
        setExportDir(dir);
    }

    try {
        setLoading(true);

        const clipArray = selected.map((c: ClipItem) => c.src);
        const format = exportFormat || "mp4";

        if (mergeEnabled) {
        const baseName = mergeFileName || ((selected[0]?.originalName || "episode") + "_merged");
        const savePath = `${dir}\\${baseName}.${format}`;

        await invoke("export_clips", {
            clips: clipArray,
            savePath: savePath,
            mergeEnabled: mergeEnabled,
        });
        } else {
        const firstClipPath = selected[0]?.src || "";
        const firstFile = firstClipPath.split(/[/\\]/).pop() || `episode_0000.${format}`;
        const firstStem = firstFile.replace(/\.[^/.]+$/, "");
        const defaultBase = firstStem.replace(/_\d{4}$/, "");
        const savePath = `${dir}\\${defaultBase}_####.${format}`;

        await invoke("export_clips", {
            clips: clipArray,
            savePath: savePath,
            mergeEnabled: false,
        });
        }
        
        console.log("Export complete");
    } catch (err) {
        console.log("Export failed:", err)
    } finally {
        setLoading(false);
    }
  }, [clips, exportDir, exportFormat, setExportDir]);

  const handlePickExportDir = useCallback(async () => {
    const dir = await open({ directory: true, multiple: false });
    if (dir) setExportDir(dir as string);
  }, [setExportDir]);

  return {
    loading,
    importToken,
    setImportToken,
    batchTotal,
    batchDone,
    batchCurrentFile,
    onImportClick,
    handleImport,
    handleExport,
    handlePickExportDir,
    handleBatchImport
  };
}