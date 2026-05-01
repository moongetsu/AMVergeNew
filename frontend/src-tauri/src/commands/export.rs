use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use tauri::{AppHandle, Emitter};

use crate::payloads::ProgressPayload;
use crate::utils::ffmpeg::resolve_bundled_tool;
use crate::utils::logging::console_log;
use crate::utils::paths::file_name_only;
use crate::utils::process::apply_no_window;

#[derive(serde::Deserialize)]
pub struct ExportSegment {
    pub source_path: String,
    pub source_start: f64,
    pub duration: f64,
}

#[tauri::command]
pub async fn export_clips(
    app: AppHandle,
    segments: Vec<ExportSegment>,
    save_path: String,
    merge_enabled: bool,
) -> Result<(), String> {
    if segments.is_empty() {
        return Ok(());
    }

    console_log(
        "EXPORT|start",
        &format!(
            "merge_enabled={} segments={} dest={}",
            merge_enabled,
            segments.len(),
            file_name_only(&save_path)
        ),
    );

    let ffmpeg = resolve_bundled_tool(&app, "ffmpeg")?;
    let mut save_path = PathBuf::from(&save_path);
    let export_start_time = Instant::now();

    if save_path.extension().is_none() {
        save_path.set_extension("mp4");
    }

    if let Some(parent) = save_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    fn format_elapsed(start_time: Instant) -> String {
        let secs = start_time.elapsed().as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        if h > 0 {
            format!("{:02}:{:02}:{:02}", h, m, s)
        } else {
            format!("{:02}:{:02}", m, s)
        }
    }

    fn emit_export_progress(app: &AppHandle, percent: u8, message: &str, start_time: Instant) {
        let p = percent.min(100);
        let msg = format!(
            "{} ({} elapsed)",
            message.replace('\n', " ").replace('\r', " "),
            format_elapsed(start_time)
        );

        let _ = app.emit(
            "scene_progress",
            ProgressPayload {
                percent: p,
                message: msg.clone(),
            },
        );
    }

    let out_str = save_path.to_string_lossy().to_string();

    if merge_enabled {
        // ---------------- MERGED EXPORT (Single Pass) ----------------
        let mut args = vec!["-y".to_string()];
        let mut filter_input = String::new();
        
        for (i, seg) in segments.iter().enumerate() {
            args.push("-ss".to_string());
            args.push(seg.source_start.to_string());
            args.push("-t".to_string());
            args.push(seg.duration.to_string());
            args.push("-i".to_string());
            args.push(seg.source_path.clone());
            filter_input.push_str(&format!("[{}:v][{}:a]", i, i));
        }

        let filter_complex = format!(
            "{}concat=n={}:v=1:a=1[v_raw][a];[v_raw]format=yuv420p[v]",
            filter_input,
            segments.len()
        );

        args.extend([
            "-filter_complex".to_string(),
            filter_complex,
            "-map".to_string(), "[v]".to_string(),
            "-map".to_string(), "[a]".to_string(),
            "-c:v".to_string(), "libx264".to_string(),
            "-crf".to_string(), "17".to_string(),
            "-preset".to_string(), "veryfast".to_string(),
            "-c:a".to_string(), "aac".to_string(),
            "-b:a".to_string(), "192k".to_string(),
            "-movflags".to_string(), "+faststart".to_string(),
            out_str,
        ]);

        let app_for_ffmpeg = app.clone();
        let ffmpeg_clone = ffmpeg.clone();
        let start_time = export_start_time;
        
        // Helper function for progress parsing
        fn run_ffmpeg_with_progress_basic(
            app: AppHandle,
            ffmpeg: PathBuf,
            mut args: Vec<String>,
            message_prefix: &str,
            start_time: Instant,
        ) -> Result<(), String> {
            args.insert(0, "-hide_banner".into());
            args.insert(0, "-nostats".into());
            args.insert(0, "pipe:2".into());
            args.insert(0, "-progress".into());

            let mut cmd = Command::new(&ffmpeg);
            apply_no_window(&mut cmd);
            let mut child = cmd
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("Failed to spawn ffmpeg: {e}"))?;

            let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;
            let reader = BufReader::new(stderr);

            for line in reader.lines().flatten() {
                if line.starts_with("out_time_ms=") {
                    let elapsed = start_time.elapsed();
                    let secs = elapsed.as_secs();
                    let elapsed_str = format!("{:02}:{:02}", secs / 60, secs % 60);
                    let _ = app.emit("scene_progress", ProgressPayload {
                        percent: 50, // Static 50% during the merge
                        message: format!("{} ({} elapsed)", message_prefix, elapsed_str),
                    });
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("FFmpeg failed".into());
            }
            Ok(())
        }

        let result = tokio::task::spawn_blocking(move || {
            run_ffmpeg_with_progress_basic(
                app_for_ffmpeg,
                ffmpeg_clone,
                args,
                "Exporting",
                start_time,
            )
        })
        .await
        .map_err(|e| format!("ffmpeg task panicked: {e}"))?;

        if let Err(e) = result {
            return Err(format!("FFmpeg export failed: {e}"));
        }

    } else {
        // ---------------- MULTIPLE EXPORT ----------------
        let destination_dir = save_path.parent().ok_or("Invalid save path")?;
        let user_stem = save_path
            .file_stem()
            .ok_or("Invalid filename")?
            .to_string_lossy()
            .to_string();

        let ext = save_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp4")
            .to_string();

        for (i, seg) in segments.iter().enumerate() {
            let code = format!("{:04}", i + 1);
            let file_stem = if user_stem.contains("####") {
                user_stem.replace("####", &code)
            } else {
                format!("{}_{}", user_stem, code)
            };

            let destination = destination_dir.join(format!("{}.{}", file_stem, ext));
            let output_str = destination.to_string_lossy().to_string();

            let msg = format!("Exporting clip {}/{}", i + 1, segments.len());
            emit_export_progress(&app, (i * 100 / segments.len()) as u8, &msg, export_start_time);

            let mut cmd = std::process::Command::new(&ffmpeg);
            apply_no_window(&mut cmd);
            
            let result = cmd.args([
                "-y",
                "-ss", &seg.source_start.to_string(),
                "-t", &seg.duration.to_string(),
                "-i", &seg.source_path,
                "-c:v", "libx264",
                "-crf", "17",
                "-preset", "veryfast",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                &output_str
            ]).output().map_err(|e| format!("Clip {} failed: {e}", i+1))?;

            if !result.status.success() {
                return Err(format!("FFmpeg failed on clip {}: {}", i+1, String::from_utf8_lossy(&result.stderr)));
            }
        }
    }

    emit_export_progress(&app, 100, "Export complete", export_start_time);
    console_log("EXPORT|end", "ok");

    Ok(())
}

#[tauri::command]
pub async fn fast_merge(
    app: AppHandle,
    clips: Vec<String>,
    output_path: String,
) -> Result<String, String> {
    if clips.is_empty() {
        return Err("No clips to merge".into());
    }

    let ffmpeg = resolve_bundled_tool(&app, "ffmpeg")?;
    
    let mut cmd = std::process::Command::new(&ffmpeg);
    apply_no_window(&mut cmd);
    
    let mut filter_input = String::new();
    let mut args = vec!["-y".to_string()];
    
    for (i, c) in clips.iter().enumerate() {
        args.push("-i".to_string());
        args.push(c.clone());
        filter_input.push_str(&format!("[{}:v][{}:a]", i, i));
    }
    
    let filter_complex = format!("{}concat=n={}:v=1:a=1[v_raw][a];[v_raw]format=yuv420p[v]", filter_input, clips.len());
    
    args.extend([
        "-filter_complex".to_string(),
        filter_complex,
        "-map".to_string(), "[v]".to_string(),
        "-map".to_string(), "[a]".to_string(),
        "-c:v".to_string(), "libx264".to_string(),
        "-crf".to_string(), "17".to_string(),
        "-preset".to_string(), "veryfast".to_string(),
        "-c:a".to_string(), "aac".to_string(),
        "-b:a".to_string(), "192k".to_string(),
        "-movflags".to_string(), "+faststart".to_string(),
        output_path.clone(),
    ]);

    let result = cmd
        .args(&args)
        .output()
        .map_err(|e| format!("FFmpeg merge failed: {e}"))?;

    if !result.status.success() {
        return Err(format!("FFmpeg merge failed: {}", String::from_utf8_lossy(&result.stderr)));
    }

    Ok(output_path)
}

#[tauri::command]
pub async fn fast_split(
    app: AppHandle,
    input_path: String,
    split_time: f64,
    output_path1: String,
    output_path2: String,
    thumb_path2: String,
) -> Result<(), String> {
    let ffmpeg = resolve_bundled_tool(&app, "ffmpeg")?;

    console_log("SPLIT", &format!("input={} split_at={:.2}s", file_name_only(&input_path), split_time));

    // Part 1: [0 -> split_time]
    let mut cmd1 = Command::new(&ffmpeg);
    apply_no_window(&mut cmd1);
    let out1 = cmd1.args([
        "-y",
        "-i", &input_path,
        "-t", &split_time.to_string(),
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        &output_path1
    ]).output().map_err(|e| format!("Part 1 failed: {e}"))?;

    if !out1.status.success() {
        return Err(format!("FFmpeg Part 1 failed: {}", String::from_utf8_lossy(&out1.stderr)));
    }

    // Part 2: [split_time -> end]
    let mut cmd2 = Command::new(&ffmpeg);
    apply_no_window(&mut cmd2);
    let out2 = cmd2.args([
        "-y",
        "-ss", &split_time.to_string(),
        "-i", &input_path,
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        &output_path2
    ]).output().map_err(|e| format!("Part 2 failed: {e}"))?;

    if !out2.status.success() {
        return Err(format!("FFmpeg Part 2 failed: {}", String::from_utf8_lossy(&out2.stderr)));
    }

    // Thumbnail for Part 2
    let mut cmd3 = Command::new(&ffmpeg);
    apply_no_window(&mut cmd3);
    let _ = cmd3.args([
        "-y",
        "-i", &output_path2,
        "-vf", "thumbnail",
        "-frames:v", "1",
        &thumb_path2
    ]).output();

    Ok(())
}