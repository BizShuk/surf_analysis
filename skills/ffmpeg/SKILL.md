---
name: ffmpeg
description: >
    Use when the user wants to transcode, trim, filter, stream, probe, or debug
    video/audio with FFmpeg from the command line. Triggers on "ffmpeg", "convert
    video", "extract audio", "make HLS", "RTMP stream", "trim clip", "video
    filter", "hardware encode nvenc/videotoolbox/qsv/vaapi", "ffprobe",
    "faststart", "CRF", "two-pass", "moov atom", "why is ffmpeg failing".
    Do NOT invoke for GUI tools (HandBrake, VLC) or non-FFmpeg encoders (x264 CLI).
version: "1.1.0"
allowed-tools: Bash, Read, WebSearch
user-invocable: false
disable-model-invocation: false
effort: low
context: fork
metadata:
    type: reference
    platforms: [macos, linux]
    verified: 2026-06-10
    verified_against:
        [github.com/FFmpeg/FFmpeg Changelog release/5.0, 5.1, 6.0, 6.1]
---

# FFmpeg

Comprehensive FFmpeg command-line reference for LLM agents. Verified against
official FFmpeg Changelog for versions 4.4 through 6.1. Version notes have been
audited — see §5 for the corrected section (original synthesis contained
fabricated version claims; this version replaces them with Changelog-anchored
facts).

## When to use

- Transcode / re-encode video or audio
- Convert container (e.g. `.mov` → `.mp4`) losslessly or with re-encode
- Trim / cut clips with `-ss` / `-t`
- Apply filters (scale, crop, overlay, drawtext, concat, deinterlace)
- Stream to RTMP / HLS / SRT
- Use hardware encoders (NVENC, VideoToolbox, QSV, VAAPI)
- Inspect a media file with `ffprobe`
- Diagnose a failed FFmpeg command

When NOT to use:

- GUI transcoders (HandBrake, VLC, Permute) — refer user to those
- Server-side streaming infrastructure setup (nginx-rtmp, SRS) — out of scope
- Real-time playback tuning — use ffplay, not ffmpeg

---

## 1. Decision trees

### 1.1 Should I re-encode, or use `-c copy`?

```
input analysis
    │
    ├─ only changing container (mkv → mp4), no stream modification?
    │       └─ YES → -c copy  (lossless, instant)
    │
    ├─ only trimming (no re-encode needed)?
    │       └─ YES → -c copy + -ss / -t
    │                  (note: trim points snap to keyframes, ~±0.5s accuracy)
    │
    ├─ need to modify video (filter, scale, color, crop)?
    │       └─ YES → re-encode with -c:v <codec>
    │
    ├─ need to modify audio (resample, mix, normalize)?
    │       └─ YES → re-encode with -c:a <codec>
    │
    └─ target container does not support source codec?
            └─ YES → re-encode
```

### 1.2 Codec selection

```
use case
    │
    ├─ web playback (H.264 / MP4) → libx264 + -movflags +faststart
    │
    ├─ archive, smaller file (H.265 / HEVC) → libx265, crf 28-32
    │
    ├─ open / royalty-free (VP9) → libvpx-vp9, crf 30-40
    │
    ├─ latest compression (AV1) → libsvtav1 (faster) or libaom-av1 (smaller)
    │
    ├─ editing / post-production (visually lossless) → prores_ks profile 3
    │
    ├─ NVIDIA GPU → h264_nvenc / hevc_nvenc / av1_nvenc
    │
    ├─ Intel QSV → h264_qsv / hevc_qsv / av1_qsv
    │
    ├─ macOS GPU → h264_videotoolbox / hevc_videotoolbox
    │
    └─ Linux VAAPI → h264_vaapi / hevc_vaapi / av1_vaapi
```

### 1.3 Rate-control mode

```
priority
    │
    ├─ fixed file size (streaming cap) → -b:v <rate>
    │       (use -minrate, -maxrate, -bufsize for CBR-ish behavior)
    │
    ├─ best quality, size flexible (archive) → -crf 18-28
    │       (lower = better quality, larger file)
    │
    ├─ target size, quality flexible → two-pass: -pass 1 / -pass 2
    │
    └─ general purpose → -crf 23 -preset medium (defaults, usually fine)
```

### 1.4 Filter syntax: `-vf` vs `-filter_complex`

```
scenario
    │
    ├─ single input, single output, simple chain?
    │       └─ YES → -vf "filter1,filter2,filter3"
    │                  (each filter operates on the previous output)
    │
    ├─ multiple inputs OR multiple outputs OR need to route streams?
    │       └─ YES → -filter_complex "[0:v]scale=...[v];[1:v]overlay=...[v2];[v2][v]concat=...[out]"
    │                  (use labels [in]...[out])
    │
    └─ only one input, only audio filter?
            └─ -af "volume=2.0,aresample=48000"
```

---

## 2. Command templates

### 2.1 Lossless container conversion (no re-encode)

```bash
# Pure stream copy
ffmpeg -i input.mov -c copy output.mp4

# Stream copy but transcode audio (e.g. PCM in MOV → AAC for browser)
ffmpeg -i input.mov -c:v copy -c:a aac output.mp4
```

### 2.2 Re-encode video, keep audio

```bash
# H.264 CRF 23, medium preset
ffmpeg -i input.mkv -c:v libx264 -preset medium -crf 23 -c:a copy output.mp4
```

### 2.3 Re-encode both

```bash
ffmpeg -i input.avi -c:v libx264 -preset medium -crf 23 \
       -c:a aac -b:a 192k output.mp4
```

### 2.4 Two-pass (target file size)

```bash
# Pass 1: analysis (writes stats, no output file)
ffmpeg -i input.mp4 -c:v libx264 -preset medium -b:v 5000k \
       -pass 1 -an -f null -

# Pass 2: encode (same -b:v, reads stats)
ffmpeg -i input.mp4 -c:v libx264 -preset medium -b:v 5000k \
       -pass 2 -c:a aac -b:a 128k output.mp4
```

### 2.5 Hardware encoders

```bash
# NVIDIA NVENC
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -rc vbr -cq 23 -c:a copy output.mp4

# macOS VideoToolbox
ffmpeg -i input.mp4 -c:v h264_videotoolbox -b:v 5000k -c:a copy output.mp4

# Intel QSV
ffmpeg -i input.mp4 -c:v h264_qsv -preset medium -global_quality 23 -c:a copy output.mp4

# Linux VAAPI (basic usage — see §3.8 for caveats)
ffmpeg -hwaccel vaapi -hwaccel_output_format vaapi \
       -i input.mp4 -c:v h264_vaapi -qp 23 output.mp4
```

### 2.6 Trim / cut

```bash
# Fast trim, keyframe-snapped (no re-encode)
ffmpeg -ss 00:01:00 -i input.mkv -t 30 -c copy output.mkv

# Frame-accurate trim (re-encodes; slower)
ffmpeg -i input.mkv -ss 00:01:00 -t 30 -c:v libx264 -c:a aac output.mp4
```

### 2.7 Filter chain

```bash
# Single-chain: scale + deinterlace + speed-up
ffmpeg -i input.mp4 -vf "scale=1280:720,yadif=0:-1:0,setpts=PTS/1.5" \
       -c:v libx264 -c:a copy output.mp4

# Multi-input overlay
ffmpeg -i main.mp4 -i logo.png \
       -filter_complex "[0:v][1:v]overlay=10:10" \
       -c:v libx264 -c:a copy output.mp4
```

### 2.8 Streaming output

```bash
# HLS (HTTP Live Streaming)
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -f hls \
       -hls_time 6 -hls_list_size 0 \
       -hls_segment_filename "segment_%03d.ts" \
       output.m3u8

# RTMP
ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -b:v 4000k \
       -c:a aac -b:a 128k -f flv rtmp://server/live/stream_key
```

### 2.9 Faststart for web playback

```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a copy -movflags +faststart output.mp4
```

### 2.10 ffprobe inspection

```bash
# Full stream + format info, JSON
ffprobe -v error -show_streams -show_format -print_format json input.mp4

# Only video stream
ffprobe -v error -select_streams v:0 -show_streams input.mp4

# Frame rate
ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 input.mp4
```

---

## 3. Parameter glossary

`verified` column:

- `✓` — cross-checked against ffmpeg.org/ffmpeg.html official docs
- `~` — works as described but specific defaults may vary by build
- `?` — not independently verified; treat as guidance, not contract

### 3.1 Input / output

| param       | type   | default  | description                              | verified |
| ----------- | ------ | -------- | ---------------------------------------- | -------- |
| `-i <file>` | path   | required | input source (file path, URL, or device) | ✓        |
| `-y`        | flag   | prompt   | overwrite output without asking          | ✓        |
| `-n`        | flag   | prompt   | never overwrite output                   | ✓        |
| `-f <fmt>`  | string | auto     | force input/output format                | ✓        |

### 3.2 Codec selection

| param          | type   | default | description                                                              | verified |
| -------------- | ------ | ------- | ------------------------------------------------------------------------ | -------- |
| `-c:v <codec>` | string | auto    | video codec (`libx264`, `libx265`, `libvpx-vp9`, `libsvtav1`, `copy`, …) | ✓        |
| `-c:a <codec>` | string | auto    | audio codec (`aac`, `libmp3lame`, `libopus`, `copy`, …)                  | ✓        |
| `-c copy`      | string | —       | copy all streams without re-encoding                                     | ✓        |
| `-an`          | flag   | —       | strip all audio                                                          | ✓        |
| `-sn`          | flag   | —       | strip all subtitles                                                      | ✓        |
| `-vn`          | flag   | —       | strip all video                                                          | ✓        |

### 3.3 Quality & bitrate

| param                              | type   | default      | range / notes                                                | verified |
| ---------------------------------- | ------ | ------------ | ------------------------------------------------------------ | -------- |
| `-crf <n>`                         | int    | 23 (libx264) | 0–51, lower = better, 0 = lossless (libx264 only)            | ✓        |
| `-b:v <rate>`                      | string | —            | e.g. `5000k`, `2M`; target bitrate                           | ✓        |
| `-minrate`, `-maxrate`, `-bufsize` | string | —            | constrain VBR to look like CBR                               | ✓        |
| `-preset <name>`                   | enum   | medium       | `ultrafast` … `veryslow`; slower = better compression        | ✓        |
| `-tune <name>`                     | enum   | —            | `film`, `animation`, `grain`, `stillimage`, `zerolatency`, … | ✓        |
| `-qp <n>`                          | int    | —            | constant QP mode (NVENC, QSV, VAAPI prefer this)             | ~        |

### 3.4 Timing / seek

| param        | type   | default | description                                                                                                                           | verified |
| ------------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `-ss <time>` | time   | 0       | start time; `00:01:30` or seconds; before `-i` = input-seek (fast, keyframe-snapped), after `-i` = output-seek (slow, frame-accurate) | ✓        |
| `-t <dur>`   | time   | full    | duration of output                                                                                                                    | ✓        |
| `-to <time>` | time   | —       | end time (alternative to `-t`)                                                                                                        | ✓        |
| `-r <fps>`   | number | —       | set frame rate (output: resample; input: assume)                                                                                      | ✓        |

### 3.5 Filters

| param                     | type   | default | description                                                               | verified |
| ------------------------- | ------ | ------- | ------------------------------------------------------------------------- | -------- |
| `-vf <chain>`             | string | —       | video filter chain (single input, comma-separated); alias for `-filter:v` | ✓        |
| `-af <chain>`             | string | —       | audio filter chain                                                        | ✓        |
| `-filter_complex <graph>` | string | —       | multi-input / multi-output filter graph; use labels `[0:v]...[out]`       | ✓        |
| `-lavfi <graph>`          | string | —       | shortcut for `-filter_complex` with no input file                         | ?        |

### 3.6 Stream mapping

| param             | type | description                        | verified |
| ----------------- | ---- | ---------------------------------- | -------- |
| `-map 0:v:0`      | —    | first video stream of input 0      | ✓        |
| `-map 0:a`        | —    | all audio streams of input 0       | ✓        |
| `-map 0`          | —    | all streams of input 0 (all types) | ✓        |
| `-map_metadata 0` | —    | copy metadata from input 0         | ~        |

### 3.7 Container / muxing

| param                         | type   | description                                                                   | verified |
| ----------------------------- | ------ | ----------------------------------------------------------------------------- | -------- |
| `-f <fmt>`                    | string | force container (`mp4`, `mkv`, `flv`, `hls`, `matroska`, …)                   | ✓        |
| `-movflags <flags>`           | string | MP4/MOV specific; common: `+faststart` (move moov to front for web streaming) | ✓        |
| `-hls_time <sec>`             | int    | HLS segment duration                                                          | ✓        |
| `-hls_list_size <n>`          | int    | segments in playlist (0 = all)                                                | ✓        |
| `-hls_segment_filename <pat>` | string | segment filename pattern                                                      | ✓        |
| `-hls_key_info_file <file>`   | path   | enable HLS encryption                                                         | ?        |

### 3.8 Hardware acceleration

| param                             | type   | description                                                                                                  | verified |
| --------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------ | -------- |
| `-hwaccel <api>`                  | enum   | hardware decode API: `vaapi`, `videotoolbox`, `dxva2`, `d3d11va`, `cuda`, `qsv`                              | ✓        |
| `-hwaccel_output_format <fmt>`    | enum   | where to land decoded frames: `vaapi`, `cuda`, `d3d11`, `dxva2_vld`, … (forces GPU frames)                   | ✓        |
| `-init_hw_device <type>[=<name>]` | string | explicitly create a hw device; `only needed for non-default device selection` (e.g. `cuda:1` for second GPU) | ✓        |
| `-hwaccel_device <id>`            | string | pick a specific hw device; auto-creates a default device if none                                             | ✓        |

#### VAAPI note (corrected)

The commonly-repeated claim that "VAAPI needs `-init_hw_device vaapi`" is
`wrong for basic use`. FFmpeg will auto-create a default VAAPI device from
`/dev/dri/renderD128` (or the first render node it finds). Use the longer form
only when:

- selecting a non-default render node (e.g. `-init_hw_device vaapi=vaapi0:/dev/dri/renderD129`)
- sharing a device across multiple FFmpeg invocations
- using CUDA → VAAPI bridge

Source: ffmpeg.org/ffmpeg.html#Hardware-acceleration

### 3.9 Debug

| param               | type   | default | description                                                                        | verified |
| ------------------- | ------ | ------- | ---------------------------------------------------------------------------------- | -------- |
| `-v <level>`        | enum   | info    | `quiet`, `panic`, `fatal`, `error`, `warning`, `info`, `verbose`, `debug`, `trace` | ✓        |
| `-loglevel <level>` | enum   | info    | alias for `-v`                                                                     | ✓        |
| `-report`           | flag   | —       | dump full log to `ffmpeg-YYYYMMDD-HHMMSS.log` in cwd                               | ✓        |
| `-stats`            | flag   | on      | show encoding progress                                                             | ~        |
| `-progress <url>`   | string | —       | pipe progress to `pipe:1`, `pipe:2`, or file                                       | ✓        |
| `-hide_banner`      | flag   | —       | suppress build info on startup                                                     | ✓        |

---

## 4. Common pitfalls

| symptom                                              | cause                                                                                    | fix                                                                                              |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| output has audio but no video                        | `-c:v copy` but source has no video, or `-map 0:a` without `-map 0:v`                    | check with `ffprobe -show_streams`; add explicit `-map 0:v:0`                                    |
| trim point off by 1–2 seconds                        | `-ss` after `-i` (output-seek) but assumed input-seek semantics, or vice versa           | for keyframe-accurate fast cut: `-ss` before `-i`; for frame-accurate slow cut: `-ss` after `-i` |
| output larger than input                             | target container needs re-encode but `-c copy` was used; or low-compression codec chosen | drop `-c copy`, use `-c:v libx264 -crf 23`                                                       |
| web player shows "loading" forever, then plays       | `moov` atom is at end of file (default MP4 layout)                                       | add `-movflags +faststart`                                                                       |
| HLS segment boundaries chop mid-frame                | `hls_time` not aligned with GOP size                                                     | set GOP first: `-g 60` (assuming 30fps, 2s GOP), or use `-hls_time 2 -g 60`                      |
| `Cannot find a matching stream` for filter           | filter applied to wrong stream type (e.g. `scale=` to audio)                             | use `-filter_complex` and explicit labels, or use `-vf` (video) / `-af` (audio)                  |
| `moov atom not found` when reading                   | file was truncated or never finalized                                                    | re-run with `-c copy` from a known-good source; check disk space on writer side                  |
| `Permission denied` on `/dev/dri/renderD128` (VAAPI) | user not in `video` or `render` group                                                    | `sudo usermod -aG video $USER` then re-login; or run with proper groups                          |
| `NVENC not available` despite NVIDIA GPU             | ffmpeg built without `--enable-nvenc`; or driver too old                                 | check `ffmpeg -hide_banner -encoders \| grep nvenc`; rebuild ffmpeg with NVENC support           |
| video plays at wrong speed after `setpts`            | time-base mismatch; forgot to also `atempo` audio                                        | for 2x speed: `-vf "setpts=PTS/2" -af "atempo=2.0"`                                              |
| `-ss` placed after `-i` is slow on huge files        | decoder must walk through every frame to reach seek point                                | use two-pass: `-ss <t> -i in -ss <t> -c copy out` (re-seek after input) or pre-trim              |
| two-pass first run produces file anyway              | forgot `-f null -` (or `-f null /dev/null`)                                              | first pass must write to null device, not a real file                                            |
| CRF change has no visible effect on file             | `-tune` or `-preset` overriding; or `-b:v` set (forces target bitrate mode)              | remove conflicting flags; CRF only works without `-b:v`                                          |

---

## 5. Version notes (Changelog-anchored, audited 2026-06-10)

> Static version diffs rot fast and are easy to fabricate. Prefer the
> "how to check" commands below over memorizing this table.

### 5.1 Real changes per official Changelog

| version | codename      | date    | change                                                                                                                 |
| ------- | ------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| 4.4     | —             | 2021-03 | `added` AV1 decode/encode (SVT-AV1, NVDEC, VAAPI, QSV, DXVA2/D3D11VA), 70+ new decoders/encoders/filters, `ffprobe -o` |
| 4.4     | —             | 2021-03 | `removed` libwavpack encoder                                                                                           |
| 5.0     | "Lorentz"     | 2022-01 | `added` VideoToolbox VP9 hwaccel, VideoToolbox ProRes hwaccel, ProRes encoder, `yadif_videotoolbox` filter             |
| 5.1     | —             | 2022    | `removed` XvMC hwaccel; `added` QOI image format, `ffprobe -o` option                                                  |
| 6.0     | "Von Neumann" | 2023-02 | `build req` ffmpeg now requires threading to be enabled                                                                |
| 6.0     | —             | 2023-02 | `behavior` every muxer runs in its own thread by default                                                               |
| 6.0     | —             | 2023-02 | `deprecated` CrystalHD decoders                                                                                        |
| 6.1     | "Heaviside"   | 2023-11 | `deprecated` `-top` CLI option; use `setfield` filter                                                                  |

### 5.2 Claims from earlier drafts that are NOT supported by Changelog

The following claims circulated in community blogs / LLM-generated docs but
`do not appear in the official Changelog` for the version stated. Do not
assert them as fact:

- ❌ "FFmpeg 5.x changed default loglevel from `info` to `warning`" — Changelog does not record this
- ❌ "FFmpeg 5.x changed `-nostats` default behavior" — Changelog does not record this
- ❌ "FFmpeg 5.x renamed `old_videotoolbox` to `h264_videotoolbox`" — 5.0 only `added` new VT features, no rename
- ❌ "FFmpeg 5.x changed libx264 default preset to `ultrafast`" — Changelog does not record this
- ❌ "FFmpeg 6.x routed `-c:v h264` default to `libx264`" — Changelog does not record this
- ❌ "FFmpeg 6.0 removed vfwCapture support" — vfwcap was removed in the 2.x era, not 6.0

### 5.3 How to check your version's behavior

```bash
# Version + build config
ffmpeg -version

# Available encoders / decoders / muxers
ffmpeg -hide_banner -encoders | head -50
ffmpeg -hide_banner -decoders | head -50
ffmpeg -hide_banner -muxers | head -30

# Help for a specific encoder (lists all options and defaults)
ffmpeg -hide_banner -h encoder=libx264
ffmpeg -hide_banner -h encoder=h264_nvenc

# Look for a specific flag
ffmpeg -h full 2>&1 | grep -E '^\s*-' | grep -i preset
```

### 5.4 Sources for version notes

- <https://github.com/FFmpeg/FFmpeg/blob/release/5.0/Changelog>
- <https://github.com/FFmpeg/FFmpeg/blob/release/5.1/Changelog>
- <https://github.com/FFmpeg/FFmpeg/blob/release/6.0/Changelog>
- <https://github.com/FFmpeg/FFmpeg/blob/release/6.1/Changelog>
- <https://github.com/FFmpeg/FFmpeg/blob/release/6.1/RELEASE_NOTES>

---

## 6. Codec quick reference

| use case                    | codec             | typical params                                     |
| --------------------------- | ----------------- | -------------------------------------------------- |
| web / general               | libx264           | `-crf 23 -preset medium -movflags +faststart`      |
| smaller file, slower encode | libx265           | `-crf 28 -preset slow`                             |
| open / royalty-free         | libvpx-vp9        | `-crf 31 -b:v 0`                                   |
| modern, smallest file       | libsvtav1         | `-crf 35 -preset 6`                                |
| editing / post-production   | prores_ks         | `-profile:v 3` (HQ) or `2` (standard)              |
| lossless                    | libx264           | `-crf 0 -preset veryslow`                          |
| NVIDIA GPU                  | h264_nvenc        | `-preset p4 -rc vbr -cq 23`                        |
| Intel QSV                   | h264_qsv          | `-preset medium -global_quality 23`                |
| macOS GPU                   | h264_videotoolbox | `-b:v 5000k` (or `-q:v 65` for VBR-like)           |
| Linux VAAPI                 | h264_vaapi        | (requires `-hwaccel vaapi` and proper render node) |

---

## 7. CRF / preset / bitrate reference

### 7.1 libx264 CRF → perceptual quality

| CRF   | quality          | typical use                      |
| ----- | ---------------- | -------------------------------- |
| 0     | lossless         | archival                         |
| 14–17 | near-lossless    | high-quality source preservation |
| 18–22 | high             | archival distribution            |
| 23    | default          | general purpose                  |
| 24–28 | acceptable       | web / streaming                  |
| 29–34 | low              | bandwidth-constrained            |
| 40+   | visibly degraded | avoid                            |

### 7.2 libx264 preset → speed vs compression

| preset    | relative speed | compression efficiency |
| --------- | -------------- | ---------------------- |
| ultrafast | 1×             | worst                  |
| superfast | 2×             | very poor              |
| veryfast  | 4×             | poor                   |
| faster    | 5×             | below average          |
| fast      | 6×             | below average          |
| medium    | 8×             | baseline (default)     |
| slow      | 12×            | better                 |
| slower    | 18×            | much better            |
| veryslow  | 30×            | best                   |

Rule of thumb: doubling encode time roughly buys 5–10% smaller file at same quality.

### 7.3 Audio bitrate guide

| codec  | bitrate range | use case                                    |
| ------ | ------------- | ------------------------------------------- |
| AAC    | 128–320 kbps  | general (`.m4a`, `.mp4`)                    |
| MP3    | 128–320 kbps  | legacy compatibility                        |
| Opus   | 64–256 kbps   | streaming (better quality per bit than AAC) |
| FLAC   | ~1000 kbps    | lossless archive                            |
| Vorbis | 128–320 kbps  | open containers (`.ogg`)                    |

---

## 8. ffprobe cheatsheet

```bash
# Streams + format, JSON output
ffprobe -v error -show_streams -show_format -print_format json input.mp4

# Only first video stream
ffprobe -v error -select_streams v:0 -show_streams input.mp4

# Frame rate (compact)
ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate,avg_frame_rate \
        -of default=noprint_wrappers=1 input.mp4

# Bit rate
ffprobe -v error -show_format input.mp4 | grep bit_rate

# Codec names across all streams
ffprobe -v error -show_streams input.mp4 | grep codec_name

# Duration in seconds
ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 input.mp4

# Read packet timestamps (debug seek/trim issues)
ffprobe -v error -read_intervals "%+#30" -show_packets -select_streams v input.mp4
```

---

## 9. Citations (anchored to official docs)

| topic                                                       | URL                                                                  |
| ----------------------------------------------------------- | -------------------------------------------------------------------- |
| synopsis & main options                                     | <https://ffmpeg.org/ffmpeg.html#Synopsis>                            |
| generic options (`-y`, `-n`, `-f`, `-i`)                    | <https://ffmpeg.org/ffmpeg.html#Generic-options>                     |
| main options (`-c`, `-map`, `-ss`, `-t`, `-to`)             | <https://ffmpeg.org/ffmpeg.html#Main-options>                        |
| advanced options (`-preset`, `-tune`, `-crf`)               | <https://ffmpeg.org/ffmpeg.html#Advanced-options>                    |
| encoding flow & bitrate control                             | <https://ffmpeg.org/ffmpeg.html#Encoding>                            |
| stream copy (`-c copy`)                                     | <https://ffmpeg.org/ffmpeg.html#Stream-copy>                         |
| filter description (single chain)                           | <https://ffmpeg.org/ffmpeg.html#Filter-description>                  |
| filtergraph syntax 1 (`-vf` simple)                         | <https://ffmpeg.org/ffmpeg.html#Filtergraph-syntax-1>                |
| filtergraph syntax 2 (`-filter_complex`)                    | <https://ffmpeg.org/ffmpeg.html#Filtergraph-syntax-2>                |
| timeline editing (`-ss`, `-t`)                              | <https://ffmpeg.org/ffmpeg.org/ffmpeg.html#Timeline-editing>         |
| HLS muxer options                                           | <https://ffmpeg.org/ffmpeg-formats.html#hls-1>                       |
| mov/mp4 muxer flags (`+faststart`)                          | <https://ffmpeg.org/ffmpeg-formats.html#mov>                         |
| hardware acceleration                                       | <https://ffmpeg.org/ffmpeg.html#Hardware-acceleration>               |
| VAAPI specifics                                             | <https://ffmpeg.org/ffmpeg.html#VAAPI>                               |
| NVENC specifics                                             | <https://ffmpeg.org/ffmpeg.html#NVENC>                               |
| VideoToolbox specifics                                      | <https://ffmpeg.org/ffmpeg.html#VideoToolbox>                        |
| log levels                                                  | <https://ffmpeg.org/ffmpeg.html#Generic-options> (search `loglevel`) |
| ffprobe reference                                           | <https://ffmpeg.org/ffprobe.html>                                    |
| filter reference (scale, crop, overlay, drawtext, yadif, …) | <https://ffmpeg.org/ffmpeg-filters.html>                             |
| codec options index                                         | <https://ffmpeg.org/ffmpeg-codecs.html>                              |
| Changelog (version history)                                 | <https://github.com/FFmpeg/FFmpeg/tree/release>                      |
