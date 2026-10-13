# transcribe-media

A Claude skill that transcribes **any audio or video** — a local file or a link such as a
YouTube video — locally and for free, with **accuracy as the priority and speed as an
afterthought**. It detects your hardware (GPU, VRAM, RAM), picks the best Whisper model that
fits, downloads it, and optionally labels who said what.

```
file or URL ─► ffmpeg (16 kHz mono) ─► Whisper large-v3 ─► speaker labels ─► quality check ─► txt / json / srt / vtt
   (yt-dlp for links)                 (faster-whisper)       (pyannote, optional)
```

By default audio never leaves your machine: the only network use is downloading the models once,
and fetching the video itself when you give a URL. It also runs inside cloud sessions and can rent a
cloud GPU when you explicitly ask it to (see [Cloud GPU](#cloud-gpu)).

## Install as a skill

Put this folder where Claude Code looks for skills, for example `~/.claude/skills/transcribe-media`.
Claude reads [`SKILL.md`](SKILL.md). The first run creates a private environment (outside this
folder) and downloads the model; you can also do that up front:

```bash
python scripts/setup.py            # detects hardware, installs only what it needs
python scripts/transcribe.py --check
python scripts/transcribe.py --self-test   # loads the real model on your GPU and runs a tiny inference
```

Requirements: FFmpeg (`winget install ffmpeg` / `apt install ffmpeg` / `brew install ffmpeg`).
[`uv`](https://docs.astral.sh/uv/) is used if present (it also fetches Python 3.12); otherwise Python 3.10+.
For YouTube audio, yt-dlp wants a JavaScript runtime: Deno 2.3+ or Node 22+.

## Use it

```bash
python scripts/transcribe.py "interview.mp4"
python scripts/transcribe.py "https://www.youtube.com/watch?v=..." --timestamps
python scripts/transcribe.py talk.m4a --language de --prompt "Zorblax, Quuxcorp" --format txt,srt
python scripts/transcribe.py video-url --compare-captions     # cross-check against YouTube's captions
```

Outputs go to the current directory as `<name>.transcript.txt` and `.json` (UTF-8). The JSON holds
word timings, per-segment confidence, the quality verdict and provenance. Existing files are never
overwritten without `--overwrite`. Run `--help` for every option.

### YouTube captions

The transcript always comes from Whisper. If the video also has YouTube captions, the run tells
you (creator-uploaded subtitles, or genuine auto-generated captions; machine-translated tracks are
ignored). Add `--compare-captions` to fetch that track, see what share of words agree, and save its
text next to your transcript.

## Hardware → model

Chosen automatically, the largest model that fits. On a GPU nothing is quantized unless memory forces it;
the CPU path always uses int8, which costs very little accuracy:

| Detected | Model | Precision |
|---|---|---|
| NVIDIA ≥ 6 GB VRAM (e.g. **RTX 5080**, 16 GB) | large-v3 | float16 |
| NVIDIA 3.5–6 GB, pre-Blackwell | large-v3 | int8_float16 |
| NVIDIA 2.5–6 GB, Blackwell | medium | float16 |
| No usable GPU, ≥ 8 GB RAM | large-v3 | int8 on CPU (slow, same accuracy) |
| No usable GPU, 4–8 GB RAM / less | medium / small | int8 on CPU |

Blackwell (RTX 50-series) needs CUDA 12.8 and CTranslate2 ≥ 4.6.3, and CTranslate2 disables INT8
there, so those cards always use float16. If CUDA cannot be initialised the tool falls back to the
CPU with a warning, and `--check` explains the fix. Apple Silicon runs on the CPU (CTranslate2 has no
Metal backend).

Models are pinned to exact Hugging Face commits and only allow-listed repositories are downloaded.
`--offline` and `--no-download` forbid network use.

## Cloud GPU

Where the job runs is a separate choice from what it does. `--backend` (or `TRANSCRIBE_MEDIA_BACKEND`):

| | Behaviour | Uploads audio? |
|---|---|---|
| `local` (default) | This machine | never |
| `auto` | local GPU if any, else a cloud GPU for audio ≥ 10 min (`--min-cloud-minutes`), else the CPU | only when it uses the cloud |
| `modal` | always a rented [Modal](https://modal.com) GPU | yes |

Everything the cloud runs is the same `transcribe_media.job.run_job` that runs locally; the GPU machine
detects its own hardware and picks `large-v3` in float16, so results match a local GPU run. Only the 16 kHz
audio is sent (lossless FLAC, not the original video) together with the transcription options, which include
any `--prompt` text, to a private volume in **your** Modal account. The audio is deleted from the GPU machine as
soon as it is decoded, and from the volume when the job ends (leftovers from a crashed run are purged after
24 hours). Your local Hugging Face token is never sent. `--offline` cannot be combined with a cloud backend.

**Set up (once, by you):**

1. Create a Modal account. At the time of writing the Starter plan has no base fee and includes $30/month of
   free credits, and an L4 GPU costs about $0.000222/s (roughly $0.80/h); check [Modal's pricing](https://modal.com/pricing).
2. `python scripts/setup.py --cloud` (installs the Modal client), then `modal token new`.
   In a **cloud session** instead add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` to the environment's settings
   (never paste them into chat) and allow Modal's hosts in its network access.
3. `python scripts/transcribe.py --self-test --backend modal` checks credentials, the GPU image and the
   model using a synthetic tone only.
4. Optional speaker labels in the cloud: `modal secret create huggingface HF_TOKEN=...` yourself, then pass
   `--cloud-hf-secret huggingface`. After the first run the model is cached on the volume.

**Cost and safety controls:** `--cloud-max-minutes` (default 120, at most 360) cancels the job and stops billing,
and the GPU function carries its own server-side timeout as a backstop in case this machine dies; Ctrl-C, SIGTERM and
SIGHUP cancel it immediately; each run uses a single GPU container and failed jobs are never retried. Separate
runs started at the same time are independent, so each has its own cap. `--cloud-gpu` accepts
`T4, L4, A10, L40S, A100, H100` (the default tries L4, then A10, then L40S; the data-center Blackwell parts B200/B300
are left out because Modal documents that B300 needs CUDA 13.1+ while this image targets CUDA 12.8).

**Other GPU hosts** (Colab, Kaggle, RunPod, any machine with a GPU): install the package there and run the
normal CLI, which detects the GPU on its own:
`pip install "transcribe-media[gpu,diarize] @ git+https://github.com/thewongdirection/transcribe-media@<commit-sha>"`
(pin a commit you have read; an unpinned branch can change under you).

**Cloud sessions without a GPU** (for example Claude Code on the web) are detected automatically. The
environment and models are rebuilt each session, runs fall back to the CPU, and `--check` reports whether a
cloud GPU is configured. Note that YouTube often refuses anonymous downloads from cloud servers (see limits).

## Speaker labels

On by default (`--diarize auto`) with pyannote's **community-1**, the best open-weights diarizer.
The model is gated, so it needs a free Hugging Face token **once**:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1
3. Set `HF_TOKEN` in your environment, then `python scripts/setup.py --diarization`
   (this is automatic on first run if a token is already configured)

After the first download no token is needed. Without a suitable token, labels are skipped and the
transcript is still written; `--diarize on` turns that into an error instead. The neural step runs
in a separate process so a crash cannot lose a long transcription.

## Design notes

* **Always transcribes.** Beam search, temperature fallback, VAD, a hallucination guard and
  multi-window language detection are on; `condition_on_previous_text` is off to avoid repetition loops.
* **We decode audio ourselves** (FFmpeg → 16 kHz WAV → array) and never use faster-whisper's PyAV
  decode path, which breaks with current PyAV releases. pyannote also receives an in-memory waveform,
  avoiding its torchcodec decoder.
* **Security.** URLs are http(s) only, carry no embedded credentials, and must resolve to public addresses.
  That is also enforced when connecting, so redirects, embedded media and DNS tricks cannot reach private
  addresses. yt-dlp runs through its API (no shell) with your config and its plugin folders ignored, only
  downloads over protocols the network guard can see, is deliberately left without an FFmpeg of its own (so a
  crafted playlist cannot make FFmpeg fetch internal addresses behind the guard's back), never expands
  playlists, refuses live streams, and stops at `--max-download-mb` (or an hour) even when the server does
  not state a size. Cookies are off unless
  you opt in with `--cookies-from-browser`. FFmpeg is restricted to the `file`/`pipe` protocols and refuses
  playlist-style files. Windows network (UNC) paths are refused for inputs and outputs, since opening one sends
  your credentials to that host, and on Windows programs are never looked up in the current folder first.
  Titles, captions, error text and results returned by a cloud job are validated and stripped of control
  characters before they are shown or stored. Tokens are read from the environment only, never accepted as arguments, and
  never written to output; the speaker-label worker runs with a stripped-down environment and pyannote's usage
  telemetry switched off. Source URLs are stored without query strings.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

Offline and hermetic: models, the network and the GPU are replaced by fakes; FFmpeg is used for real
where available. Live checks (a real transcription, YouTube metadata and captions) were also run by hand.

## Known limits and what is not yet verified

* **Not run on an RTX 5080 or on Windows yet.** The selection logic, CUDA library loading and error
  guidance are unit-tested and follow CTranslate2 and PyTorch release notes, but no GPU was available
  during development. `python scripts/transcribe.py --self-test` is the check to run on your machine.
* **Speaker labels** were verified up to package installation, dependency resolution and failure
  handling; the gated model itself was not run end to end (it needs a token).
* **The Modal backend has not been run against a live Modal account** (no credentials were available during
  development). Its logic is tested against a fake provider, and the provider calls it makes are checked
  against the real `modal` 1.6 package, but the GPU image build and an actual cloud run are unverified.
  `--self-test --backend modal` is the check to run first.
* **YouTube audio download** can be refused (`Sign in to confirm you're not a bot`, HTTP 403),
  typically from cloud servers and VPNs. Metadata and caption fetching work; see `--cookies-from-browser`.
* The CUDA 12.8 PyTorch index currently tops out at torch 2.11; that is what speaker labels use on NVIDIA.
* The connect-time network guard covers connections made by this process (yt-dlp's own HTTP client). With an HTTP
  proxy configured, the connection to that one proxy address and port is allowed and the proxy's own rules then
  decide what is reachable. The JavaScript runtime yt-dlp starts for YouTube (Deno or Node) is outside the guard;
  yt-dlp starts it without permissions (no network or file access).
* Python dependencies use lower and upper version bounds but are not hash-locked; models are pinned to commits.
* Two runs at the same time are not coordinated: both can pick the same output name before either has written it.
  Files are replaced atomically, so one result wins and none is half-written. A recording is read into memory as
  16 kHz audio, peaking at roughly 350 MB per hour of audio, so the 10-hour default limit can need several GB.
* Single videos only (playlists are refused). No license file is included yet.
