"""Subprocess entry point that runs pyannote and prints speaker turns as JSON.

Kept separate from :mod:`transcribe_media.diarize` so torch is only ever
imported here, in a process that exits when diarization is done.

Output contract: the last line of stdout is one JSON object
``{"turns": [[start, end, label], ...], "note": "..."}``; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import audio
from .cuda_libs import configure_cuda_environment


def load_waveform(path: str, torch: Any) -> tuple[Any, int]:
    """Read our decoded WAV into a (1, samples) float tensor.

    Handing pyannote a tensor avoids its own file decoding (torchcodec), which
    needs extra FFmpeg shared libraries on some platforms.
    """
    samples = audio.load_samples(Path(path))
    return torch.from_numpy(samples).unsqueeze(0), audio.TARGET_SR


def cuda_supported(torch: Any, index: int) -> str | None:
    """None if this torch build can run kernels on GPU ``index``, else the reason."""
    if not torch.cuda.is_available():
        return "torch cannot see a CUDA device"
    major, minor = torch.cuda.get_device_capability(index)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        return f"this PyTorch build has no kernels for {arch} (Blackwell needs a CUDA 12.8 build, torch >= 2.8)"
    return None


def annotation_to_turns(annotation: Any) -> list[list[Any]]:
    turns = [[float(seg.start), float(seg.end), str(label)]
             for seg, _, label in annotation.itertracks(yield_label=True)]
    turns.sort(key=lambda t: t[0])
    return turns


def run(args: argparse.Namespace) -> dict:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(args.pipeline)
    if pipeline is None:
        raise RuntimeError("pyannote returned no pipeline from the local model directory")

    note = ""
    if args.device == "cuda":
        problem = cuda_supported(torch, args.device_index)
        if problem:
            note = f"diarization ran on the CPU: {problem}"
        else:
            pipeline.to(torch.device("cuda", args.device_index))

    waveform, rate = load_waveform(args.wav, torch)
    kwargs = {k: v for k, v in (("num_speakers", args.num_speakers),
                                ("min_speakers", args.min_speakers),
                                ("max_speakers", args.max_speakers)) if v is not None}
    output = pipeline({"waveform": waveform, "sample_rate": rate}, **kwargs)
    # The exclusive variant has no overlapping turns, which suits aligning to ASR words.
    return {"turns": annotation_to_turns(output.exclusive_speaker_diarization), "note": note}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diarize_worker")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args(argv)

    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"  # usage telemetry is on by default
    configure_cuda_environment()  # before torch is imported
    try:
        payload = run(args)
    except Exception as exc:  # report any failure as a clean one-line error
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
