"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  acquisition_module.py
=============================================================================
DataAcquisitionModule — master orchestrator.

Routes incoming requests to the correct acquisition mode and returns
standardised PipelinePacket(s) ready for Module 3 (Preprocessing).

Acquisition modes
-----------------
  2.1  Import       — load existing CSV/folder data
  2.2  Simulate     — generate new data via Module 1A
  2.3  Live         — stream from a wearable device with annotation
  2.4  Deployment   — ingest non-annotated data for inference

Pipeline handoff
----------------
  PipelinePacket.is_annotated = True  → training/testing path (Modules 3–8)
  PipelinePacket.is_annotated = False → deployment inference path (Module 9)

=============================================================================
"""

from __future__ import annotations
from pipeline_packet import PipelinePacket
import json
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

# ── Module directory on path ──────────────────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# VERSION / OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M1"
OUTPUT_ROOT = "outputs"

MODE_IMPORT = "2.1"
MODE_SIMULATE = "2.2"
MODE_LIVE = "2.3"
MODE_DEPLOYMENT = "2.4"

VALID_MODES = {MODE_IMPORT, MODE_SIMULATE, MODE_LIVE, MODE_DEPLOYMENT}

MODE_DESCRIPTIONS = {
    MODE_IMPORT: "Import existing data from a folder or CSV",
    MODE_SIMULATE: "Create new simulated data via Module 1A",
    MODE_LIVE: "Stream live data from wearable device with annotation",
    MODE_DEPLOYMENT: "Ingest non-annotated data for deployment inference",
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def next_run_folder(mode: str, custom_name: str = None) -> Path:
    """Auto-numbered output folder: outputs/M2_v1.0.0_mode2.1_run_001/"""
    output_root = MODULE_DIR / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    mode_tag = f"mode{mode.replace('.', '_')}"
    version_tag = f"{MODULE_LABEL}_v{MODULE_VERSION}_{mode_tag}"

    if custom_name:
        base = output_root / custom_name
        if not base.exists():
            base.mkdir(parents=True)
            return base
        prefix = custom_name
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    else:
        prefix = f"{version_tag}_run"
        pattern = re.compile(rf"^{re.escape(version_tag)}_run_(\d+)$")

    existing = [
        int(m.group(1))
        for d in output_root.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    next_num = (max(existing) + 1) if existing else 1
    folder = output_root / f"{prefix}_{next_num:03d}"
    folder.mkdir(parents=True)
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# DATA ACQUISITION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class DataAcquisitionModule:
    """
    Main entry point for Module 2.

    Parameters
    ----------
    mode         : '2.1' | '2.2' | '2.3' | '2.4'
    save_packets : save PipelinePackets to output folder after acquisition
    out_name     : custom output folder name (auto-numbered if None)
    verbose      : print progress messages
    """

    def __init__(
        self,
        mode: str = MODE_SIMULATE,
        save_packets: bool = True,
        out_name: str = None,
        verbose: bool = True,
        output_dir: str = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {sorted(VALID_MODES)}.\n"
                f"  2.1 – {MODE_DESCRIPTIONS[MODE_IMPORT]}\n"
                f"  2.2 – {MODE_DESCRIPTIONS[MODE_SIMULATE]}\n"
                f"  2.3 – {MODE_DESCRIPTIONS[MODE_LIVE]}\n"
                f"  2.4 – {MODE_DESCRIPTIONS[MODE_DEPLOYMENT]}"
            )
        self.mode = mode
        self.save_packets = save_packets
        self.out_name = out_name
        self.verbose = verbose
        self.output_dir = output_dir

    # ── Mode 2.1: Import ───────────────────────────────────────────────────

    def run_import(
        self,
        source_path: str | Path,
        user_id: str = "imported_user",
    ) -> PipelinePacket:
        """
        Mode 2.1 — Import existing data.

        Parameters
        ----------
        source_path : Module 1A output folder, or any signal CSV file
        user_id     : participant identifier
        """
        from mode_1_1_import import DataImporter
        self._print_mode_header(MODE_IMPORT, source=str(source_path))

        packet = DataImporter(
            source_path=source_path,
            user_id=user_id,
            verbose=self.verbose,
        ).run()

        return self._finalise([packet], MODE_IMPORT)[0]

    # ── Mode 2.2: Simulate ─────────────────────────────────────────────────

    def run_simulate(
        self,
        duration_s: float = 300,
        n_events: Union[int, str] = 5,
        event_duration_s: Union[float, str] = 30.0,
        emotions=None,
        noise_level: str = "medium",
        seed: int = 42,
        n_users: int = 1,
        shared_events: bool = False,
        save_m1a_output: bool = False,
    ) -> List[PipelinePacket]:
        """
        Mode 2.2 — Simulate new data via Module 1A.

        Returns a list of PipelinePackets (one per user).
        """
        from mode_1_2_simulate import SimulationConnector
        self._print_mode_header(MODE_SIMULATE,
                                users=n_users, duration=duration_s)

        packets = SimulationConnector(
            duration_s=duration_s,
            n_events=n_events,
            event_duration_s=event_duration_s,
            emotions=emotions,
            noise_level=noise_level,
            seed=seed,
            n_users=n_users,
            shared_events=shared_events,
            save_m1a_output=save_m1a_output,
            verbose=self.verbose,
        ).run()

        return self._finalise(packets, MODE_SIMULATE)

    def run_simulate_interactive(self) -> List[PipelinePacket]:
        """Mode 2.2 — Interactive simulation setup."""
        from mode_1_2_simulate import simulate_interactive
        self._print_mode_header(MODE_SIMULATE)
        packets = simulate_interactive()
        return self._finalise(packets, MODE_SIMULATE)

    # ── Mode 2.3: Live ─────────────────────────────────────────────────────

    def run_live_file(
        self,
        csv_path: str | Path,
        user_id: str = "test_participant",
        speed_factor: float = 10.0,
        max_duration_s: float = None,
    ) -> PipelinePacket:
        """
        Mode 2.3 — Replay a CSV file as a live stream (for testing).

        Parameters
        ----------
        csv_path       : path to a combined_signals.csv from Module 1A
        speed_factor   : playback speed (10.0 = 10x faster than real time)
        max_duration_s : auto-stop after N seconds (None = manual stop)
        """
        from mode_1_3_live import FileStreamAdapter, LiveDataCollector
        self._print_mode_header(MODE_LIVE, source=str(csv_path))

        adapter = FileStreamAdapter(csv_path, speed_factor=speed_factor)
        packet = LiveDataCollector(
            device_adapter=adapter,
            user_id=user_id,
            max_duration_s=max_duration_s,
            verbose=self.verbose,
        ).run()

        return self._finalise([packet], MODE_LIVE)[0]

    def run_live_e4(
        self,
        user_id: str = "participant_001",
        host: str = "127.0.0.1",
        port: int = 28000,
        max_duration_s: float = None,
    ) -> PipelinePacket:
        """
        Mode 2.3 — Live acquisition from Empatica E4 device.

        Requires the E4 BLE Streaming Server to be running.
        """
        from mode_1_3_live import EmpaticaE4Adapter, LiveDataCollector
        self._print_mode_header(MODE_LIVE,
                                source=f"Empatica E4 @ {host}:{port}")

        adapter = EmpaticaE4Adapter(host=host, port=port)
        packet = LiveDataCollector(
            device_adapter=adapter,
            user_id=user_id,
            max_duration_s=max_duration_s,
            verbose=self.verbose,
        ).run()

        return self._finalise([packet], MODE_LIVE)[0]

    # ── Mode 2.4: Deployment ───────────────────────────────────────────────

    def run_deployment(
        self,
        source_path: str | Path,
        user_id: str = "deployment_participant",
        strip_labels: bool = True,
    ) -> PipelinePacket:
        """
        Mode 2.4 — Ingest non-annotated data for deployment inference.

        Parameters
        ----------
        source_path  : folder or CSV file
        user_id      : participant identifier (can be anonymised)
        strip_labels : remove any existing annotation columns
        """
        from mode_1_4_deployment import DeploymentIngester
        self._print_mode_header(MODE_DEPLOYMENT, source=str(source_path))

        packet = DeploymentIngester(
            source_path=source_path,
            user_id=user_id,
            strip_labels=strip_labels,
            verbose=self.verbose,
        ).run()

        return self._finalise([packet], MODE_DEPLOYMENT)[0]

    # ── Finalise: save packets + return ───────────────────────────────────

    def _finalise(
        self, packets: List[PipelinePacket], mode: str
    ) -> List[PipelinePacket]:
        """Save packets to disk and print summary."""
        if not self.save_packets:
            return packets

        if self.output_dir is not None:
            run_folder = Path(self.output_dir)
            run_folder.mkdir(parents=True, exist_ok=True)
        else:
            run_folder = next_run_folder(mode, self.out_name)

        for packet in packets:
            uid = packet.user_id
            save_dir = (run_folder / uid) if len(packets) > 1 else run_folder
            packet.save(save_dir)
            self._log(
                f"  [M2] Packet saved → {save_dir.relative_to(MODULE_DIR)}"
            )

        # Write module-level run summary
        summary = {
            "module": f"{MODULE_LABEL} v{MODULE_VERSION}",
            "mode": mode,
            "mode_desc": MODE_DESCRIPTIONS[mode],
            "n_packets": len(packets),
            "run_folder": str(run_folder),
            "completed_at": datetime.now().isoformat(),
            "packets": [
                {
                    "session_id": p.session_id,
                    "user_id": p.user_id,
                    "source_type": p.source_type,
                    "is_annotated": p.is_annotated,
                    "duration_s": p.duration_s,
                    "n_events": p.n_events,
                    "labels": p.unique_labels,
                }
                for p in packets
            ],
        }
        with open(run_folder / "module2_run_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        self._log(
            f"\n  [M2] Run complete. {len(packets)} packet(s) saved to:\n"
            f"       {run_folder}"
        )
        return packets

    def _print_mode_header(self, mode: str, **kwargs):
        self._log("\n" + "=" * 60)
        self._log(f"  MODULE 2 – Data Acquisition  |  v{MODULE_VERSION}")
        self._log(f"  Mode {mode}: {MODE_DESCRIPTIONS[mode]}")
        for k, v in kwargs.items():
            self._log(f"  {k.capitalize():<12}: {v}")
        self._log("=" * 60)

    def _log(self, msg):
        if self.verbose:
            print(msg)
