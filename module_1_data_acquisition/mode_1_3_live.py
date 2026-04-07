"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  mode_2_3_live.py
=============================================================================
Mode 2.3 — Live Device Data Acquisition

Accepts physiological data from a wearable device in real time, buffers it,
applies event annotations, and converts the session into a PipelinePacket
for downstream training and testing.

Architecture
------------
The LiveDataCollector implements a device-agnostic streaming interface:

  Device / CSV stream
       │
       ▼
  StreamBuffer     — thread-safe ring buffer, one per signal channel
       │
       ▼
  SessionAnnotator — receives real-time event markers from the researcher
       │
       ▼
  PipelinePacket   — assembled at session end, ready for preprocessing

Device support
--------------
Currently implemented:
  - FILE_STREAM   : replay a CSV file at realtime speed (for testing)
  - EMPATICA_E4   : Empatica E4 BLE Streaming Server (TCP socket)
  - MANUAL_ENTRY  : researcher types values / triggers for small tests

The device interface is abstracted behind DeviceAdapter — adding a new
device requires only implementing the DeviceAdapter base class.

Annotation
----------
The researcher annotates emotion/behaviour events in real time by pressing
keys or sending commands during recording. Events are timestamped and
stored alongside the signal data.

=============================================================================
"""

from __future__ import annotations
import sys
import time
import threading
import queue
import json
import select
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from pipeline_packet import (
    PipelinePacket, build_packet_from_dataframes,
    SOURCE_LIVE, _new_session_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEVICE_FILE_STREAM  = "file_stream"
DEVICE_EMPATICA_E4  = "empatica_e4"
DEVICE_MANUAL       = "manual"

SAMPLING_RATES = {
    "EDA": 4, "BVP": 64, "IBI": None, "ST": 4,
    "ACC_X": 32, "ACC_Y": 32, "ACC_Z": 32,
}

try:
    from config import EMOTIONS_ALL
except ImportError:
    EMOTIONS_ALL = [
        "Happy", "Anger", "Fear", "Disgust", "Sad", "Surprise",
        "Hunger", "Thirst", "Toilet", "Tired",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# LIVE EVENT ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveEvent:
    """A researcher-annotated event during a live recording."""
    emotion:    str
    start_s:    float
    end_s:      Optional[float] = None   # None = event still active
    category:   str             = ""

    def close(self, end_s: float):
        self.end_s = end_s

    @property
    def is_open(self) -> bool:
        return self.end_s is None

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_s is None:
            return None
        return round(self.end_s - self.start_s, 3)

    def to_dict(self) -> dict:
        return {
            "emotion":    self.emotion,
            "category":   self.category,
            "start_s":    round(self.start_s, 3),
            "end_s":      round(self.end_s, 3) if self.end_s else None,
            "duration_s": self.duration_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STREAM BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class StreamBuffer:
    """
    Thread-safe buffer accumulating samples for one signal channel.

    Accumulates (timestamp_s, value) tuples as they arrive from the device.
    """

    def __init__(self, name: str):
        self.name     = name
        self._times:  list = []
        self._values: list = []
        self._lock    = threading.Lock()

    def append(self, timestamp_s: float, value):
        with self._lock:
            self._times.append(timestamp_s)
            if isinstance(value, (list, tuple)):
                self._values.append(list(value))
            else:
                self._values.append(float(value))

    def to_arrays(self):
        with self._lock:
            return (
                np.array(self._times),
                np.array(self._values),
            )

    def __len__(self):
        with self._lock:
            return len(self._times)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE ADAPTERS
# ─────────────────────────────────────────────────────────────────────────────

class DeviceAdapter(ABC):
    """Abstract base class for device adapters."""

    @abstractmethod
    def connect(self):
        """Establish connection to the device."""

    @abstractmethod
    def start_streaming(self, buffers: Dict[str, StreamBuffer]):
        """Start streaming data into the provided buffers."""

    @abstractmethod
    def stop_streaming(self):
        """Stop streaming and disconnect."""

    @property
    @abstractmethod
    def device_info(self) -> dict:
        """Return device metadata dict."""


class FileStreamAdapter(DeviceAdapter):
    """
    Replay a Module 1A combined_signals.csv at realtime speed.

    Useful for:
      - Testing the live acquisition pipeline without a physical device
      - Validating the annotation interface
      - Demonstrating the pipeline to stakeholders

    Parameters
    ----------
    csv_path     : path to combined_signals.csv from Module 1A
    speed_factor : 1.0 = realtime; 10.0 = 10x faster (for testing)
    """

    def __init__(self, csv_path: str | Path, speed_factor: float = 10.0):
        self.csv_path     = Path(csv_path)
        self.speed_factor = speed_factor
        self._thread:  Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._df:      Optional[pd.DataFrame] = None

    def connect(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self._df = pd.read_csv(self.csv_path)
        print(f"  [FileStream] Loaded {self.csv_path.name} "
              f"({len(self._df):,} rows, {self._df['timestamp_s'].max():.1f}s)")

    def start_streaming(self, buffers: Dict[str, StreamBuffer]):
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._stream_loop,
            args=(buffers,),
            daemon=True,
        )
        self._thread.start()

    def _stream_loop(self, buffers: Dict[str, StreamBuffer]):
        df       = self._df
        t_start  = time.time()
        prev_ts  = 0.0

        col_map = {
            "EDA":   ("EDA_uS",),
            "BVP":   ("BVP_nT",),
            "IBI":   ("IBI_ms",),
            "ST":    ("ST_degC",),
            "ACC":   ("ACC_X_g", "ACC_Y_g", "ACC_Z_g"),
        }

        for _, row in df.iterrows():
            if self._stop_evt.is_set():
                break

            ts  = float(row["timestamp_s"])
            dt  = (ts - prev_ts) / self.speed_factor
            if dt > 0:
                time.sleep(max(0, dt))
            prev_ts = ts

            real_ts = time.time() - t_start

            for buf_name, cols in col_map.items():
                if buf_name not in buffers:
                    continue
                present = [c for c in cols if c in row.index]
                if not present:
                    continue
                if buf_name == "IBI":
                    val = row.get("IBI_ms", np.nan)
                    if pd.notna(val):
                        buffers["IBI"].append(real_ts, float(val))
                elif buf_name == "ACC":
                    vals = [row.get(c, 0.0) for c in present]
                    buffers["ACC"].append(real_ts, vals)
                else:
                    buffers[buf_name].append(real_ts, float(row[present[0]]))

    def stop_streaming(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def device_info(self) -> dict:
        return {
            "device_type":   DEVICE_FILE_STREAM,
            "source_file":   str(self.csv_path),
            "speed_factor":  self.speed_factor,
        }


class EmpaticaE4Adapter(DeviceAdapter):
    """
    Adapter for Empatica E4 via the BLE Streaming Server (TCP).

    The E4 Streaming Server must be running on the host machine:
      https://developer.empatica.com/windows-streaming-server-usage.html

    Default: localhost:28000

    Parameters
    ----------
    host : streaming server host
    port : streaming server port (default 28000)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 28000):
        self.host    = host
        self.port    = port
        self._sock   = None
        self._thread = None
        self._stop   = threading.Event()

    def connect(self):
        import socket
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(5.0)
            print(f"  [E4 Adapter] Connected to {self.host}:{self.port}")
            # Send connect command to E4 Server
            self._send("device_connect")
            resp = self._recv()
            if "ERR" in resp:
                raise ConnectionError(f"E4 server error: {resp}")
            # Subscribe to all streams
            for stream in ["acc", "bvp", "gsr", "tmp", "ibi", "bat", "tag"]:
                self._send(f"device_subscribe {stream} ON")
                _ = self._recv()
            print("  [E4 Adapter] Subscribed to all streams.")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Empatica E4 Streaming Server "
                f"at {self.host}:{self.port}.\n"
                f"Ensure the E4 Streaming Server is running.\n"
                f"Error: {e}"
            )

    def start_streaming(self, buffers: Dict[str, StreamBuffer]):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, args=(buffers,), daemon=True
        )
        self._thread.start()

    def _send(self, msg: str):
        self._sock.sendall((msg + "\r\n").encode())

    def _recv(self) -> str:
        return self._sock.recv(1024).decode().strip()

    def _recv_loop(self, buffers: Dict[str, StreamBuffer]):
        """Parse E4 Streaming Server messages and route to buffers."""
        buf = ""
        t0  = time.time()

        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(1024).decode()
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._parse_e4_line(line, buffers, t0)
            except Exception:
                break

    def _parse_e4_line(self, line: str, buffers: Dict[str, StreamBuffer], t0: float):
        """Parse one E4 data line and route to appropriate buffer."""
        parts = line.split(",")
        if len(parts) < 3:
            return
        stream = parts[0]
        ts     = float(parts[1]) - t0

        try:
            if stream == "E4_Gsr" and "EDA" in buffers:
                buffers["EDA"].append(ts, float(parts[2]))
            elif stream == "E4_Bvp" and "BVP" in buffers:
                buffers["BVP"].append(ts, float(parts[2]))
            elif stream == "E4_Ibi" and "IBI" in buffers:
                buffers["IBI"].append(ts, float(parts[2]) * 1000)  # s→ms
            elif stream == "E4_Temperature" and "ST" in buffers:
                buffers["ST"].append(ts, float(parts[2]))
            elif stream == "E4_Acc" and "ACC" in buffers and len(parts) >= 5:
                buffers["ACC"].append(ts, [float(p) for p in parts[2:5]])
        except (ValueError, IndexError):
            pass

    def stop_streaming(self):
        self._stop.set()
        if self._sock:
            try:
                self._send("device_disconnect")
            except Exception:
                pass
            self._sock.close()

    @property
    def device_info(self) -> dict:
        return {"device_type": DEVICE_EMPATICA_E4,
                "host": self.host, "port": self.port}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SESSION ANNOTATOR  (keyboard-driven event marking)
# ─────────────────────────────────────────────────────────────────────────────

class SessionAnnotator:
    """
    Real-time event annotation during a live recording session.

    The researcher uses simple commands to mark emotion events:
      start <emotion>  — begin an event
      end              — close the current event
      list             — show all emotions available
      events           — show events logged so far
      stop             — end the recording session
    """

    def __init__(self, session_start_time: float):
        self.t0     = session_start_time
        self.events: List[LiveEvent] = []
        self._open:  Optional[LiveEvent] = None
        self._lock   = threading.Lock()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def start_event(self, emotion: str) -> str:
        emotion = emotion.strip().title()
        if emotion not in EMOTIONS_ALL:
            return (f"  Unknown emotion '{emotion}'. "
                    f"Valid: {', '.join(EMOTIONS_ALL)}")

        with self._lock:
            if self._open:
                self._open.close(self.elapsed())

            cat = "physiological_need" if emotion in [
                "Hunger","Thirst","Toilet","Tired"] else "affective"
            ev = LiveEvent(
                emotion  = emotion,
                start_s  = self.elapsed(),
                category = cat,
            )
            self.events.append(ev)
            self._open = ev
            return (f"  ▶ Event started: {emotion} "
                    f"@ t={ev.start_s:.1f}s")

    def end_event(self) -> str:
        with self._lock:
            if not self._open:
                return "  No active event to end."
            self._open.close(self.elapsed())
            ev = self._open
            self._open = None
            return (f"  ■ Event ended:   {ev.emotion} "
                    f"(duration={ev.duration_s:.1f}s)")

    def close_open_event(self):
        with self._lock:
            if self._open:
                self._open.close(self.elapsed())
                self._open = None

    def get_events(self) -> List[LiveEvent]:
        with self._lock:
            return list(self.events)

    def summary(self) -> str:
        evs = self.get_events()
        if not evs:
            return "  No events annotated."
        lines = ["  Annotated events:"]
        for ev in evs:
            dur = f"{ev.duration_s:.1f}s" if ev.duration_s else "open"
            lines.append(f"    {ev.emotion:<12} t={ev.start_s:.1f}–"
                         f"{ev.end_s:.1f if ev.end_s else '?':.1f}s  ({dur})")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DATA COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class LiveDataCollector:
    """
    Orchestrates a live recording session:
      1. Connect to device
      2. Stream data into buffers
      3. Accept real-time researcher annotations
      4. Assemble PipelinePacket at session end

    Parameters
    ----------
    device_adapter : DeviceAdapter instance (FileStreamAdapter or E4Adapter)
    user_id        : participant identifier
    max_duration_s : auto-stop after this many seconds (None = manual stop)
    verbose        : print status messages
    """

    def __init__(
        self,
        device_adapter: DeviceAdapter,
        user_id:        str            = "live_participant",
        max_duration_s: Optional[float]= None,
        verbose:        bool           = True,
    ):
        self.adapter        = device_adapter
        self.user_id        = user_id
        self.max_duration_s = max_duration_s
        self.verbose        = verbose
        self.session_id     = _new_session_id("LIV")

        # Buffers — one per channel
        self.buffers: Dict[str, StreamBuffer] = {
            name: StreamBuffer(name)
            for name in ("EDA", "BVP", "IBI", "ST", "ACC")
        }
        self._annotator: Optional[SessionAnnotator] = None
        self._t_start:   Optional[float] = None

    # ── Run session ───────────────────────────────────────────────────────

    def run(self) -> PipelinePacket:
        """Start session, collect data, return PipelinePacket."""
        self._print_header()

        # Connect
        self.adapter.connect()
        self._t_start = time.time()
        self._annotator = SessionAnnotator(self._t_start)

        # Start streaming
        self.adapter.start_streaming(self.buffers)
        self._log(f"\n  Recording started.  Session: {self.session_id}")
        self._log(f"  Device: {self.adapter.device_info}")

        if self.max_duration_s:
            self._log(f"  Auto-stop after {self.max_duration_s:.0f}s "
                      "(or type 'stop' to end early)")
        else:
            self._log("  Type 'stop' to end the recording.")

        # Annotation loop
        self._annotation_loop()

        # Stop streaming
        self.adapter.stop_streaming()
        self._annotator.close_open_event()
        elapsed = time.time() - self._t_start

        self._log(f"\n  Recording stopped. Duration: {elapsed:.1f}s")
        self._log(self._annotator.summary())

        # Build packet
        packet = self._build_packet(elapsed)
        self._log("\n" + packet.summary())
        return packet

    # ── Annotation loop ───────────────────────────────────────────────────

    def _annotation_loop(self):
        """
        Non-blocking annotation loop — accepts commands while data streams.
        Uses a background thread for auto-stop so the main thread can
        remain responsive to user input.
        """
        stop_evt = threading.Event()

        if self.max_duration_s:
            def _auto_stop():
                time.sleep(self.max_duration_s)
                stop_evt.set()
            threading.Thread(target=_auto_stop, daemon=True).start()

        self._log("\n  Commands:  start <emotion>  |  end  |  events  |  "
                  "list  |  stop\n")

        while not stop_evt.is_set():
            try:
                raw = input("  > ").strip()
            except EOFError:
                break

            if not raw:
                continue

            parts = raw.lower().split(maxsplit=1)
            cmd   = parts[0]
            arg   = parts[1] if len(parts) > 1 else ""

            if cmd == "stop":
                stop_evt.set()
                break
            elif cmd == "start":
                if not arg:
                    print("  Usage: start <emotion>")
                else:
                    print(self._annotator.start_event(arg))
            elif cmd == "end":
                print(self._annotator.end_event())
            elif cmd == "events":
                print(self._annotator.summary())
            elif cmd == "list":
                print(f"  Emotions: {', '.join(EMOTIONS_ALL)}")
            else:
                elapsed = self._annotator.elapsed()
                n_eda   = len(self.buffers["EDA"])
                n_bvp   = len(self.buffers["BVP"])
                print(f"  Unknown command '{cmd}'. "
                      f"[t={elapsed:.1f}s | EDA={n_eda} | BVP={n_bvp} samples]")

    # ── Build PipelinePacket from buffers ─────────────────────────────────

    def _build_packet(self, duration_s: float) -> PipelinePacket:
        events     = self._annotator.get_events()
        is_annotated = len(events) > 0

        def _labels(timestamps):
            n   = len(timestamps)
            lbl = np.full(n, "baseline", dtype=object)
            eid = np.zeros(n, dtype=int)
            cat = np.full(n, "baseline", dtype=object)
            for i, ev in enumerate(events):
                if ev.end_s is None:
                    continue
                mask = (timestamps >= ev.start_s) & (timestamps < ev.end_s)
                lbl[mask] = ev.emotion
                eid[mask] = i + 1
                cat[mask] = ev.category
            return lbl, eid, cat

        signals = {}
        col_map = {
            "EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms", "ST": "ST_degC"
        }

        for name, col in col_map.items():
            ts, vals = self.buffers[name].to_arrays()
            if len(ts) == 0:
                continue
            l, e, c = _labels(ts)
            signals[name] = pd.DataFrame({
                "timestamp_s": np.round(ts, 5),
                col: vals,
                "target_label": l, "event_id": e, "category": c,
            })

        # ACC (3-axis)
        ts, vals = self.buffers["ACC"].to_arrays()
        if len(ts) > 0:
            l, e, c = _labels(ts)
            try:
                signals["ACC"] = pd.DataFrame({
                    "timestamp_s": np.round(ts, 5),
                    "ACC_X_g": vals[:, 0],
                    "ACC_Y_g": vals[:, 1],
                    "ACC_Z_g": vals[:, 2],
                    "target_label": l, "event_id": e, "category": c,
                })
            except (IndexError, TypeError):
                pass

        # Build combined from BVP reference
        combined = self._build_combined(signals, duration_s, _labels)

        extra = {
            "device_info":    self.adapter.device_info,
            "duration_s":     round(duration_s, 2),
            "is_annotated":   is_annotated,
            "n_events":       len(events),
            "events":         [ev.to_dict() for ev in events],
            "recorded_at":    datetime.now().isoformat(),
        }

        return build_packet_from_dataframes(
            signals      = signals,
            combined     = combined,
            source_type  = SOURCE_LIVE,
            is_annotated = is_annotated,
            user_id      = self.user_id,
            session_id   = self.session_id,
            extra_meta   = extra,
        )

    def _build_combined(self, signals, duration_s, label_fn):
        """Build combined DataFrame from live buffers."""
        from scipy.interpolate import interp1d

        if "BVP" not in signals or len(signals["BVP"]) == 0:
            # Fall back to first available signal
            if not signals:
                return pd.DataFrame()
            ref_df = list(signals.values())[0]
        else:
            ref_df = signals["BVP"]

        t_ref  = ref_df["timestamp_s"].values
        l, e, c = label_fn(t_ref)
        combined = pd.DataFrame({
            "timestamp_s":  t_ref,
            "target_label": l, "event_id": e, "category": c,
        })

        col_map = {"EDA": "EDA_uS", "BVP": "BVP_nT",
                   "IBI": "IBI_ms", "ST": "ST_degC"}
        for sn, col in col_map.items():
            if sn in signals and col in signals[sn].columns:
                ts  = signals[sn]["timestamp_s"].values
                val = signals[sn][col].values
                if len(ts) == len(t_ref):
                    combined[col] = val
                elif len(ts) > 1:
                    f = interp1d(ts, val, kind="linear",
                                 bounds_error=False, fill_value="extrapolate")
                    combined[col] = f(t_ref)

        if "ACC" in signals:
            for ax in ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"]:
                if ax in signals["ACC"].columns:
                    ts  = signals["ACC"]["timestamp_s"].values
                    val = signals["ACC"][ax].values
                    if len(ts) > 1:
                        f = interp1d(ts, val, kind="linear",
                                     bounds_error=False, fill_value="extrapolate")
                        combined[ax] = f(t_ref)

        return combined

    def _print_header(self):
        print("\n" + "=" * 60)
        print("  MODULE 2.3 — Live Data Acquisition")
        print(f"  Participant : {self.user_id}")
        print(f"  Session ID  : {self.session_id}")
        print("=" * 60)

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def collect_from_file(
    csv_path:       str | Path,
    user_id:        str   = "test_participant",
    speed_factor:   float = 10.0,
    max_duration_s: float = None,
) -> PipelinePacket:
    """
    Replay a Module 1A CSV as a live stream (for testing).

    Useful for validating the live pipeline without a physical device.
    """
    adapter = FileStreamAdapter(csv_path, speed_factor=speed_factor)
    return LiveDataCollector(
        device_adapter  = adapter,
        user_id         = user_id,
        max_duration_s  = max_duration_s,
    ).run()


def collect_from_empatica_e4(
    user_id: str  = "participant_001",
    host:    str  = "127.0.0.1",
    port:    int  = 28000,
    max_duration_s: float = None,
) -> PipelinePacket:
    """
    Collect live data from an Empatica E4 device.

    Requires the E4 BLE Streaming Server to be running on the host machine.
    """
    adapter = EmpaticaE4Adapter(host=host, port=port)
    return LiveDataCollector(
        device_adapter  = adapter,
        user_id         = user_id,
        max_duration_s  = max_duration_s,
    ).run()
