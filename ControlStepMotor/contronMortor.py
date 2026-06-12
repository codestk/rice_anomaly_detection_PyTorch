"""
CNC 3060 step motor control UI.

This app is built for a GRBL-compatible CNC controller over USB serial.
Install pyserial before running:

    pip install pyserial

Run:

    python ControlStepMotor/contronMortor.py
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception as exc:  # pragma: no cover - import error shown in main()
    tk = None
    ttk = None
    messagebox = None
    TK_IMPORT_ERROR = exc
else:
    TK_IMPORT_ERROR = None

try:
    import serial
    from serial.tools import list_ports
except Exception:  # pragma: no cover - app can still open in demo mode
    serial = None
    list_ports = None


APP_TITLE = "CNC 3060 Step Motor Controller"
DEFAULT_BAUD = 115200
FALLBACK_BAUDS = ("115200", "57600", "38400", "19200", "9600")


def clean_gcode_line(line: str) -> str:
    """Remove common comments before sending one G-code line."""
    line = re.sub(r"\(.*?\)", "", line)
    line = line.split(";", 1)[0]
    return line.strip()


def parse_grbl_status(line: str) -> dict[str, object]:
    """
    Parse GRBL status lines such as:
    <Idle|MPos:1.000,2.000,3.000|WPos:0.000,0.000,0.000|FS:0,0>
    """
    result: dict[str, object] = {}
    if not (line.startswith("<") and line.endswith(">")):
        return result

    parts = line[1:-1].split("|")
    if parts:
        result["state"] = parts[0]

    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if key in {"MPos", "WPos"}:
            try:
                result[key] = tuple(float(item) for item in value.split(",")[:3])
            except ValueError:
                pass
        elif key == "FS":
            try:
                feed, spindle = value.split(",", 1)
                result["feed"] = float(feed)
                result["spindle"] = float(spindle)
            except ValueError:
                pass
    return result


@dataclass
class MachineState:
    status: str = "Disconnected"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    feed: float = 0.0
    spindle: float = 0.0


class SerialController:
    def __init__(
        self,
        on_line: Callable[[str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._serial = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._on_line = on_line
        self._on_status = on_status
        self._on_error = on_error

    @property
    def connected(self) -> bool:
        return bool(self._serial and self._serial.is_open)

    def connect(self, port: str, baud: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        if not port:
            raise RuntimeError("Select a COM port first.")

        self.disconnect()
        self._stop_event.clear()
        self._serial = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.08,
            write_timeout=3.0,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )
        time.sleep(1.8)
        self._serial.reset_input_buffer()

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._on_status(f"Connected {port} @ {baud}")

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._serial:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None
        self._on_status("Disconnected")

    def send_line(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        self._write((command + "\n").encode("ascii", errors="ignore"))

    def send_realtime(self, data: bytes) -> None:
        self._write(data)

    def _write(self, data: bytes) -> None:
        if not self.connected:
            raise RuntimeError("Serial is not connected.")
        with self._write_lock:
            self._serial.write(data)
            self._serial.flush()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self._serial or not self._serial.is_open:
                    break
                raw = self._serial.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        self._on_line(line)
            except Exception as exc:
                self._on_error(str(exc))
                break


class CncControllerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x760")
        self.root.minsize(1060, 700)

        self.state = MachineState()
        self.message_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.program_stop = threading.Event()
        self.program_thread: Optional[threading.Thread] = None
        self.port_devices: dict[str, str] = {}

        self.serial = SerialController(
            on_line=lambda line: self.message_queue.put(("line", line)),
            on_status=lambda text: self.message_queue.put(("status", text)),
            on_error=lambda text: self.message_queue.put(("error", text)),
        )

        self._build_style()
        self._build_ui()
        self.refresh_ports()
        self._poll_messages()
        self._poll_machine_status()

    def _build_style(self) -> None:
        self.colors = {
            "bg": "#f5f7fa",
            "panel": "#ffffff",
            "text": "#1f2937",
            "muted": "#64748b",
            "border": "#d8dee9",
            "accent": "#2563eb",
            "danger": "#dc2626",
            "ok": "#16a34a",
            "warn": "#ca8a04",
        }
        self.root.configure(bg=self.colors["bg"])

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"], relief="solid", borderwidth=1)
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtle.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Position.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Consolas", 20, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 7))
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 7))
        style.map("Accent.TButton", foreground=[("active", "#ffffff"), ("!disabled", "#ffffff")], background=[("active", "#1d4ed8"), ("!disabled", self.colors["accent"])])
        style.map("Danger.TButton", foreground=[("active", "#ffffff"), ("!disabled", "#ffffff")], background=[("active", "#b91c1c"), ("!disabled", self.colors["danger"])])
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure("TLabelframe", background=self.colors["panel"], bordercolor=self.colors["border"])
        style.configure("TLabelframe.Label", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Disconnected")
        self.status_label = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#fee2e2",
            fg="#991b1b",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=5,
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        self._build_connection_panel(shell)

        main = ttk.Frame(shell)
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        self._build_position_panel(left)
        self._build_jog_panel(left)
        self._build_zero_panel(left)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self._build_machine_panel(right)
        self._build_goto_panel(right)
        self._build_console_panel(right)

    def _build_connection_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for col in range(9):
            panel.columnconfigure(col, weight=0)
        panel.columnconfigure(8, weight=1)

        ttk.Label(panel, text="Port", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(panel, textvariable=self.port_var, width=42)
        self.port_combo.grid(row=0, column=1, sticky="w", padx=(8, 8))

        ttk.Button(panel, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=(0, 12))

        ttk.Label(panel, text="Baud", style="Panel.TLabel").grid(row=0, column=3, sticky="w")
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        self.baud_combo = ttk.Combobox(panel, textvariable=self.baud_var, width=10, values=FALLBACK_BAUDS)
        self.baud_combo.grid(row=0, column=4, sticky="w", padx=(8, 12))

        self.connect_button = ttk.Button(panel, text="Connect", style="Accent.TButton", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=5, padx=(0, 8))
        ttk.Button(panel, text="Disconnect", command=self.disconnect).grid(row=0, column=6, padx=(0, 12))

        self.lock_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Clamp to 3060 work area", variable=self.lock_check_var).grid(row=0, column=7, sticky="w")

    def _build_position_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Work Position", padding=12)
        panel.pack(fill="x", pady=(0, 12))

        self.position_vars = {
            "X": tk.StringVar(value="0.000"),
            "Y": tk.StringVar(value="0.000"),
            "Z": tk.StringVar(value="0.000"),
        }

        for row, axis in enumerate(("X", "Y", "Z")):
            ttk.Label(panel, text=axis, style="Panel.TLabel", font=("Segoe UI", 12, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Label(panel, textvariable=self.position_vars[axis], style="Position.TLabel", width=9).grid(row=row, column=1, sticky="e", pady=4)
            ttk.Label(panel, text="mm", style="Subtle.TLabel").grid(row=row, column=2, sticky="w", padx=(6, 0), pady=4)

        meta = ttk.Frame(panel, style="Panel.TFrame")
        meta.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.machine_state_var = tk.StringVar(value="State: -")
        self.feed_state_var = tk.StringVar(value="Feed: 0 | Spindle: 0")
        ttk.Label(meta, textvariable=self.machine_state_var, style="Subtle.TLabel").pack(anchor="w")
        ttk.Label(meta, textvariable=self.feed_state_var, style="Subtle.TLabel").pack(anchor="w", pady=(4, 0))

    def _build_jog_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Jog Control", padding=12)
        panel.pack(fill="x", pady=(0, 12))

        self.step_var = tk.StringVar(value="10")
        self.feed_var = tk.StringVar(value="800")
        ttk.Label(panel, text="Step mm", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self.step_var, width=8, values=("0.1", "1", "5", "10", "50", "100")).grid(row=0, column=1, sticky="w", padx=(8, 12))
        ttk.Label(panel, text="Feed", style="Panel.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(panel, from_=50, to=5000, increment=50, textvariable=self.feed_var, width=8).grid(row=0, column=3, sticky="w", padx=(8, 0))

        pad = ttk.Frame(panel, style="Panel.TFrame")
        pad.grid(row=1, column=0, columnspan=4, pady=(14, 8))

        ttk.Button(pad, text="Y+", command=lambda: self.jog("Y", 1)).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(pad, text="X-", command=lambda: self.jog("X", -1)).grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(pad, text="Stop", style="Danger.TButton", command=self.jog_cancel).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(pad, text="X+", command=lambda: self.jog("X", 1)).grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(pad, text="Y-", command=lambda: self.jog("Y", -1)).grid(row=2, column=1, padx=4, pady=4, sticky="ew")

        zpad = ttk.Frame(panel, style="Panel.TFrame")
        zpad.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(zpad, text="Z+", command=lambda: self.jog("Z", 1)).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(zpad, text="Z-", command=lambda: self.jog("Z", -1)).pack(side="left", expand=True, fill="x", padx=(4, 0))

        limit = ttk.Frame(panel, style="Panel.TFrame")
        limit.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        self.max_x_var = tk.StringVar(value="300")
        self.max_y_var = tk.StringVar(value="600")
        self.max_z_var = tk.StringVar(value="80")
        for col, (label, var) in enumerate((("Max X", self.max_x_var), ("Max Y", self.max_y_var), ("Max Z", self.max_z_var))):
            ttk.Label(limit, text=label, style="Subtle.TLabel").grid(row=0, column=col * 2, sticky="w", padx=(0, 4))
            ttk.Entry(limit, textvariable=var, width=6).grid(row=0, column=col * 2 + 1, sticky="w", padx=(0, 8))

    def _build_zero_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Set Work Zero", padding=12)
        panel.pack(fill="x")

        ttk.Button(panel, text="Zero X", command=lambda: self.send_command("G10 L20 P1 X0")).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(panel, text="Zero Y", command=lambda: self.send_command("G10 L20 P1 Y0")).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(panel, text="Zero Z", command=lambda: self.send_command("G10 L20 P1 Z0")).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(panel, text="Zero All", style="Accent.TButton", command=lambda: self.send_command("G10 L20 P1 X0 Y0 Z0")).grid(row=1, column=0, columnspan=3, padx=4, pady=(8, 4), sticky="ew")

        for col in range(3):
            panel.columnconfigure(col, weight=1)

    def _build_machine_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Machine", padding=12)
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        controls = [
            ("Unlock", lambda: self.send_command("$X")),
            ("Home", lambda: self.send_command("$H")),
            ("Status", self.query_status),
            ("Hold", lambda: self.send_realtime(b"!")),
            ("Resume", lambda: self.send_realtime(b"~")),
            ("Reset", lambda: self.send_realtime(b"\x18")),
            ("Spindle On", self.spindle_on),
            ("Spindle Off", lambda: self.send_command("M5")),
            ("Coolant On", lambda: self.send_command("M8")),
            ("Coolant Off", lambda: self.send_command("M9")),
        ]

        for index, (text, command) in enumerate(controls):
            style = "Danger.TButton" if text in {"Hold", "Reset"} else "TButton"
            ttk.Button(panel, text=text, style=style, command=command).grid(row=index // 5, column=index % 5, padx=4, pady=4, sticky="ew")

        ttk.Label(panel, text="Spindle RPM", style="Panel.TLabel").grid(row=2, column=0, sticky="w", padx=4, pady=(12, 4))
        self.spindle_speed_var = tk.StringVar(value="1000")
        ttk.Spinbox(panel, from_=0, to=24000, increment=500, textvariable=self.spindle_speed_var, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=(12, 4))

        for col in range(5):
            panel.columnconfigure(col, weight=1)

    def _build_goto_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Move To Work Position", padding=12)
        panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.goto_x_var = tk.StringVar(value="0")
        self.goto_y_var = tk.StringVar(value="0")
        self.goto_z_var = tk.StringVar(value="0")
        self.goto_feed_var = tk.StringVar(value="800")
        self.goto_mode_var = tk.StringVar(value="G0")

        fields = (("X", self.goto_x_var), ("Y", self.goto_y_var), ("Z", self.goto_z_var), ("Feed", self.goto_feed_var))
        for index, (label, var) in enumerate(fields):
            ttk.Label(panel, text=label, style="Panel.TLabel").grid(row=0, column=index * 2, sticky="w", padx=(0, 6))
            ttk.Entry(panel, textvariable=var, width=10).grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 12))

        ttk.Radiobutton(panel, text="Rapid G0", variable=self.goto_mode_var, value="G0").grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Radiobutton(panel, text="Feed G1", variable=self.goto_mode_var, value="G1").grid(row=1, column=2, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(panel, text="Go", style="Accent.TButton", command=self.goto_position).grid(row=1, column=6, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_console_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="G-code Console", padding=12)
        panel.grid(row=2, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        quick = ttk.Frame(panel, style="Panel.TFrame")
        quick.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        quick.columnconfigure(1, weight=1)
        self.command_var = tk.StringVar()
        ttk.Label(quick, text="Command", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        command_entry = ttk.Entry(quick, textvariable=self.command_var)
        command_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        command_entry.bind("<Return>", lambda _event: self.send_manual_command())
        ttk.Button(quick, text="Send", command=self.send_manual_command).grid(row=0, column=2, sticky="e")

        body = ttk.PanedWindow(panel, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew")

        log_frame = ttk.Frame(body, style="Panel.TFrame")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame,
            height=18,
            wrap="word",
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            font=("Consolas", 10),
            relief="flat",
            padx=10,
            pady=10,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        program_frame = ttk.Frame(body, style="Panel.TFrame")
        program_frame.columnconfigure(0, weight=1)
        program_frame.rowconfigure(0, weight=1)
        self.program_text = tk.Text(
            program_frame,
            height=18,
            wrap="none",
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            font=("Consolas", 10),
            relief="flat",
            padx=10,
            pady=10,
        )
        self.program_text.insert("1.0", "G21\nG90\n; paste G-code here\n")
        self.program_text.grid(row=0, column=0, sticky="nsew")
        program_scroll = ttk.Scrollbar(program_frame, command=self.program_text.yview)
        program_scroll.grid(row=0, column=1, sticky="ns")
        self.program_text.configure(yscrollcommand=program_scroll.set)

        body.add(log_frame, weight=1)
        body.add(program_frame, weight=1)

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Run Program", style="Accent.TButton", command=self.run_program).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Stop Program", style="Danger.TButton", command=self.stop_program).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Clear Log", command=lambda: self.log_text.delete("1.0", "end")).pack(side="right")

    def refresh_ports(self) -> None:
        self.port_devices = {}
        displays = []
        if list_ports is not None:
            for port in list_ports.comports():
                description = port.description or "Serial device"
                display = f"{port.device} - {description}"
                if port.manufacturer:
                    display += f" ({port.manufacturer})"
                self.port_devices[display] = port.device
                displays.append(display)

        self.port_combo["values"] = displays
        if displays and not self._selected_port_device():
            self.port_var.set(self._best_port_display(displays))
        self.log(f"Ports: {', '.join(displays) if displays else 'none found'}")

    def _best_port_display(self, displays: list[str]) -> str:
        likely_words = ("ch340", "ch341", "usb-serial", "usb serial", "arduino", "wch", "silicon labs", "ftdi", "serial")
        for display in displays:
            lowered = display.lower()
            if any(word in lowered for word in likely_words):
                return display
        return displays[0]

    def _selected_port_device(self) -> str:
        selected = self.port_var.get().strip()
        if not selected:
            return ""
        if selected in self.port_devices:
            return self.port_devices[selected]
        match = re.match(r"^([A-Za-z]+[0-9]+|/dev/\S+)", selected)
        return match.group(1) if match else selected

    def toggle_connection(self) -> None:
        if self.serial.connected:
            self.disconnect()
            return
        try:
            self.serial.connect(self._selected_port_device(), int(self.baud_var.get()))
            self.connect_button.configure(text="Connected")
            self.query_status()
        except Exception as exc:
            self.show_error(str(exc))

    def disconnect(self) -> None:
        self.serial.disconnect()
        self.connect_button.configure(text="Connect")

    def send_command(self, command: str) -> None:
        self.log(f">> {command}")
        try:
            self.serial.send_line(command)
        except Exception as exc:
            self.show_error(str(exc))

    def send_realtime(self, payload: bytes) -> None:
        label = payload.decode("ascii", errors="replace")
        self.log(f">> realtime {label!r}")
        try:
            self.serial.send_realtime(payload)
        except Exception as exc:
            self.show_error(str(exc))

    def send_manual_command(self) -> None:
        command = self.command_var.get().strip()
        if command:
            self.send_command(command)
            self.command_var.set("")

    def query_status(self) -> None:
        self.send_realtime(b"?")

    def jog_cancel(self) -> None:
        self.send_realtime(b"\x85")

    def spindle_on(self) -> None:
        try:
            rpm = max(0, int(float(self.spindle_speed_var.get())))
        except ValueError:
            self.show_error("Spindle RPM must be a number.")
            return
        self.send_command(f"M3 S{rpm}")

    def jog(self, axis: str, direction: int) -> None:
        try:
            step = abs(float(self.step_var.get()))
            feed = max(1.0, float(self.feed_var.get()))
        except ValueError:
            self.show_error("Step and feed must be numbers.")
            return

        delta = step * direction
        if self.lock_check_var.get() and not self._jog_inside_limits(axis, delta):
            self.show_error(f"{axis} move would exceed CNC 3060 work area.")
            return

        self.send_command(f"$J=G91 G21 {axis}{delta:.3f} F{feed:.0f}")

    def _jog_inside_limits(self, axis: str, delta: float) -> bool:
        try:
            max_x = float(self.max_x_var.get())
            max_y = float(self.max_y_var.get())
            max_z = float(self.max_z_var.get())
        except ValueError:
            return True

        current = {"X": self.state.x, "Y": self.state.y, "Z": self.state.z}[axis]
        target = current + delta
        limits = {"X": (0.0, max_x), "Y": (0.0, max_y), "Z": (-max_z, max_z)}
        low, high = limits[axis]
        return low <= target <= high

    def goto_position(self) -> None:
        try:
            x = float(self.goto_x_var.get())
            y = float(self.goto_y_var.get())
            z = float(self.goto_z_var.get())
            feed = max(1.0, float(self.goto_feed_var.get()))
        except ValueError:
            self.show_error("X, Y, Z and feed must be numbers.")
            return

        if self.lock_check_var.get() and not self._target_inside_limits(x, y, z):
            self.show_error("Target position exceeds CNC 3060 work area.")
            return

        mode = self.goto_mode_var.get()
        if mode == "G0":
            self.send_command(f"G90 G21 G0 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        else:
            self.send_command(f"G90 G21 G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.0f}")

    def _target_inside_limits(self, x: float, y: float, z: float) -> bool:
        try:
            max_x = float(self.max_x_var.get())
            max_y = float(self.max_y_var.get())
            max_z = float(self.max_z_var.get())
        except ValueError:
            return True
        return 0.0 <= x <= max_x and 0.0 <= y <= max_y and -max_z <= z <= max_z

    def run_program(self) -> None:
        if self.program_thread and self.program_thread.is_alive():
            self.show_error("Program is already running.")
            return
        lines = self.program_text.get("1.0", "end").splitlines()
        commands = [clean_gcode_line(line) for line in lines]
        commands = [command for command in commands if command]
        if not commands:
            self.show_error("No G-code commands to run.")
            return

        self.program_stop.clear()
        self.program_thread = threading.Thread(target=self._program_worker, args=(commands,), daemon=True)
        self.program_thread.start()

    def _program_worker(self, commands: list[str]) -> None:
        self.message_queue.put(("log", f"Program start: {len(commands)} lines"))
        for command in commands:
            if self.program_stop.is_set():
                self.message_queue.put(("log", "Program stopped by user."))
                break
            try:
                self.message_queue.put(("log", f">> {command}"))
                self.serial.send_line(command)
                time.sleep(0.08)
            except Exception as exc:
                self.message_queue.put(("error", str(exc)))
                break
        else:
            self.message_queue.put(("log", "Program finished."))

    def stop_program(self) -> None:
        self.program_stop.set()
        try:
            self.serial.send_realtime(b"!")
        except Exception:
            pass

    def _poll_messages(self) -> None:
        while True:
            try:
                kind, text = self.message_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                self.handle_serial_line(text)
            elif kind == "status":
                self.set_status(text)
            elif kind == "error":
                self.show_error(text)
            elif kind == "log":
                self.log(text)
        self.root.after(80, self._poll_messages)

    def _poll_machine_status(self) -> None:
        if self.serial.connected:
            try:
                self.serial.send_realtime(b"?")
            except Exception:
                pass
        self.root.after(700, self._poll_machine_status)

    def handle_serial_line(self, line: str) -> None:
        self.log(f"<< {line}")
        parsed = parse_grbl_status(line)
        if not parsed:
            return

        self.state.status = str(parsed.get("state", self.state.status))
        position = parsed.get("WPos") or parsed.get("MPos")
        if isinstance(position, tuple) and len(position) >= 3:
            self.state.x, self.state.y, self.state.z = position[:3]
        self.state.feed = float(parsed.get("feed", self.state.feed))
        self.state.spindle = float(parsed.get("spindle", self.state.spindle))
        self.update_position_ui()

    def update_position_ui(self) -> None:
        self.position_vars["X"].set(f"{self.state.x:.3f}")
        self.position_vars["Y"].set(f"{self.state.y:.3f}")
        self.position_vars["Z"].set(f"{self.state.z:.3f}")
        self.machine_state_var.set(f"State: {self.state.status}")
        self.feed_state_var.set(f"Feed: {self.state.feed:.0f} | Spindle: {self.state.spindle:.0f}")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        connected = text.startswith("Connected")
        if connected:
            self.status_label.configure(bg="#dcfce7", fg="#166534")
        else:
            self.status_label.configure(bg="#fee2e2", fg="#991b1b")

    def show_error(self, text: str) -> None:
        self.log(f"!! {text}")
        if messagebox is not None:
            messagebox.showwarning(APP_TITLE, text)

    def log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {text}\n")
        self.log_text.see("end")

    def close(self) -> None:
        self.program_stop.set()
        self.serial.disconnect()
        self.root.destroy()


def main() -> int:
    if tk is None:
        print("tkinter is not available in this Python installation.")
        print(f"Import error: {TK_IMPORT_ERROR}")
        print("On Windows, install Python from python.org with Tcl/Tk enabled.")
        print("On Ubuntu/WSL, install it with: sudo apt install python3-tk")
        return 1

    root = tk.Tk()
    app = CncControllerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
