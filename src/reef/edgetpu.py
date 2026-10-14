"""Finding, installing and diagnosing the Edge TPU runtime (libedgetpu).

Google's last official libedgetpu release (2022) predates current TFLite/LiteRT, and its apt
repository is gone. We use the community builds at github.com/feranick/libedgetpu, which track
current TensorFlow and ship macOS (Intel and Apple Silicon) dylibs.

On macOS those dylibs are linked against MacPorts' libusb (``/opt/local/lib``). ``reef
setup-edgetpu`` rewrites that load path to Homebrew's libusb and re-signs the dylib ad hoc, so it
works without MacPorts and without sudo. The result goes to ``~/.cache/reef``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FERANICK_TAG = "16.0TF2.19.1-1"
FERANICK_URL = "https://github.com/feranick/libedgetpu/releases/download/{tag}/libedgetpu-16.0-tf2.19.1-1_MacOS_{arch}.zip"
CACHE = Path(os.environ.get("REEF_CACHE", Path.home() / ".cache" / "reef"))
MAC_LIB = "libedgetpu.1.dylib"


def candidate_paths() -> list[Path]:
    env = os.environ.get("REEF_EDGETPU_LIB")
    paths = [Path(env)] if env else []
    if sys.platform == "darwin":
        paths += [CACHE / MAC_LIB, Path("/opt/homebrew/lib") / MAC_LIB, Path("/usr/local/lib") / MAC_LIB]
    else:
        paths += [CACHE / "libedgetpu.so.1", Path("/usr/lib/x86_64-linux-gnu/libedgetpu.so.1"),
                  Path("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1"), Path("/usr/lib/libedgetpu.so.1")]
    return paths


def find_library() -> Path | None:
    for p in candidate_paths():
        if p.exists():
            return p
    return None


class _Device(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("path", ctypes.c_char_p)]


def list_devices(lib_path: Path | None = None) -> list[str]:
    """Enumerate Edge TPUs through libedgetpu's C API (no interpreter needed)."""
    lib_path = lib_path or find_library()
    if lib_path is None:
        return []
    lib = ctypes.CDLL(str(lib_path))
    lib.edgetpu_list_devices.restype = ctypes.POINTER(_Device)
    n = ctypes.c_size_t(0)
    devs = lib.edgetpu_list_devices(ctypes.byref(n))
    out = [f"{'usb' if devs[i].type == 1 else 'pci'}:{devs[i].path.decode()}" for i in range(n.value)]
    if n.value:
        lib.edgetpu_free_devices(devs)
    return out


def _brew_libusb() -> Path | None:
    for prefix in ("/opt/homebrew", "/usr/local"):
        p = Path(prefix) / "lib" / "libusb-1.0.0.dylib"
        if p.exists():
            return p
    try:
        res = subprocess.run(["brew", "--prefix", "libusb"], capture_output=True, text=True, check=True)
        prefix = res.stdout.strip()
        p = Path(prefix) / "lib" / "libusb-1.0.0.dylib"
        return p if p.exists() else None
    except (OSError, subprocess.CalledProcessError):
        return None


def setup_macos(throttled: bool = False, force: bool = False) -> Path:
    """Download, patch and install libedgetpu for macOS into ``~/.cache/reef``."""
    target = CACHE / MAC_LIB
    if target.exists() and not force:
        print(f"already installed: {target}")
        return target
    libusb = _brew_libusb()
    if libusb is None:
        raise SystemExit("libusb not found. Install it with:  brew install libusb")
    arch = "Silicon" if platform.machine() == "arm64" else "Intel"
    url = FERANICK_URL.format(tag=FERANICK_TAG, arch=arch)
    variant = "throttled" if throttled else "direct"
    sub = "darwin_arm64" if arch == "Silicon" else "darwin_x86_64"
    print(f"downloading {url}")
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "libedgetpu.zip"
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith(f"{variant}/{sub}/libedgetpu.1.0.dylib"))
            z.extract(member, td)
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(td) / member, target)
    target.chmod(0o755)
    deps = subprocess.run(["otool", "-L", str(target)], capture_output=True, text=True).stdout
    old = next((ln.split()[0] for ln in deps.splitlines() if "libusb" in ln), None)
    if old and old != str(libusb):
        subprocess.run(["install_name_tool", "-change", old, str(libusb), str(target)], check=True)
    subprocess.run(["install_name_tool", "-id", f"@rpath/{MAC_LIB}", str(target)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(target)], check=True, capture_output=True)
    print(f"installed {variant} libedgetpu -> {target} (libusb: {libusb})")
    return target


def setup(throttled: bool = False, force: bool = False) -> Path | None:
    if sys.platform == "darwin":
        return setup_macos(throttled, force)
    print(
        "On Linux, install libedgetpu1-std (or -max) from https://github.com/feranick/libedgetpu/releases\n"
        "  e.g.  sudo apt install ./libedgetpu1-std_16.0tf2.19.1-1.bookworm_arm64.deb\n"
        "then re-plug the accelerator so the udev rules apply."
    )
    return find_library()


def _usb_link_speed() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["system_profiler", "SPUSBHostDataType"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if "0x18d1" in ln or "0x1a6e" in ln:
            for back in lines[max(0, i - 6) : i + 6]:
                if "Link Speed" in back:
                    return back.split(":", 1)[1].strip()
    return None


def doctor() -> int:
    """Print a health report. Returns a process exit code."""
    ok = True
    lib = find_library()
    print(f"platform        : {platform.system()} {platform.machine()} / Python {platform.python_version()}")
    print(f"libedgetpu      : {lib or 'NOT FOUND - run `reef setup-edgetpu`'}")
    if sys.platform == "darwin":
        print(f"libusb          : {_brew_libusb() or 'NOT FOUND - brew install libusb'}")
    if lib is None:
        return 1
    try:
        devs = list_devices(lib)
    except OSError as e:
        print(f"devices         : failed to load library ({e})")
        return 1
    print(f"devices         : {devs or 'none found'}")
    if not devs:
        ok = False
        print("  hint: unplug/replug the Coral. On macOS also check System Settings → Privacy & Security →\n"
              "        'Allow accessories to connect'. The device reconnects with a new USB id (18d1:9302)\n"
              "        after its firmware loads, and macOS may ask again.")
    speed = _usb_link_speed()
    if speed:
        note = "  (USB 2: works, but a USB 3 port/cable cuts transfer time)" if "480" in speed else ""
        print(f"usb link speed  : {speed}{note}")
    return 0 if ok else 2
