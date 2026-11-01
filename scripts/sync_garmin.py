#!/usr/bin/env python3
"""Download ORIGINAL Hammock activities, preserving bytes and auditing raw FIT data.

No watch writes, activity edits, uploads or deletes. See docs/GARMIN_CONNECT_SYNC.md.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from getpass import getpass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
import zipfile
import zlib

import numpy as np
from fitparse import FitFile
from garminconnect import (
    Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError,
    GarminConnectNotFoundError, GarminConnectTooManyRequestsError,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "offline_processing"))
from fit_recording import expand_acceleration_batch  # noqa: E402
from fit_bbi import decode_snapshots  # noqa: E402

LOG = logging.getLogger("hammock.sync")
DEFAULT_OUTPUT = ROOT / "offline_processing" / "data" / "garmin_connect"
DEFAULT_SESSION = Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".local"))) / "Hammock" / "garmin-session.bin"
MANIFEST = ".garmin-sync-manifest.json"
MAX_BYTES = 512 * 1024 * 1024
REAUTH = "Interactive reauthentication required: run scripts/sync_garmin.py auth in a terminal."


class SyncError(Exception):
    """A safe, public diagnostic, containing no upstream response text."""


class ReauthenticationRequired(SyncError):
    pass


class RateLimited(SyncError):
    pass


def atomic_write(path: Path, data: bytes):
    """Write on the destination volume; publish only after flush/fsync succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".hammock-", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dpapi(data: bytes, *, decrypt=False) -> bytes:
    """Windows user-scoped encryption. Never place a plaintext token on disk."""
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ReauthenticationRequired("Cannot access the encrypted session under this Windows account. " + REAUTH)
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


class SessionStore:
    def __init__(self, path):
        self.path = Path(path).expanduser().absolute()

    def check_path(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise SyncError("Session paths must not contain symbolic links.")

    def read(self):
        self.check_path()
        try:
            data = self.path.read_bytes()
            if os.name == "nt" and data.startswith(b"DPAPI1\0"):
                return dpapi(data[7:], decrypt=True).decode("utf-8")
            if os.name == "posix" and data.startswith(b"POSIX1\0"):
                if self.path.stat().st_mode & 0o077 or self.path.parent.stat().st_mode & 0o077:
                    raise SyncError("Session file/directory must be owner-only (chmod 600/700).")
                return data[7:].decode("utf-8")
        except (OSError, UnicodeError):
            raise ReauthenticationRequired(REAUTH) from None
        raise ReauthenticationRequired(REAUTH)

    def save(self, api):
        self.check_path()
        payload = api.client.dumps().encode("utf-8")
        if not json.loads(payload).get("di_token"):
            raise ReauthenticationRequired("Garmin returned a session without persistable DI tokens. " + REAUTH)
        if os.name == "nt":
            data = b"DPAPI1\0" + dpapi(payload)
        elif os.name == "posix":
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.path.parent.chmod(0o700)
            data = b"POSIX1\0" + payload
        else:
            raise SyncError("Secure session storage is unsupported on this OS.")
        atomic_write(self.path, data)
        if os.name == "posix":
            self.path.chmod(0o600)


def status_code(error):
    """Use response metadata, then the library's documented error prefix only."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    match = re.search(r"(?:API Error|Error|HTTP)\s*(\d{3})", str(error))
    return int(match[1]) if match else None


def retry_after(error):
    # Look through causes because the library's 429 translation may drop response.
    for _ in range(8):
        if error is None:
            break
        response = getattr(error, "response", None)
        value = getattr(response, "headers", {}).get("Retry-After")
        if value is not None:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError):
                    return 60.0
        error = error.__cause__
    return 60.0


def api_call(operation, *, persist=None, attempts=3, sleep=time.sleep):
    """Bounded retries; no retry of auth/access failures and no raw exception logs."""
    for attempt in range(attempts):
        try:
            return operation()
        except GarminConnectAuthenticationError:
            raise ReauthenticationRequired(REAUTH) from None
        except GarminConnectNotFoundError:
            raise SyncError("Garmin activity/original file is not available; check phone sync or retry later.") from None
        except (GarminConnectTooManyRequestsError, GarminConnectConnectionError) as error:
            status = status_code(error)
            if status in (401, 403):
                raise ReauthenticationRequired(REAUTH) from None
            limited = isinstance(error, GarminConnectTooManyRequestsError) or status in (427, 429)
            if limited:
                delay = retry_after(error)
                # Do not shorten a server cooldown to fit our bounded run.
                if attempt == attempts - 1 or delay > 60 or not np.isfinite(delay):
                    raise RateLimited("Garmin rate limit: sync stopped; retry later after the server cooldown.") from None
            else:
                if status is not None and not 500 <= status <= 599:
                    raise SyncError("Garmin denied the request or original download is unavailable.") from None
                if attempt == attempts - 1:
                    raise SyncError("Garmin connection failed after bounded retries; retry on the next scheduled run.") from None
                delay = 2 ** (attempt + 1)
            LOG.warning("Temporary Garmin failure; retry %d/%d in %.0f seconds", attempt + 2, attempts, delay)
            sleep(delay)
        finally:
            if persist is not None:
                persist()


def authenticate(store, *, interactive=False):
    if interactive:
        if not sys.stdin.isatty():
            raise ReauthenticationRequired("The auth command requires an interactive terminal.")
        # getpass hides the account identifier as well as password/MFA.
        api = Garmin(email=getpass("Garmin email (hidden): ").strip(),
                     password=getpass("Garmin password: "),
                     prompt_mfa=lambda: getpass("Garmin MFA code: ").strip(), retry_attempts=0)
        # Explicit inline empty state prevents GARMINTOKENS from loading an
        # unrelated plaintext tokenstore from the environment.
        try:
            api_call(lambda: api.login("{}"), attempts=1)
        finally:
            api.password = None
    else:
        api = Garmin(retry_attempts=0)
        # A token refresh may succeed before a later profile request fails.
        # Persist that rotation even on failure and read the latest state on retry.
        api_call(lambda: api.login(store.read()),
                 persist=lambda: store.save(api) if api.client.di_token else None)
    store.save(api)
    return api


@contextmanager
def output_lock(directory, name=".garmin-sync.lock"):
    """OS-held lock releases on process exit, including a killed scheduled task."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SyncError("Another sync is using this output directory.") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def extract_original(payload):
    """Return one unchanged FIT. Reject ambiguous archives; never extract paths."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_BYTES:
        raise SyncError("Original download is empty, invalid, or exceeds the 512 MiB safety limit.")
    if payload.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                files = [entry for entry in archive.infolist() if not entry.is_dir()]
                fits = [entry for entry in files if entry.filename.casefold().endswith(".fit")]
                if len(fits) != 1:
                    raise SyncError("Original ZIP must contain exactly one FIT; missing or ambiguous recording.")
                entry = fits[0]
                if entry.file_size > MAX_BYTES or entry.flag_bits & 1:
                    raise SyncError("Original FIT is oversized or encrypted.")
                with archive.open(entry) as stream:
                    result = stream.read(MAX_BYTES + 1)
                if len(result) != entry.file_size or len(result) > MAX_BYTES:
                    raise SyncError("Original ZIP contains an incomplete FIT.")
                return result
        except (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError, zlib.error):
            raise SyncError("Original ZIP is corrupted or unsupported.") from None
    return payload


def hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint_update(digest, value):
    # Include raw values, definition identities, units, message boundaries and
    # developer descriptors. Do not normalize, interpolate, rescale or sort samples.
    digest.update(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode("utf-8"))
    digest.update(b"\n")


def raw_fields(message):
    return [(f.def_num, f.field_type, f.base_type.identifier,
             getattr(f.field, "dev_data_index", None), f.name, f.raw_value, f.units)
            for f in message]


def normalize_packed_snapshot(name, fields):
    """Adapt schema 4 to the existing sequence decoder, without changing FIT bytes.

    main's decoder supports schemas 2/3. This mapping only names the atomic
    packed slots; sequence recovery remains in fit_bbi.decode_snapshots.
    """
    prefix = "final_" if name == "session" else ""
    packed = fields.get(prefix + "bbi_snapshot")
    if packed is None:
        return fields
    if (not isinstance(packed, (list, tuple)) or len(packed) != 56 or packed[0] != 4
            or any(not isinstance(v, int) or not 0 <= v < 4294967295 for v in packed)):
        raise ValueError("Invalid packed BBI snapshot")
    result = {key: value for key, value in fields.items() if key != prefix + "logger_meta"}
    result.update({prefix + key: value for key, value in {
        "logger_schema": 2, "bbi_total": packed[1], "bbi_history_count": min(packed[1], 24),
        "elapsed_ms": packed[6], "start_unix_s": packed[7],
        "bbi_history": packed[8:32], "bbi_rx_ms": packed[32:56],
    }.items()})
    return result


def validate_fit(path):
    """Separate FIT transport validity from conservative raw research validity."""
    report = {"fit_valid": False, "research_valid": False, "issues": [],
              "acceleration_samples": 0, "bbi_count": 0, "duration_seconds": None}
    path = Path(path)
    try:
        with path.open("rb") as stream:
            header = stream.read(14)
        if (len(header) < 14 or header[0] not in (12, 14) or header[8:12] != b".FIT"
                or path.stat().st_size != header[0] + struct.unpack_from("<I", header, 4)[0] + 2):
            raise ValueError("FIT size/header mismatch")
        with FitFile(str(path), check_crc=True) as fit:
            fit.parse()  # Includes complete data and CRC; no apparently valid truncation.
            report["fit_valid"] = True
            accel_hash, developer_hash = hashlib.sha256(), hashlib.sha256()
            snapshots, summaries, schemas = [], [], set()
            first_time = last_time = None
            acceleration_present = False
            for message in fit.get_messages():
                fields = message.get_values()
                if message.name == "accelerometer_data":
                    acceleration_present = True
                    fingerprint_update(accel_hash, raw_fields(message))
                    try:
                        times, xyz = expand_acceleration_batch(fields)
                        report["acceleration_samples"] += len(times)
                        if not np.isfinite(xyz).all():
                            report["issues"].append("nonfinite_acceleration")
                        if not len(times) or (np.diff(times) <= 0).any() or (last_time is not None and times[0] <= last_time):
                            report["issues"].append("nonmonotonic_or_insufficient_acceleration")
                        if len(times):
                            first_time = times[0] if first_time is None else first_time
                            last_time = times[-1]
                    except (ValueError, TypeError, OverflowError):
                        report["issues"].append("missing_or_invalid_raw_xyz_or_sample_timing")
                developer_fields = [f for f in message if f.field_type == "devfield"]
                if developer_fields or message.name in ("field_description", "developer_data_id"):
                    fingerprint_update(developer_hash, [message.name, raw_fields(developer_fields or message)])
                if message.name in ("record", "session"):
                    prefix = "final_" if message.name == "session" else ""
                    packed = fields.get(prefix + "bbi_snapshot")
                    metadata = fields.get(prefix + "logger_meta")
                    normalized = normalize_packed_snapshot(message.name, fields)
                    if packed is not None:
                        schemas.add(4)
                    elif metadata is not None:
                        schemas.add(metadata[0])
                    elif fields.get(prefix + "logger_schema") is not None:
                        schemas.add(fields[prefix + "logger_schema"])
                    # Older decoder does not check monotonicity/elapsed bounds.
                    total = normalized.get(prefix + "bbi_total")
                    history = normalized.get(prefix + "bbi_rx_ms")
                    if metadata is not None:
                        elapsed = metadata[6]
                    else:
                        elapsed = normalized.get(prefix + "elapsed_ms")
                    if total is not None and history is not None:
                        arrivals = [history[(s - 1) % 24] for s in range(max(1, total - 23), total + 1)]
                        if (any(v is None or v < 0 for v in arrivals)
                                or any(a > b for a, b in zip(arrivals, arrivals[1:]))
                                or (elapsed is not None and any(v > elapsed for v in arrivals))):
                            raise ValueError("Inconsistent BBI callback timing")
                    snapshots.append((message.name, normalized))
                    if message.name == "session":
                        summaries.append((fields, packed, metadata))
            report["acceleration_fingerprint"] = accel_hash.hexdigest()
            report["developer_fingerprint"] = developer_hash.hexdigest()
        if not acceleration_present:
            report["issues"].append("missing_raw_accelerometry")
        if first_time is not None:
            report["duration_seconds"] = float((last_time - first_time) / 1e9)
        if report["acceleration_samples"] < 2:
            report["issues"].append("insufficient_acceleration_samples")
        if len(schemas) > 1:
            raise ValueError("Mixed BBI schemas")
        rows, bbi = decode_snapshots(snapshots)
        if schemas == {4}:
            bbi["schema"] = 4
        report["bbi"] = bbi
        report["bbi_count"] = len(rows)
        if bbi["schema"] is None:
            report["issues"].append("missing_bbi_developer_histories")
        if not rows:
            report["issues"].append("no_bbi_intervals")
        if not bbi["final_summary_present"] or len(summaries) != 1:
            report["issues"].append("missing_or_ambiguous_final_summary")
        if bbi["missing_received_intervals"]:
            report["issues"].append("missing_bbi_sequences")
        if bbi["invalid_recovered_intervals"]:
            report["issues"].append("invalid_bbi_intervals")
        if len(summaries) == 1:
            fields, packed, metadata = summaries[0]
            native_count = packed[5] if packed is not None else (metadata[5] if metadata is not None else fields.get("final_logged_accel_samples"))
            elapsed_ms = packed[6] if packed is not None else (metadata[6] if metadata is not None else fields.get("final_elapsed_ms"))
            anchor = packed[7] if packed is not None else (metadata[7] if metadata is not None else fields.get("final_start_unix_s"))
            report["logger_acceleration_samples"] = native_count
            report["logger_duration_seconds"] = elapsed_ms / 1000 if elapsed_ms is not None else None
            if native_count is None or native_count != report["acceleration_samples"]:
                report["issues"].append("native_acceleration_count_mismatch_or_missing")
            # At most two seconds of startup/flush slack; this is structural,
            # not a physiological coverage or calibrated sample-rate claim.
            if elapsed_ms is None or elapsed_ms <= 0:
                report["issues"].append("missing_recording_duration")
            elif report["duration_seconds"] is not None and abs(report["duration_seconds"] - elapsed_ms / 1000) > 2:
                report["issues"].append("acceleration_duration_mismatch")
            if first_time is not None and (anchor is None or abs(first_time / 1e9 - anchor) > 2):
                report["issues"].append("acceleration_bbi_start_anchor_mismatch")
            arrivals = [row["callback_elapsed_ms"] for row in rows]
            if (any(a > b for a, b in zip(arrivals, arrivals[1:]))
                    or (elapsed_ms is not None and any(v > elapsed_ms for v in arrivals))):
                report["issues"].append("inconsistent_bbi_callback_timing")
    except Exception:
        # Upstream parser errors can include raw field contents and local paths.
        report["issues"].append("corrupt_fit" if not report["fit_valid"] else "raw_payload_invalid_or_unsupported")
    report["issues"] = list(dict.fromkeys(report["issues"]))
    report["research_valid"] = report["fit_valid"] and not report["issues"]
    return report


def print_validation(report):
    LOG.info("FIT valid=%s; research valid=%s; acceleration samples=%d; BBI=%d; duration seconds=%s",
             report["fit_valid"], report["research_valid"], report["acceleration_samples"],
             report["bbi_count"], report["duration_seconds"])
    if "bbi" in report:
        LOG.info("Missing BBI sequences=%s; invalid BBI=%s", report["bbi"]["missing_sequence_ranges"], report["bbi"]["invalid_recovered_intervals"])
    if report["issues"]:
        LOG.error("Research validation failed: %s", ", ".join(report["issues"]))
        if any(i.startswith("missing_raw") or i in ("missing_bbi_developer_histories", "missing_or_invalid_raw_xyz_or_sample_timing") for i in report["issues"]):
            LOG.error("Required raw signals/developer fields are absent. If present in the USB original, Garmin Connect stripped them: this route is unusable for the intended raw-data workflow.")


def read_manifest(directory):
    path = directory / MANIFEST
    if not path.exists():
        return {"version": 1, "activities": {}}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["version"] != 1 or not isinstance(result["activities"], dict):
            raise ValueError()
        return result
    except (ValueError, KeyError, TypeError):
        raise SyncError("Sync manifest is invalid; restore its local backup or move it aside and resync.") from None


def already_downloaded(directory, entry):
    if not isinstance(entry, dict) or not entry.get("validation", {}).get("research_valid"):
        return False
    relative = Path(entry.get("file", ""))
    if relative.is_absolute() or ".." in relative.parts:
        return False
    path = directory / relative
    return path.is_file() and hash_file(path) == entry.get("sha256")


def activity_id(value):
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise SyncError("Garmin activity ID must be a positive integer.")
    return text


def matches_activity(activity, names, device_id=None):
    normalized = " ".join(str(activity.get("activityName", "")).split()).casefold()
    return (normalized in {" ".join(name.split()).casefold() for name in names}
            and (device_id is None or str(activity.get("deviceId")) == str(device_id)))


def sync(api, directory, *, start, end, names, device_id=None, dry_run=False, redownload=None, persist=None):
    directory = Path(directory)
    manifest = read_manifest(directory)
    call = lambda operation: api_call(operation, persist=persist)
    if redownload:
        identifier = activity_id(redownload)
        activities = [call(lambda: api.get_activity(identifier))]
        if not isinstance(activities[0], dict):
            raise SyncError("Garmin returned invalid activity metadata.")
        activities[0] = dict(activities[0], activityId=identifier)
    else:
        activities = call(lambda: api.get_activities_by_date(start.isoformat(), end.isoformat()))
    if not isinstance(activities, list) or any(not isinstance(item, dict) for item in activities):
        raise SyncError("Garmin returned an unexpected activity list.")
    failures, matched, seen = 0, 0, set()
    for activity in activities:
        if not redownload and not matches_activity(activity, names, device_id):
            continue
        identifier = activity_id(activity.get("activityId"))
        if identifier in seen:
            continue
        seen.add(identifier)
        matched += 1
        previous = manifest["activities"].get(identifier)
        if not redownload and already_downloaded(directory, previous):
            LOG.info("Activity %s: skipped; validated local bytes already present", identifier)
            continue
        if dry_run:
            LOG.info("Activity %s: would download ORIGINAL (raw preservation not yet verified)", identifier)
            continue
        try:
            payload = call(lambda: api.download_activity(identifier, dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL))
            original = extract_original(payload)
            digest = hashlib.sha256(original).hexdigest()
            # Validate an unpublished file. A .part can never be mistaken for a FIT.
            staging = directory / ".staging"
            staging.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=staging, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                atomic_write(temporary, original)
                report = validate_fit(temporary)
                basename = f"garmin_{identifier}_{digest[:16]}"
                relative = Path(basename + ".fit") if report["research_valid"] else Path("quarantine") / (basename + (".fit" if report["fit_valid"] else ".download"))
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            manifest["activities"][identifier] = {"file": relative.as_posix(), "sha256": digest,
                "downloaded": True, "validation": report}
            atomic_write(directory / MANIFEST, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
            LOG.info("Activity %s: original download complete; %s", identifier,
                     "raw checks passed" if report["research_valid"] else "QUARANTINED; not a valid research recording")
            print_validation(report)
            failures += not report["research_valid"]
        except (ReauthenticationRequired, RateLimited):
            raise  # Stop the batch rather than hammering the remaining activities.
        except (SyncError, GarminConnectNotFoundError):
            failures += 1
            LOG.error("Activity %s: original FIT unavailable/incomplete or download failed; will retry next sync", identifier)
    if not matched:
        LOG.warning("No matching recordings found. This does not establish successful FIT sync; check the phone sync and exact activity name.")
    LOG.info("Sync finished: matched=%d; failures=%d", matched, failures)
    return 2 if failures else 0


def compare(cloud, usb):
    cloud_report, usb_report = validate_fit(cloud), validate_fit(usb)
    result = {"byte_identical": hash_file(cloud) == hash_file(usb),
              "cloud": cloud_report, "usb": usb_report}
    for key in ("acceleration_fingerprint", "developer_fingerprint"):
        result[key + "_equal"] = (key in cloud_report and key in usb_report
                                    and cloud_report[key] == usb_report[key])
    result["raw_equal"] = all(result[key + "_equal"] for key in ("acceleration_fingerprint", "developer_fingerprint"))
    print(json.dumps(result, indent=2))
    if not result["raw_equal"]:
        LOG.error("Cloud/USB raw data differ or cannot be verified. Do not use the cloud route for research until resolved.")
    return 0 if result["raw_equal"] and cloud_report["research_valid"] and usb_report["research_valid"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    auth = commands.add_parser("auth", help="Interactive first login/MFA; encrypt reusable session")
    auth.add_argument("--session-file", type=Path, default=DEFAULT_SESSION)
    synchronize = commands.add_parser("sync", help="Sync original FITs; never prompts for credentials")
    synchronize.add_argument("--session-file", type=Path, default=DEFAULT_SESSION)
    synchronize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    synchronize.add_argument("--days", type=int, default=7, help="Recent local-calendar days, including today (default 7)")
    synchronize.add_argument("--start-date", type=date.fromisoformat)
    synchronize.add_argument("--end-date", type=date.fromisoformat)
    synchronize.add_argument("--activity-name", action="append", help="Exact name, case/whitespace insensitive; repeatable")
    synchronize.add_argument("--device-id", help="Optional deviceId restriction from Garmin metadata")
    synchronize.add_argument("--dry-run", "--list-only", action="store_true")
    synchronize.add_argument("--redownload", metavar="ACTIVITY_ID", help="Download this ID even if renamed/previously downloaded; ignores date/name filters")
    validate = commands.add_parser("validate", help="Audit a local FIT without Garmin access")
    validate.add_argument("fit", type=Path)
    comparison = commands.add_parser("compare", help="Compare cloud FIT raw messages/developer fields with USB original")
    comparison.add_argument("--cloud", required=True, type=Path)
    comparison.add_argument("--usb", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Suppress all upstream response/body/profile diagnostics, even on errors.
    for name in ("garminconnect", "urllib3", "requests", "curl_cffi"):
        logging.getLogger(name).disabled = True
        logging.getLogger(name).addHandler(logging.NullHandler())
        logging.getLogger(name).propagate = False
    try:
        if args.command == "validate":
            report = validate_fit(args.fit)
            print(json.dumps(report, indent=2))
            print_validation(report)
            return 0 if report["research_valid"] else 2
        if args.command == "compare":
            return compare(args.cloud, args.usb)
        store = SessionStore(args.session_file)
        if args.command == "auth":
            with output_lock(store.path.parent, ".garmin-session.lock"):
                authenticate(store, interactive=True)
            LOG.info("Authentication configured. Scheduled sync can now reuse the protected session.")
            return 0
        if args.days < 1:
            parser.error("--days must be positive")
        end = args.end_date or date.today()
        start = args.start_date or end - timedelta(days=args.days - 1)
        if start > end:
            parser.error("--start-date must be on/before --end-date")
        if args.redownload:
            activity_id(args.redownload)
        def run():
            api = authenticate(store)
            return sync(api, args.output_dir.expanduser().absolute(), start=start, end=end,
                        names=args.activity_name or ["Overnight Accel BBI"], device_id=args.device_id,
                        dry_run=args.dry_run, redownload=args.redownload, persist=lambda: store.save(api))
        # Serialize refresh-token rotation across output directories and list-only
        # runs, as well as the files/manifest in each output directory.
        with output_lock(store.path.parent, ".garmin-session.lock"):
            if args.dry_run:
                return run()
            with output_lock(args.output_dir.expanduser().absolute()):
                return run()
    except ReauthenticationRequired as error:
        LOG.error("%s", error)
        return 3
    except SyncError as error:
        LOG.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOG.error("Sync interrupted; unpublished partial files are not valid recordings. Rerun sync.")
        return 130
    except Exception:
        LOG.error("Operation failed; check local file permissions, free disk space and connectivity. No upstream details logged for privacy.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
