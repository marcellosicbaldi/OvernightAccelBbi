"""Offline sync tests with mocked Garmin responses and generated FIT bytes."""

from contextlib import redirect_stdout
from datetime import date
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from scripts import sync_garmin as sync
from synthetic_fit import synthetic_fit


def archive(payload, name="recording.FIT", extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(name, payload)
        if extra:
            zipped.writestr(*extra)
    return stream.getvalue()


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "downloads"
        self.api = MagicMock()
        self.api.get_activities_by_date.return_value = [
            {"activityId": 123, "activityName": "Overnight Accel BBI", "deviceId": 99}]
        self.api.download_activity.return_value = archive(synthetic_fit())

    def fit_path(self, **kwargs):
        path = self.root / "synthetic.fit"
        path.write_bytes(synthetic_fit(**kwargs))
        return path

    def run_sync(self, **kwargs):
        return sync.sync(self.api, self.output, start=date(2026, 1, 1), end=date(2026, 1, 7),
                         names=["Overnight Accel BBI"], **kwargs)

    def test_name_filter_is_exact_case_whitespace_insensitive(self):
        for name, expected in [("overnight   accel bbi", True), ("Running", False),
                               ("Overnight Accel BBI extra", False), (None, False)]:
            with self.subTest(name=name):
                self.assertEqual(sync.matches_activity({"activityName": name}, ["Overnight Accel BBI"]), expected)
        self.assertFalse(sync.matches_activity({"activityName": "Overnight Accel BBI", "deviceId": 8}, ["Overnight Accel BBI"], 9))

    def test_filter_and_date_range_use_current_library(self):
        self.api.get_activities_by_date.return_value += [{"activityId": 456, "activityName": "Running"}]
        self.assertEqual(self.run_sync(), 0)
        self.api.get_activities_by_date.assert_called_once_with("2026-01-01", "2026-01-07")
        self.api.download_activity.assert_called_once_with("123", dl_fmt=sync.Garmin.ActivityDownloadFormat.ORIGINAL)
        self.assertEqual(next(self.output.glob("*.fit")).read_bytes(), synthetic_fit())

    def test_valid_schema3_and_schema4_real_binary_fit(self):
        for schema in [3, 4]:
            with self.subTest(schema=schema):
                report = sync.validate_fit(self.fit_path(schema=schema))
                self.assertTrue(report["research_valid"], report)
                self.assertEqual(report["bbi"]["schema"], schema)
                self.assertEqual(report["acceleration_samples"], 2)
                self.assertEqual(report["bbi_count"], 2)
                self.assertEqual(report["duration_seconds"], 1)

    def test_12_byte_header_supported(self):
        self.assertTrue(sync.validate_fit(self.fit_path(header_size=12))["research_valid"])

    def test_absent_raw_signals_never_valid(self):
        for kwargs, issue in [({"acceleration": False}, "missing_raw_accelerometry"),
                              ({"bbi": False}, "missing_bbi_developer_histories"),
                              ({"missing_timing": True}, "missing_or_invalid_raw_xyz_or_sample_timing"),
                              ({"missing_axis": True}, "missing_or_invalid_raw_xyz_or_sample_timing")]:
            with self.subTest(kwargs=kwargs):
                report = sync.validate_fit(self.fit_path(**kwargs))
                self.assertTrue(report["fit_valid"], report)
                self.assertFalse(report["research_valid"])
                self.assertIn(issue, report["issues"])

    def test_missing_final_sequence_gap_zero_and_invalid_bbi(self):
        for kwargs, issue in [({"final": False}, "missing_or_ambiguous_final_summary"),
                              ({"total": 26, "record": False}, "missing_bbi_sequences"),
                              ({"total": 0}, "no_bbi_intervals"),
                              ({"invalid_bbi": True}, "invalid_bbi_intervals"),
                              ({"native_count": 50}, "native_acceleration_count_mismatch_or_missing"),
                              ({"elapsed_ms": 10000}, "acceleration_duration_mismatch"),
                              ({"reversed_times": True}, "nonmonotonic_or_insufficient_acceleration"),
                              ({"callback_after_end": True}, "raw_payload_invalid_or_unsupported")]:
            with self.subTest(kwargs=kwargs):
                report = sync.validate_fit(self.fit_path(**kwargs))
                self.assertFalse(report["research_valid"], report)
                self.assertIn(issue, report["issues"])
        report = sync.validate_fit(self.fit_path(total=26, record=False))
        self.assertEqual(report["bbi"]["missing_sequence_ranges"], [(1, 2)])

    def test_nonfinite_xyz_rejected(self):
        self.assertFalse(sync.validate_fit(self.fit_path(nonfinite=True))["research_valid"])

    def test_corruption_truncation_and_trailing_garbage_rejected(self):
        good = synthetic_fit()
        for content in [b"", b"<html>not FIT</html>", good[:-1], good + b"garbage", good[:-3] + b"\xff" + good[-2:]]:
            with self.subTest(size=len(content)):
                path = self.root / "bad.fit"
                path.write_bytes(content)
                report = sync.validate_fit(path)
                self.assertFalse(report["fit_valid"], report)
                self.assertFalse(report["research_valid"])

    def test_zip_direct_and_traversal_filename_never_extracted_to_path(self):
        good = synthetic_fit()
        for payload in [good, archive(good), archive(good, "../../escape.FIT", ("readme.txt", b"ignored"))]:
            self.assertEqual(sync.extract_original(payload), good)
        self.assertFalse((self.root.parent / "escape.FIT").exists())

    def test_zip_missing_multiple_corrupt_oversize_encrypted(self):
        for content in [archive(b"GPX", "export.gpx"), archive(b"FIT", extra=("second.fit", b"FIT")), b"PKbad"]:
            with self.assertRaises(sync.SyncError):
                sync.extract_original(content)
        with patch.object(sync, "MAX_BYTES", 10), self.assertRaises(sync.SyncError):
            sync.extract_original(synthetic_fit())
        zipped = archive(synthetic_fit())
        with patch.object(zipfile.ZipFile, "infolist") as entries:
            item = zipfile.ZipInfo("a.fit")
            item.flag_bits = 1
            entries.return_value = [item]
            with self.assertRaises(sync.SyncError):
                sync.extract_original(zipped)

    def test_duplicate_ids_skipped_and_modified_local_file_redownloaded(self):
        self.api.get_activities_by_date.return_value *= 2
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(self.api.download_activity.call_count, 1)
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(self.api.download_activity.call_count, 1)
        target = next(self.output.glob("*.fit"))
        target.write_bytes(b"interrupted or externally changed")
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(self.api.download_activity.call_count, 2)
        self.assertEqual(target.read_bytes(), synthetic_fit())
        target.unlink()
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(self.api.download_activity.call_count, 3)

    def test_invalid_research_recording_quarantined_and_retried(self):
        self.api.download_activity.return_value = archive(synthetic_fit(bbi=False))
        self.assertEqual(self.run_sync(), 2)
        self.assertFalse(list(self.output.glob("*.fit")))
        self.assertEqual(len(list((self.output / "quarantine").glob("*.fit"))), 1)
        entry = sync.read_manifest(self.output)["activities"]["123"]
        self.assertTrue(entry["downloaded"])
        self.assertFalse(entry["validation"]["research_valid"])
        self.assertEqual(self.run_sync(), 2)
        self.assertEqual(self.api.download_activity.call_count, 2)

    def test_corrupt_payload_kept_as_download_not_fit(self):
        self.api.download_activity.return_value = b"not FIT"
        self.assertEqual(self.run_sync(), 2)
        self.assertFalse(list(self.output.rglob("*.fit")))
        self.assertTrue(list(self.output.rglob("*.download")))

    def test_redownload_renamed_activity_preserves_prior_original(self):
        self.run_sync()
        self.api.get_activity.return_value = {"activityName": "Renamed"}
        self.api.download_activity.return_value = synthetic_fit(x=2)
        self.assertEqual(self.run_sync(redownload="123"), 0)
        self.api.get_activity.assert_called_once_with("123")
        self.assertEqual(len(list(self.output.glob("*.fit"))), 2)

    def test_dry_run_does_not_download_or_create_output(self):
        self.assertEqual(self.run_sync(dry_run=True), 0)
        self.api.download_activity.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_no_match_warns_that_sync_not_proven(self):
        self.api.get_activities_by_date.return_value = []
        with self.assertLogs(sync.LOG, level="WARNING") as logs:
            self.assertEqual(self.run_sync(), 0)
        self.assertIn("does not establish successful FIT sync", " ".join(logs.output))

    def test_interrupted_publish_does_not_create_fit_or_manifest(self):
        real_replace = os.replace
        def interrupt(src, dst):
            if str(dst).endswith(".fit"):
                raise KeyboardInterrupt()
            return real_replace(src, dst)
        with patch.object(sync.os, "replace", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_sync()
        self.assertFalse(list(self.output.rglob("*.fit")))
        self.assertFalse(list(self.output.rglob("*.part")))
        self.assertFalse((self.output / sync.MANIFEST).exists())
        self.assertEqual(self.run_sync(), 0)

    def test_interrupted_atomic_write_preserves_existing_target(self):
        target = self.root / "existing.fit"
        target.write_bytes(b"old bytes")
        with patch.object(sync.os, "fsync", side_effect=OSError("disk full")), self.assertRaises(OSError):
            sync.atomic_write(target, b"new bytes")
        self.assertEqual(target.read_bytes(), b"old bytes")
        self.assertFalse(list(self.root.glob("*.part")))

    def test_manifest_corruption_fails_without_downloading(self):
        self.output.mkdir()
        (self.output / sync.MANIFEST).write_text("{bad", encoding="utf-8")
        with self.assertRaises(sync.SyncError):
            self.run_sync()
        self.api.download_activity.assert_not_called()

    def test_manifest_cannot_skip_via_path_traversal(self):
        self.assertFalse(sync.already_downloaded(self.output, {"file": "../other.fit", "validation": {"research_valid": True}}))

    def test_batch_stops_on_auth_failure_or_rate_limit(self):
        self.api.get_activities_by_date.return_value.append(
            {"activityId": 456, "activityName": "Overnight Accel BBI"})
        for error, expected in [(sync.GarminConnectAuthenticationError("secret"), sync.ReauthenticationRequired),
                                (sync.GarminConnectTooManyRequestsError("secret"), sync.RateLimited)]:
            error.response = MagicMock(status_code=429, headers={"Retry-After": "120"})
            self.api.download_activity.reset_mock()
            self.api.download_activity.side_effect = error
            with self.assertRaises(expected):
                self.run_sync()
            self.assertEqual(self.api.download_activity.call_count, 1)

    def test_missing_original_is_retried_on_next_sync(self):
        self.api.download_activity.side_effect = sync.GarminConnectNotFoundError("secret personal response")
        self.assertEqual(self.run_sync(), 2)
        self.assertFalse((self.output / sync.MANIFEST).exists())
        self.api.download_activity.side_effect = None
        self.assertEqual(self.run_sync(), 0)

    def test_failed_manifest_commit_leaves_only_complete_recoverable_fit(self):
        original_atomic = sync.atomic_write
        def fail_manifest(path, data):
            if path.name == sync.MANIFEST:
                raise OSError("disk full")
            return original_atomic(path, data)
        with patch.object(sync, "atomic_write", side_effect=fail_manifest), self.assertRaises(OSError):
            self.run_sync()
        self.assertTrue(sync.validate_fit(next(self.output.glob("*.fit")))["research_valid"])
        self.assertEqual(self.run_sync(), 0)
        self.assertEqual(len(list(self.output.glob("*.fit"))), 1)

    def test_output_lock_excludes_concurrent_sync_and_releases(self):
        with sync.output_lock(self.output):
            with self.assertRaises(sync.SyncError):
                with sync.output_lock(self.output):
                    self.fail("lock admitted a second writer")
        with sync.output_lock(self.output):
            pass

    def test_cloud_usb_comparison_detects_raw_and_developer_changes(self):
        cloud = self.fit_path()
        usb = self.root / "usb.fit"
        usb.write_bytes(cloud.read_bytes())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sync.compare(cloud, usb), 0)
            for kwargs in [{"x": 3}, {"diagnostic": 43}, {"bbi": False}]:
                usb.write_bytes(synthetic_fit(**kwargs))
                self.assertEqual(sync.compare(cloud, usb), 2)


class AuthenticationTests(unittest.TestCase):
    def test_connection_retries_bounded_without_exception_contents(self):
        operation, sleep = MagicMock(), MagicMock()
        operation.side_effect = sync.GarminConnectConnectionError("private@example.com SECRET-TOKEN")
        with self.assertLogs(sync.LOG, level="WARNING") as logs:
            with self.assertRaises(sync.SyncError) as caught:
                sync.api_call(operation, sleep=sleep)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("SECRET", str(caught.exception) + " ".join(logs.output))

    def test_auth_failure_never_retried_and_batch_stops(self):
        operation = MagicMock(side_effect=sync.GarminConnectAuthenticationError("private token"))
        with self.assertRaises(sync.ReauthenticationRequired):
            sync.api_call(operation)
        self.assertEqual(operation.call_count, 1)

    def test_rate_limit_retry_after_and_long_cooldown(self):
        error = sync.GarminConnectTooManyRequestsError("private response")
        error.response = MagicMock(status_code=429, headers={"Retry-After": "12"})
        operation = MagicMock(side_effect=[error, "success"])
        sleep = MagicMock()
        self.assertEqual(sync.api_call(operation, sleep=sleep), "success")
        sleep.assert_called_once_with(12)
        error.response.headers = {"Retry-After": "3600"}
        operation = MagicMock(side_effect=error)
        with self.assertRaises(sync.RateLimited):
            sync.api_call(operation, sleep=sleep)
        self.assertEqual(operation.call_count, 1)

    def test_http_access_denied_and_missing_original_are_not_retried(self):
        for status, expected in [(403, sync.ReauthenticationRequired), (404, sync.SyncError)]:
            error = sync.GarminConnectConnectionError("HTTP " + str(status))
            operation = MagicMock(side_effect=error)
            with self.assertRaises(expected):
                sync.api_call(operation)
            self.assertEqual(operation.call_count, 1)

    def test_first_login_hidden_prompts_mfa_callback_and_session_reuse(self):
        api, store = MagicMock(), MagicMock()
        with patch.object(sync, "Garmin", return_value=api) as factory, patch.object(sync.sys.stdin, "isatty", return_value=True), patch.object(sync, "getpass", side_effect=["synthetic-email", "synthetic-password", "123456"]):
            self.assertIs(sync.authenticate(store, interactive=True), api)
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["prompt_mfa"](), "123456")
            api.login.assert_called_once_with("{}")
            self.assertIsNone(api.password)
            store.save.assert_called_once_with(api)
        api.reset_mock()
        store.read.return_value = '{"di_token":"synthetic"}'
        with patch.object(sync, "Garmin", return_value=api):
            sync.authenticate(store)
        api.login.assert_called_once_with(store.read.return_value)

    def test_missing_session_and_noninteractive_auth_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(sync.ReauthenticationRequired):
                sync.SessionStore(Path(directory) / "absent.bin").read()
        with patch.object(sync.sys.stdin, "isatty", return_value=False), self.assertRaises(sync.ReauthenticationRequired):
            sync.authenticate(MagicMock(), interactive=True)

    def test_refresh_state_saved_even_on_api_failure(self):
        persist = MagicMock()
        with self.assertRaises(sync.ReauthenticationRequired):
            sync.api_call(MagicMock(side_effect=sync.GarminConnectAuthenticationError()), persist=persist)
        persist.assert_called_once()

    def test_secure_store_roundtrip_synthetic_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = sync.SessionStore(Path(directory) / "session.bin")
            api = MagicMock()
            payload = '{"di_token":"synthetic-token","di_refresh_token":"synthetic-refresh"}'
            api.client.dumps.return_value = payload
            store.save(api)
            self.assertEqual(store.read(), payload)
            if os.name == "nt":
                self.assertNotIn(b"synthetic-token", store.path.read_bytes())
                damaged = bytearray(store.path.read_bytes())
                damaged[-1] ^= 1
                store.path.write_bytes(damaged)
                with self.assertRaises(sync.ReauthenticationRequired):
                    store.read()
            else:
                self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)

    def test_cli_unattended_requires_auth_and_never_prompts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sync, "getpass") as prompt:
            self.assertEqual(sync.main(["sync", "--session-file", str(Path(directory) / "missing.bin"), "--output-dir", str(Path(directory) / "data")]), 3)
            prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
