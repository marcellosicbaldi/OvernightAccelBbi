# Garmin Connect ORIGINAL FIT sync (review and real-device testing)

This optional Python script retrieves Hammock activities **after** the watch has
saved them and Garmin Connect on the phone has uploaded them. It does not initiate
Bluetooth sync, change the recorder, change the FIT schema, edit the analysis
pipeline, upload data, or delete anything from the watch/Connect.

**Garmin account/device interoperability is unverified.** Automated tests use
mocked API responses and generated synthetic FITs. Before replacing daily USB
copying, complete the five-minute test and compare its cloud FIT with the **same
activity's USB original**. A Garmin Connect graph or an activity listing does not
prove that SensorLogger samples or developer fields reached the cloud.

## 1. Windows setup

Use Python **3.12+**. In PowerShell, work in the checkout containing the
`feature/garmin-connect-sync` branch. Keep unrelated edits in their original
checkout. If you prefer an independent review copy:

```powershell
git clone --branch feature/garmin-connect-sync https://github.com/marcellosicbaldi/OvernightAccelBbi.git Hammock-sync
cd Hammock-sync
```

Create an environment and install the small, separate sync requirements (the
notebook dependency list stays unchanged):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts\requirements-sync.txt
```

If `py` is unavailable, use the full path of your installed Python 3.12 executable
in the first command. Activation and PowerShell execution-policy changes are not
needed when invoking `.venv\Scripts\python.exe` directly.

Default recordings go to `offline_processing\data\garmin_connect\`, independent
of the current terminal directory. Override with `--output-dir "D:\Hammock\FIT"`.

## 2. First login (interactive, once)

```powershell
.\.venv\Scripts\python.exe scripts\sync_garmin.py auth
```

Enter your Garmin account email, password and, if requested by Garmin, MFA code.
All three prompts hide typed input. Credentials are never accepted on the command
line, hardcoded, logged, or saved. Do not send them to a chat or commit them.

On Windows the reusable DI session is encrypted with **current-user Windows
DPAPI**, saved atomically to `%LOCALAPPDATA%\Hammock\garmin-session.bin`. Plaintext
tokens never go to disk. Use the **same Windows user on the same PC** for scheduled
sync. The library refreshes DI tokens; the script persists refreshed state after
each API operation, including failed operations that may already have refreshed.
Session locks prevent concurrent refreshes. `--session-file "C:\private\garmin-session.bin"`
is available on both `auth` and `sync`; use the same path in both commands.

Sync never prompts or falls back to a password. When the session expires, is
revoked, cannot be decrypted, or requires MFA again, it exits with code **3** and
instructs you to rerun `auth` in a terminal. The pinned library only serializes DI
tokens; a JWT-only login cannot be saved for unattended use and fails explicitly.
On POSIX systems, sessions use an owner-only directory (0700) and file (0600),
rather than DPAPI. Do not share or sync session files.

## 3. Five-minute physical recording test

1. Wear your Instinct 3 AMOLED normally and open your existing Hammock recorder.
   Do not install another recorder build for this sync test.
2. Record about five minutes, with a short wrist movement between quiet periods.
   Check that native FIT acceleration and received BBI counters increase. An
   ordinary HR trace is insufficient evidence of BBI acquisition.
3. Stop/save and exit the watch app. A partial-save warning needs investigation.
4. Open Garmin Connect on the phone, sync the watch, and wait until the activity
   appears. Keep its name **Overnight Accel BBI** for automatic identification.
5. List matching activities without downloading or writing a manifest:

   ```powershell
   .\.venv\Scripts\python.exe scripts\sync_garmin.py sync --days 7 --dry-run
   ```

6. Download and audit originals:

   ```powershell
   .\.venv\Scripts\python.exe scripts\sync_garmin.py sync --days 7
   ```

7. Read the report: acceleration sample count, BBI count, missing BBI sequence
   ranges, and acceleration duration. `original download complete` and
   `research valid=True` describe different checks. Successful local raw checks
   still cannot prove equivalence to an unseen USB original.
8. For this initial verification, connect USB once and **copy** that same FIT from
   `GARMIN\ACTIVITY` to `offline_processing\data\usb-test\`. Preserve the watch file.
   Use timestamps/recording duration locally to select the same recording; the
   comparison includes acceleration timestamps and BBI metadata, not just counts.

   ```powershell
   .\.venv\Scripts\python.exe scripts\sync_garmin.py compare --cloud "offline_processing\data\garmin_connect\garmin_ACTIVITYID_HASH.fit" --usb "offline_processing\data\usb-test\ORIGINAL.fit"
   ```

   Replace the example filenames with the actual paths. `byte_identical: true`
   establishes complete file equality. If bytes differ, `raw_equal: true` requires
   exact equality of ordered native acceleration messages and all developer field
   values/descriptors, including callback timing, history slots and diagnostics.
   Both files must also pass structural raw validation for exit code 0. This
   comparison deliberately rejects differences in message layout/metadata rather
   than guessing that reordered or reconstructed signals are equivalent.

If raw signals/developer histories exist in USB but are absent or different in
the cloud original, **this cloud route is unusable for the intended raw-data
workflow until resolved**. Keep using USB; do not substitute GPX/TCX/CSV, Garmin
graphs, activity summaries, or reconstructed signals. The script reports this
condition, quarantines the file, and returns a failure code. If the original
endpoint is unavailable or only derived exports exist, that is also a blocker.

## Validation and storage rules

- FIT header/declared size and CRC must parse fully. Empty, truncated, corrupted,
  concatenated and trailing-garbage payloads cannot pass. ZIP downloads must
  contain exactly one FIT. Archive paths are never extracted, and a 512 MiB
  payload/FIT safety limit rejects oversized or encrypted inputs. This library
  returns the entire HTTP payload in memory; budget RAM for large overnight FITs.
- The script reuses `fit_recording.expand_acceleration_batch` and
  `fit_bbi.decode_snapshots`; it does not implement another FIT decoder. Native
  `accelerometer_data` must retain XYZ arrays, timestamps and sample offsets with
  matching lengths, finite values and strictly increasing sample times. Here
  "raw" means the recorder's original, unfiltered **calibrated XYZ sample
  channels**, not a claim that ADC-count channels exist. Scale/units stay exactly
  as written; no mg/g conversion or fixed sample rate is assumed.
- Numbered BBI histories, callback timestamps and one final SESSION snapshot are
  required. Gaps, invalid intervals, conflicting snapshots, mixed schemas and
  missing summaries fail research validation. BBI counts describe intervals the
  app received, not all physiological beats; callback arrival times are not exact
  beat times. Main's schema 2/3 decoder is reused. A sync-only schema 4 slot-name
  adapter feeds its packed arrays to that same sequence decoder; FIT bytes and
  the offline pipeline are unchanged. Unknown/legacy schemas fail explicitly.
- This validator is deliberately strict about inconsistent schema 3 snapshots;
  it may quarantine a recording that a newer analysis-branch decoder can partly
  recover. Never infer cloud stripping from that fact alone; compare the USB FIT.
- Native sample count in final logger metadata must equal the decoded acceleration
  count. The sample span must agree with final logger elapsed time within two
  seconds, and the acceleration start must agree with the BBI UTC anchor within
  two seconds. Callback arrival offsets must be ordered and within final elapsed
  time. These checks detect many incomplete recordings, but cannot prove that no
  internal samples were altered without a USB comparison. A count mismatch can
  also originate in the recorder; compare both originals before attributing blame.
- Passing structural checks is not physiological/clinical validation. The script
  never silently fills gaps or claims that unobserved beats were recorded.

Validated files are named `garmin_<stable activity ID>_<SHA256 prefix>.fit`.
`quarantine\` contains unchanged but rejected files: `.fit` when FIT structure
passes, `.download` for corrupt/non-FIT payloads. Files pass through unpublished
`.part` staging, are flushed/fsynced, validated, then atomically renamed on the
same volume. A hard process kill can leave `.part` files, which are never used
as recordings; rerunning sync is safe. OS-held locks release after a crash.

`.garmin-sync-manifest.json` records only IDs, relative filenames, hashes and
validation reports. A prior recording is skipped only when its saved validation
passed, the file still exists, and its SHA-256 matches. Invalid/quarantined or
missing/changed files are retried on the next sync. If a redownload differs,
both versions remain on disk; the manifest points to the latest result. A crash
after publishing but before manifest commit can cause a harmless repeat download.
Damaged manifests fail clearly; move the damaged file aside and resync to rebuild
it. Do not hand-edit a validation status to force acceptance.

Recordings, ZIPs, manifests, partial downloads and conventional token files are
Git-ignored. Custom output/session paths are your private local data; do not
force-add or share them. Console reports contain IDs and aggregate data-quality
counts/duration, never account names, credentials, token strings, activity titles,
locations, raw samples, or upstream exception bodies. Treat reports/logs as private.

## Other commands

```powershell
# Date range: inclusive activity-start calendar dates as used by Garmin Connect.
# An overnight activity is selected by its START date, not the morning it ends.
.\.venv\Scripts\python.exe scripts\sync_garmin.py sync --start-date 2026-10-20 --end-date 2026-11-01

# Custom destination, exact alternate name; --activity-name can be repeated.
.\.venv\Scripts\python.exe scripts\sync_garmin.py sync --output-dir "D:\Hammock\FIT" --activity-name "Overnight Accel BBI"

# Optional deviceId restriction; a missing/nonmatching deviceId is excluded.
.\.venv\Scripts\python.exe scripts\sync_garmin.py sync --device-id 123456789

# Explicit ID overrides name/date filters, useful for renamed activities.
.\.venv\Scripts\python.exe scripts\sync_garmin.py sync --redownload 123456789

# Local validation, no Garmin access or credentials needed.
.\.venv\Scripts\python.exe scripts\sync_garmin.py validate "D:\Hammock\FIT\recording.fit"
```

Activity selection uses the exact normalized name (case/whitespace insensitive).
The list API does not supply a dependable Connect IQ application identifier, so
the script does not guess from sport type or descriptions. Device restriction is
optional; raw structure/schema validation provides the second check. Explicit ID
downloads still undergo the same validation. Names/date ranges and device IDs
are not persisted in the manifest.

Exit codes: **0** completed (possibly no new/matching activities; inspect the
warning), **1** operational/network/rate-limit/storage error, **2** rejected raw
recording/comparison, **3** interactive reauthentication required, **130** interrupted.
`--dry-run` / `--list-only` never downloads or changes the output directory or
manifest; it still authenticates, locks and may refresh the protected session.

Network/5xx failures get at most three application attempts with exponential
backoff. Authentication/access failures do not retry. Rate limits respect
`Retry-After` (seconds or HTTP date); absent headers use 60 seconds. A longer
cooldown aborts the run instead of retrying early. An exhausted rate limit stops
the entire batch. The library's extra transport retries are disabled; its standard
authentication and token-refresh flows remain in use. Do not repeatedly rerun
login/sync against an account restriction.

## 4. Daily Windows Task Scheduler sync

Configure this **after** the cloud/USB comparison succeeds. Nothing installs a
scheduled task automatically.

1. Open **Task Scheduler → Create Task**, name it **Hammock Garmin sync**.
2. Select the same Windows account used for `auth`. The simplest initial setting
   is **Run only when user is logged on**. To run while logged out, choose **Run
   whether user is logged on or not** with that user's normal Windows credentials;
   do not select **Do not store password** (network access is needed). No elevated
   privileges or SYSTEM account are needed. DPAPI must access that user's profile.
3. Add a **Daily** trigger, for example **09:00**, after your usual phone sync.
4. Add **Start a program**:
   - Program: full path to this checkout's `.venv\Scripts\python.exe`.
   - Arguments: `"C:\full\path\Hammock-sync\scripts\sync_garmin.py" sync --days 7`
   - Start in: `C:\full\path\Hammock-sync` (no quotes in this field).
   Use your actual paths. Add the same `--output-dir`/`--session-file` overrides
   you tested, if any; do not put passwords or tokens in the task.
5. Enable **Run task as soon as possible after a scheduled start is missed** and
   **Do not start a new instance**. Allow enough runtime for a large overnight
   original. Optionally enable waking the PC and requiring a network connection.
6. Right-click the task → **Run**. Check **Last Run Result** and new local FITs/
   manifest. `0x0` with no new file can mean no matching cloud activity, not a
   successful new download. `0x2` means inspect quarantine/validation; `0x3` means
   run `auth` interactively and rerun the task. Run the sync command manually to
   view diagnostics when Task Scheduler hides its console.

This is a daily poll, not an event triggered by the phone. A late phone upload is
retrieved on the next run while its **start date** remains within the seven-day
lookback. If the PC/phone was offline longer, use a wider date range manually.
The PC needs to be awake and online at execution time. Preserve private local
backups; this feature does not establish a cloud retention guarantee.

## API investigation and automated verification

Checked on **2026-11-01**, using the installed and pinned **garminconnect 0.3.20**:

- [Version requirements](https://github.com/cyberjunky/python-garminconnect/blob/0.3.20/pyproject.toml):
  Python 3.12+, native DI authentication, no old `garth.save()` integration.
- [Authentication implementation](https://github.com/cyberjunky/python-garminconnect/blob/0.3.20/garminconnect/api/base.py):
  `Garmin(email=..., password=..., prompt_mfa=...)`, `login()`, saved-token JSON
  passed to `login(tokenstore=...)`, DI refresh and credential clearing.
- [Token serialization](https://github.com/cyberjunky/python-garminconnect/blob/0.3.20/garminconnect/client/tokens.py):
  `client.dumps()`/`loads()`. This script wraps serialized tokens with DPAPI;
  it never writes a plaintext library tokenstore.
- [Activity methods](https://github.com/cyberjunky/python-garminconnect/blob/0.3.20/garminconnect/api/activities.py):
  `get_activities_by_date(startdate, enddate)` paginates; `get_activity(id)` reads
  metadata; `download_activity(id, dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL)`
  returns original-download bytes, usually ZIP, unlike its default TCX format.
- [Garmin export documentation](https://support.garmin.com/en-US/?faq=W1TvTPW8JZ6LfJSfK512Q8)
  and [FIT developer-data format](https://developer.garmin.com/fit/cookbook/developer-data/)
  describe original FIT exports/developer fields. Neither is evidence that this
  particular SensorLogger recording survives your account's phone/cloud route.

The package is an unofficial client of Garmin Connect. Endpoints, restrictions
and authentication can change; update the pin only after reviewing API changes
and rerunning synthetic tests plus a physical comparison. No bypass is provided.

Run offline tests without Garmin credentials or personal recordings:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The existing analysis suite remains independently runnable with
`offline_processing/requirements.txt`. CI runs the sync tests on Windows and Linux,
including real Windows DPAPI encryption/decryption of **synthetic** session data.
This is not a Garmin login test. Physical phone sync, account/MFA behavior, actual
overnight file size/runtime and USB/cloud equality remain the reviewer's test.
