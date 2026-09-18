import Toybox.Activity;
import Toybox.ActivityRecording;
import Toybox.FitContributor;
import Toybox.Lang;
import Toybox.Sensor;
import Toybox.SensorLogging;
import Toybox.System;
import Toybox.Time;
import Toybox.WatchUi;

class OvernightAccelBbiRecorder {
    const VERSION = "1.1.1";
    private const CALLBACK_HZ = 25;
    private var _session;
    private var _logger as SensorLogging.SensorLogger?;
    private var _recording as Boolean = false;
    private var _listenerRegistered as Boolean = false;
    private var _hrEnabled as Boolean = false;
    private var _startedMs as Number = 0;
    private var _startUnix as Number = 0;
    private var _elapsedMs as Number = 0;
    private var _status as String = "Ready";
    private var _lastError as String?;
    private var _bbi as BbiBuffer;
    private var _batchCount as Number = 0;
    private var _lastBbiMs as Number = -1;
    private var _emptyCallbacks as Number = 0;
    private var _accelSamples as Number = 0;
    private var _loggedSamples as Number = 0;
    private var _callbackCount as Number = 0;
    private var _fields as Array<Field>;
    private var _summaryFields as Array<Field>;

    function initialize() {
        _bbi = new BbiBuffer();
        _fields = [];
        _summaryFields = [];
    }

    function start() as Boolean {
        if (_session != null) {
            return fail("Save pending", "Save the existing session first");
        }
        resetStats();
        _status = "Starting";
        try {
            _logger = new SensorLogging.SensorLogger({:accelerometer => {:enabled => true}});
            _session = ActivityRecording.createSession({
                :name => "Overnight Accel BBI",
                :sport => Activity.SPORT_GENERIC,
                :sensorLogger => _logger
            });
            setupFitFields();
            _hrEnabled = Sensor.enableSensorType(Sensor.SENSOR_HEARTRATE);
            _startedMs = System.getTimer();
            _startUnix = Time.now().value();
            publishFields();
            if (!_session.start()) {
                cleanupUnstartedSession();
                return fail("Start failed", "Garmin could not start recording");
            }
            _recording = true;
            // Preserve the known-working cadence even when BBI is absent.
            // This rate does NOT configure SensorLogger's FIT sample rate.
            _listenerRegistered = true;
            Sensor.registerSensorDataListener(method(:onSensorData), {
                :period => 1,
                :accelerometer => {:enabled => true, :sampleRate => CALLBACK_HZ},
                :heartBeatIntervals => {:enabled => true}
            });
            _status = "Recording";
            WatchUi.requestUpdate();
            return true;
        } catch (e) {
            // Never discard a session that has started collecting data.
            if (_recording) {
                stopAndSave();
            } else {
                cleanupUnstartedSession();
            }
            return fail("Start error", e.getErrorMessage());
        }
    }

    function stopAndSave() as Boolean {
        if (_session == null) { return true; }
        try {
            var warning = null;
            // A diagnostic-field error must not prevent saving raw acceleration.
            try {
                updateLoggedSamples();
                publishFields();
            } catch (fieldError) {
                warning = fieldError.getErrorMessage();
            }
            if (_session.isRecording() && !_session.stop()) {
                return fail("Stop failed", "Recording retained; retry save");
            }
            if (_recording) {
                _elapsedMs = System.getTimer() - _startedMs;
            }
            _recording = false;
            releaseSensors();
            // SESSION fields preserve the tail without waiting for another RECORD.
            try {
                updateLoggedSamples();
                publishSummary();
            } catch (summaryError) {
                warning = summaryError.getErrorMessage();
            }
            if (!_session.save()) {
                return fail("Save failed", "Session retained; retry save");
            }
            _session = null;
            _logger = null;
            _fields = [];
            _summaryFields = [];
            _status = warning == null ? "Saved" : "Saved partial";
            _lastError = warning;
            if (warning != null) { System.println("Saved partial: " + warning); }
            WatchUi.requestUpdate();
            return true;
        } catch (e) {
            return fail("Save failed", e.getErrorMessage());
        }
    }

    function onSensorData(data as SensorData) as Void {
        if (!_recording) { return; }
        try {
            _callbackCount++;
            var accel = data.accelerometerData;
            if (accel != null) { _accelSamples += accel.x.size(); }
            var intervals = data.heartRateData == null ? null : data.heartRateData.heartBeatIntervals;
            _batchCount = intervals == null ? 0 : intervals.size();
            var elapsed = getElapsedMs();
            if (_batchCount == 0) {
                _emptyCallbacks++;
            } else {
                _lastBbiMs = elapsed;
            }
            _bbi.append(intervals, elapsed);
            // Keep compact scalar diagnostics; the history holds the intervals.
            _fields[0].setData(_batchCount < 255 ? _batchCount : 254);
            _fields[1].setData(_batchCount == 0 ? 0 : _bbi.last);
            updateLoggedSamples();
            publishFields();
        } catch (e) {
            var message = e.getErrorMessage();
            var saved = stopAndSave();
            fail(saved ? "Saved partial" : "Sensor error", message);
        }
    }

    function isRecording() as Boolean { return _recording; }
    function hasSession() as Boolean { return _session != null; }
    function getStatus() as String { return _status; }
    function getLastError() as String? { return _lastError; }
    function getElapsedMs() as Number { return _recording ? System.getTimer() - _startedMs : _elapsedMs; }
    function getElapsedSeconds() as Number { return getElapsedMs() / 1000; }
    function getAccelSamples() as Number { return _accelSamples; }
    function getLoggedSamples() as Number { return _loggedSamples; }
    function getTotalBbiCount() as Number { return _bbi.total; }
    function getCallbackCount() as Number { return _callbackCount; }
    function getFitFieldCount() as Number { return _fields.size() + _summaryFields.size(); }

    function getBbiStatus() as String {
        if (_startUnix == 0) { return "BBI --"; }
        if (_bbi.total == 0) {
            return _recording && getElapsedSeconds() < 30 ? "BBI waiting" : "No BBI received";
        }
        if (_recording && getElapsedMs() - _lastBbiMs > 10000) { return "BBI stale"; }
        return _bbi.last == 0 ? "BBI invalid" : "BBI " + _bbi.last.toString() + " ms";
    }

    private function resetStats() as Void {
        _bbi = new BbiBuffer();
        _batchCount = 0;
        _lastBbiMs = -1;
        _emptyCallbacks = 0;
        _accelSamples = 0;
        _loggedSamples = 0;
        _callbackCount = 0;
        _elapsedMs = 0;
        _startUnix = 0;
        _lastError = null;
        _fields = [];
        _summaryFields = [];
    }

    private function addField(name as String, id as Number, type as Number, units as String, count as Number, summary as Boolean) as Field {
        // Field-count limits apply across RECORD and SESSION, not just byte sizes.
        if (getFitFieldCount() >= 16) {
            throw new Lang.InvalidValueException("Too many FIT fields");
        }
        return _session.createField(name, id, type, {
            :mesgType => summary ? FitContributor.MESG_TYPE_SESSION : FitContributor.MESG_TYPE_RECORD,
            :units => units, :count => count
        });
    }

    private function setupFitFields() as Void {
        _fields.add(addField("bbi_count", 0, FitContributor.DATA_TYPE_UINT8, "count", 1, false));
        _fields.add(addField("bbi_latest", 1, FitContributor.DATA_TYPE_UINT16, "ms", 1, false));
        _fields.add(addField("accel_samples", 10, FitContributor.DATA_TYPE_UINT32, "samples", 1, false));
        _fields.add(addField("bbi_total", 11, FitContributor.DATA_TYPE_UINT32, "count", 1, false));
        _fields.add(addField("bbi_history", 13, FitContributor.DATA_TYPE_UINT16, "ms", 24, false));
        _fields.add(addField("bbi_rx_ms", 14, FitContributor.DATA_TYPE_UINT32, "ms", 24, false));
        _fields.add(addField("logger_meta", 24, FitContributor.DATA_TYPE_UINT32, "mixed", 8, false));
        _summaryFields.add(addField("final_bbi_history", 40, FitContributor.DATA_TYPE_UINT16, "ms", 24, true));
        _summaryFields.add(addField("final_bbi_rx_ms", 41, FitContributor.DATA_TYPE_UINT32, "ms", 24, true));
        _summaryFields.add(addField("final_bbi_total", 42, FitContributor.DATA_TYPE_UINT32, "count", 1, true));
        _summaryFields.add(addField("final_logger_meta", 51, FitContributor.DATA_TYPE_UINT32, "mixed", 8, true));
        _fields[0].setData(0);
        _fields[1].setData(0);
    }

    private function publishFields() as Void {
        if (_fields.size() != 7) { return; }
        _fields[2].setData(_accelSamples);
        _fields[3].setData(_bbi.total);
        _fields[4].setData(_bbi.values.slice(0, 24));
        _fields[5].setData(_bbi.receivedMs.slice(0, 24));
        _fields[6].setData(getMetadata());
    }

    private function publishSummary() as Void {
        if (_summaryFields.size() != 4) { return; }
        _summaryFields[0].setData(_bbi.values.slice(0, 24));
        _summaryFields[1].setData(_bbi.receivedMs.slice(0, 24));
        _summaryFields[2].setData(_bbi.total);
        _summaryFields[3].setData(getMetadata());
    }

    private function getMetadata() as Array<Number> {
        // Schema, history count, callbacks, empty callbacks, invalid intervals,
        // native sample count, elapsed milliseconds, and Unix-second anchor.
        return [3, _bbi.count(), _callbackCount, _emptyCallbacks, _bbi.invalidTotal,
            _loggedSamples, getElapsedMs(), _startUnix];
    }

    private function updateLoggedSamples() as Void {
        if (_logger != null) {
            var stats = _logger.getStats();
            if (stats != null) { _loggedSamples = stats.sampleCount; }
        }
    }

    private function releaseSensors() as Void {
        if (_listenerRegistered) {
            Sensor.unregisterSensorDataListener();
            _listenerRegistered = false;
        }
        if (_hrEnabled) {
            Sensor.disableSensorType(Sensor.SENSOR_HEARTRATE);
            _hrEnabled = false;
        }
    }

    private function cleanupUnstartedSession() as Void {
        try {
            releaseSensors();
            if (_session != null && !_session.isRecording() && _session.discard()) {
                _session = null;
                _logger = null;
            }
        } catch (e) {
            System.println("Cleanup: " + e.getErrorMessage());
        }
    }

    private function fail(status as String, message as String) as Boolean {
        _status = status;
        _lastError = message;
        System.println(status + ": " + message);
        WatchUi.requestUpdate();
        return false;
    }

    (:testHelper)
    function testSetSession(session) as Void { _session = session; }
}
