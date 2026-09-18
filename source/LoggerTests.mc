import Toybox.Lang;
import Toybox.Test;
import Toybox.Graphics;

(:test)
function bufferKeepsEqualIntervals(logger as Test.Logger) as Boolean {
    var b = new BbiBuffer();
    b.append([1000, 1000], 1234);
    b.append(null, 2000);
    b.append([], 3000);
    Test.assertEqual(b.total, 2);
    Test.assertEqual(b.values[0], 1000);
    Test.assertEqual(b.values[1], 1000);
    Test.assertEqual(b.receivedMs[1], 1234);
    return true;
}

(:test)
function bufferWrapsAndCountsInvalid(logger as Test.Logger) as Boolean {
    var b = new BbiBuffer();
    for (var i = 1; i <= 30; i++) { b.append([800 + i], i * 1000); }
    Test.assertEqual(b.count(), 24);
    Test.assertEqual(b.total, 30);
    Test.assertEqual(b.values[0], 825);
    Test.assertEqual(b.values[5], 830);
    Test.assertEqual(b.values[6], 807);
    Test.assertEqual(b.receivedMs[5], 30000);
    b.append([0, -1, 65535, 65534], 31000);
    Test.assertEqual(b.invalidTotal, 3);
    Test.assertEqual(b.total, 34);
    Test.assertEqual(b.last, 65534);
    return true;
}

(:test)
class TestSession {
    var saveOk = false;
    var stopOk = true;
    var running = false;
    var saveCalls = 0;
    function initialize() {}
    function isRecording() as Boolean { return running; }
    function stop() as Boolean {
        if (stopOk) { running = false; }
        return stopOk;
    }
    function save() as Boolean { saveCalls++; return saveOk; }
}

(:test)
function saveFailureRetainsSession(logger as Test.Logger) as Boolean {
    var r = new OvernightAccelBbiRecorder();
    var session = new TestSession();
    r.testSetSession(session);
    Test.assertEqual(r.stopAndSave(), false);
    Test.assertEqual(r.hasSession(), true);
    Test.assertEqual(r.getStatus(), "Save failed");
    Test.assertEqual(r.start(), false);
    Test.assertEqual(r.hasSession(), true);
    session.saveOk = true;
    Test.assertEqual(r.stopAndSave(), true);
    Test.assertEqual(r.hasSession(), false);
    Test.assertEqual(r.getStatus(), "Saved");
    Test.assertEqual(r.stopAndSave(), true);
    Test.assertEqual(session.saveCalls, 2);
    return true;
}

(:test)
function stopFailureNeverSaves(logger as Test.Logger) as Boolean {
    var r = new OvernightAccelBbiRecorder();
    var session = new TestSession();
    session.running = true;
    session.stopOk = false;
    r.testSetSession(session);
    Test.assertEqual(r.stopAndSave(), false);
    Test.assertEqual(r.hasSession(), true);
    Test.assertEqual(session.saveCalls, 0);
    return true;
}

(:test)
function simulatorSessionSmoke(logger as Test.Logger) as Boolean {
    var r = new OvernightAccelBbiRecorder();
    Test.assertEqualMessage(r.start(), true, "Actual session/field allocation failed");
    Test.assertEqual(r.isRecording(), true);
    Test.assertEqual(r.getFitFieldCount(), 11);
    Test.assertEqualMessage(r.stopAndSave(), true, "Actual session save failed");
    Test.assertEqual(r.isRecording(), false);
    Test.assertEqual(r.hasSession(), false);
    return true;
}

(:test)
function renderReadyAndSaveFailure(logger as Test.Logger) as Boolean {
    var bitmapRef = Graphics.createBufferedBitmap({:width => 390, :height => 390});
    var bitmap = bitmapRef.get();
    var r = new OvernightAccelBbiRecorder();
    var view = new OvernightAccelBbiView(r);
    view.onUpdate(bitmap.getDc());
    r.testSetSession(new TestSession());
    Test.assertEqual(r.stopAndSave(), false);
    view.onUpdate(bitmap.getDc());
    return true;
}
