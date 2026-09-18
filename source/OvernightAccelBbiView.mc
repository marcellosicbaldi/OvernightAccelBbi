import Toybox.Graphics;
import Toybox.Lang;
import Toybox.Math;
import Toybox.WatchUi;

class OvernightAccelBbiView extends WatchUi.View {
    private var _recorder as OvernightAccelBbiRecorder;

    function initialize(recorder as OvernightAccelBbiRecorder) {
        View.initialize();
        _recorder = recorder;
    }

    function onUpdate(dc as Dc) as Void {
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_BLACK);
        dc.clear();
        var h = dc.getHeight();
        drawLine(dc, h * 15 / 100, "Overnight v" + _recorder.VERSION);
        dc.setColor(_recorder.getLastError() != null ? Graphics.COLOR_RED :
            (_recorder.isRecording() ? Graphics.COLOR_RED : Graphics.COLOR_GREEN), Graphics.COLOR_TRANSPARENT);
        drawLine(dc, h * 26 / 100, _recorder.getStatus());
        dc.setColor(Graphics.COLOR_WHITE, Graphics.COLOR_TRANSPARENT);
        var seconds = _recorder.getElapsedSeconds();
        drawLine(dc, h * 36 / 100, twoDigits(seconds / 3600) + ":" +
            twoDigits((seconds % 3600) / 60) + ":" + twoDigits(seconds % 60));
        drawLine(dc, h * 46 / 100, "FIT accel " + _recorder.getLoggedSamples().toString());
        drawLine(dc, h * 56 / 100, _recorder.getBbiStatus());
        drawLine(dc, h * 66 / 100, "BBI received " + _recorder.getTotalBbiCount().toString());
        var error = _recorder.getLastError();
        dc.setColor(error == null ? Graphics.COLOR_LT_GRAY : Graphics.COLOR_RED, Graphics.COLOR_TRANSPARENT);
        drawLine(dc, h * 76 / 100, error == null ?
            "Callbacks " + _recorder.getCallbackCount().toString() : error);
    }

    function toggleRecording() as Boolean {
        return _recorder.hasSession() ? _recorder.stopAndSave() : _recorder.start();
    }

    function stopRecording() as Boolean { return _recorder.stopAndSave(); }
    function hasSession() as Boolean { return _recorder.hasSession(); }

    function refresh() as Void { WatchUi.requestUpdate(); }

    private function drawLine(dc as Dc, y as Number, text as String) as Void {
        var font = Graphics.FONT_XTINY;
        // Constrain every line to the round screen's chord, including long errors.
        var radius = dc.getWidth() / 2;
        var edge = y + dc.getFontHeight(font) - dc.getHeight() / 2;
        var top = y - dc.getHeight() / 2;
        if (top * top > edge * edge) { edge = top; }
        var available = (2 * Math.sqrt(radius * radius - edge * edge) - 24).toNumber();
        if (dc.getTextWidthInPixels(text, font) > available) {
            while (text.length() > 0 && dc.getTextWidthInPixels(text + "...", font) > available) {
                text = text.substring(0, text.length() - 1);
            }
            text += "...";
        }
        dc.drawText(radius, y, font, text, Graphics.TEXT_JUSTIFY_CENTER);
    }

    private function twoDigits(value as Number) as String {
        return value < 10 ? "0" + value.toString() : value.toString();
    }
}
