import Toybox.Lang;
import Toybox.WatchUi;

class OvernightAccelBbiDelegate extends WatchUi.BehaviorDelegate {
    private var _view as OvernightAccelBbiView;

    function initialize(view as OvernightAccelBbiView) {
        BehaviorDelegate.initialize();
        _view = view;
    }

    function onSelect() as Boolean {
        _view.toggleRecording();
        return true;
    }

    function onMenu() as Boolean {
        _view.refresh();
        return true;
    }

    function onNextPage() as Boolean {
        _view.refresh();
        return true;
    }

    function onPreviousPage() as Boolean {
        _view.refresh();
        return true;
    }

    function onBack() as Boolean {
        if (_view.hasSession()) {
            _view.stopRecording();
            // Keep the result visible, and never exit after a failed save.
            return true;
        }
        return false;
    }
}
