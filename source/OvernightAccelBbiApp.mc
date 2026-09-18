import Toybox.Application;
import Toybox.Lang;
import Toybox.WatchUi;

class OvernightAccelBbiApp extends Application.AppBase {
    private var _recorder;

    function initialize() {
        AppBase.initialize();
    }

    // onStart() is called on application start up
    function onStart(state as Dictionary?) as Void {
    }

    // onStop() is called when your application is exiting
    function onStop(state as Dictionary?) as Void {
        if (_recorder != null) {
            _recorder.stopAndSave();
        }
    }

    // Return the initial view of your application here
    function getInitialView() as [Views] or [Views, InputDelegates] {
        _recorder = new OvernightAccelBbiRecorder();
        var view = new OvernightAccelBbiView(_recorder);
        return [ view, new OvernightAccelBbiDelegate(view) ];
    }

}

function getApp() as OvernightAccelBbiApp {
    return Application.getApp() as OvernightAccelBbiApp;
}
