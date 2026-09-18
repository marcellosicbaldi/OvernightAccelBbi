import Toybox.Lang;
import Toybox.System;
import Toybox.WatchUi;

class OvernightAccelBbiMenuDelegate extends WatchUi.MenuInputDelegate {

    function initialize() {
        MenuInputDelegate.initialize();
    }

    function onMenuItem(item as Symbol) as Void {
        if (item == :item_1) {
            System.println("Start/stop is handled by the main view.");
        } else if (item == :item_2) {
            System.println("Back exits after saving an active recording.");
        }
    }

}
