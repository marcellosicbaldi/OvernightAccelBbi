import Toybox.Lang;

// Repeated snapshots retain recent intervals across missed FIT record writes.
// Slots are addressed by (one-based sequence - 1) % CAPACITY.
class BbiBuffer {
    const CAPACITY = 24;
    var values as Array<Number>;
    var receivedMs as Array<Number>;
    var total as Number = 0;
    var invalidTotal as Number = 0;
    var last as Number = 0;

    function initialize() {
        values = [];
        receivedMs = [];
        for (var i = 0; i < CAPACITY; i++) {
            values.add(0);
            receivedMs.add(0);
        }
    }

    function append(intervals as Array<Number>?, elapsedMs as Number) as Void {
        if (intervals == null) { return; }
        for (var i = 0; i < intervals.size(); i++) {
            var value = intervals[i];
            // Zero marks an unrepresentable/invalid interval, never a valid beat.
            if (value <= 0 || value >= 65535) {
                invalidTotal++;
                value = 0;
            }
            var slot = total % CAPACITY;
            values[slot] = value;
            receivedMs[slot] = elapsedMs;
            total++;
            last = value;
        }
    }

    function count() as Number {
        return total < CAPACITY ? total : CAPACITY;
    }
}
