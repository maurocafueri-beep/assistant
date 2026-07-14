// ui/qml/Ripple.qml — effetto ripple in filosofia Material: un cerchio che
// si espande dal punto di pressione e sfuma. Da mettere DENTRO il background
// del controllo (eredita il radius del genitore per il clipping visivo).
//
// Uso:
//     background: Rectangle {
//         radius: 12
//         color: ...
//         Ripple { id: rip; rippleColor: Qt.rgba(0,0,0,0.10) }
//     }
//     onPressed: rip.trigger(pressX, pressY)   // o trigger() dal centro

import QtQuick

Item {
    id: root
    anchors.fill: parent
    clip: true
    property color rippleColor: Qt.rgba(0, 0, 0, 0.10)

    function trigger(px, py) {
        circle.x = (px === undefined ? width / 2 : px) - circle.maxD / 2
        circle.y = (py === undefined ? height / 2 : py) - circle.maxD / 2
        anim.restart()
    }

    Rectangle {
        id: circle
        readonly property real maxD: Math.max(root.width, root.height) * 2.2
        width: maxD
        height: maxD
        radius: maxD / 2
        color: root.rippleColor
        scale: 0
        opacity: 0
    }

    ParallelAnimation {
        id: anim
        NumberAnimation {
            target: circle; property: "scale"
            from: 0.05; to: 1.0
            duration: 420
            easing.type: Easing.OutCubic
        }
        SequentialAnimation {
            NumberAnimation { target: circle; property: "opacity"
                              from: 0; to: 1; duration: 90 }
            NumberAnimation { target: circle; property: "opacity"
                              to: 0; duration: 330; easing.type: Easing.InQuad }
        }
    }
}
