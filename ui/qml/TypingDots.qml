// ui/qml/TypingDots.qml — indicatore "sta scrivendo" (tre punti che ondeggiano)
import QtQuick

Row {
    id: dots
    spacing: 5
    property color dotColor: "#8b949e"

    Repeater {
        model: 3
        delegate: Rectangle {
            width: 7; height: 7; radius: 3.5
            color: dots.dotColor
            anchors.verticalCenter: parent.verticalCenter

            SequentialAnimation on y {
                running: dots.visible
                loops: Animation.Infinite
                PauseAnimation { duration: index * 140 }
                NumberAnimation { to: -5; duration: 260; easing.type: Easing.OutQuad }
                NumberAnimation { to: 0;  duration: 260; easing.type: Easing.InQuad }
                PauseAnimation { duration: (2 - index) * 140 }
            }
        }
    }
}
