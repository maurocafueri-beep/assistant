// ui/qml/Glass.qml — card "frosted glass" su tema CHIARO (stile reference:
// pastelli soffici, card bianche traslucide, ombre leggere e diffuse).
// I contenuti vanno nel default property (children).

import QtQuick
import QtQuick.Effects
import "."

Item {
    id: glass
    default property alias content: inner.data
    property real  glassRadius: Theme.radius
    property color fill: Theme.card
    property color line: Theme.cardLine
    property bool  shadow: true
    property real  shadowStrength: 0.10

    MultiEffect {
        visible: glass.shadow
        source: panel
        anchors.fill: panel
        shadowEnabled: true
        shadowBlur: 1.0
        shadowOpacity: glass.shadowStrength
        shadowVerticalOffset: 8
        shadowColor: Theme.shadowCol
    }

    Rectangle {
        id: panel
        anchors.fill: parent
        radius: glass.glassRadius
        color: glass.fill
        border.width: 1
        border.color: glass.line

        // sheen delicato dall'alto (vetro satinato)
        Rectangle {
            anchors.fill: parent
            radius: parent.radius
            gradient: Gradient {
                GradientStop { position: 0.0; color: Theme.dark ? Qt.rgba(1,1,1,0.05) : Qt.rgba(1,1,1,0.35) }
                GradientStop { position: 0.5; color: Theme.dark ? Qt.rgba(1,1,1,0.01) : Qt.rgba(1,1,1,0.05) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }

        Item {
            id: inner
            anchors.fill: parent
        }
    }
}
