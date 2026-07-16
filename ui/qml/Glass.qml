// ui/qml/Glass.qml — superficie di vetro traslucido in stile macOS:
// riempimento semi-trasparente (il desktop filtra attraverso la finestra),
// filo speculare superiore appena percettibile, ombra morbida. La gerarchia
// resta tonale (MD3), la materia torna vetro.

import QtQuick
import QtQuick.Effects
import "."

Item {
    id: glass
    default property alias content: inner.data
    property real  glassRadius: Theme.radius
    property color fill: Theme.panelGlass
    property color line: "transparent"      // MD3: superfici senza bordo
    property bool  shadow: true
    property real  shadowStrength: 0.14     // elevazione 1-2

    MultiEffect {
        visible: glass.shadow
        source: panel
        anchors.fill: panel
        shadowEnabled: true
        shadowBlur: 0.7
        shadowOpacity: glass.shadowStrength
        shadowVerticalOffset: 3
        shadowColor: Theme.shadowCol
    }

    Rectangle {
        id: panel
        anchors.fill: parent
        radius: glass.glassRadius
        color: glass.fill
        border.width: glass.line.a > 0 ? 1 : 0
        border.color: glass.line

        // filo speculare superiore (luce radente, quasi impercettibile)
        Rectangle {
            anchors.top: parent.top
            anchors.topMargin: 1
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width - parent.radius * 1.8
            height: 1
            opacity: 0.8
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "transparent" }
                GradientStop { position: 0.5; color: Theme.specular }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }

        Item {
            id: inner
            anchors.fill: parent
        }
    }
}
