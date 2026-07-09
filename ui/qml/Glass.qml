// ui/qml/Glass.qml — pannello "Liquid Glass" in stile macOS Tahoe.
// Vetro traslucido: riempimento scuro semi-trasparente, bordo speculare
// (più luminoso in alto, come luce radente), sheen verticale tenue e
// ombra morbida. I contenuti vanno nel default property (children).

import QtQuick
import QtQuick.Effects

Item {
    id: glass
    default property alias content: inner.data
    property real  glassRadius: 22
    property color tint: "transparent"        // velatura opzionale (es. accent)
    property real  fillOpacity: 0.45
    property bool  shadow: true

    // ombra morbida sotto il pannello
    MultiEffect {
        visible: glass.shadow
        source: panel
        anchors.fill: panel
        shadowEnabled: true
        shadowBlur: 0.9
        shadowOpacity: 0.45
        shadowVerticalOffset: 6
        shadowColor: "#000000"
    }

    Rectangle {
        id: panel
        anchors.fill: parent
        radius: glass.glassRadius
        color: Qt.rgba(0.086, 0.106, 0.133, glass.fillOpacity)   // #161b22 traslucido
        border.width: 1
        border.color: Qt.rgba(1, 1, 1, 0.10)

        // velatura colorata opzionale
        Rectangle {
            anchors.fill: parent
            radius: parent.radius
            color: glass.tint
            opacity: glass.tint.a > 0 ? 0.16 : 0
        }

        // sheen: luce radente dall'alto
        Rectangle {
            anchors.fill: parent
            radius: parent.radius
            gradient: Gradient {
                GradientStop { position: 0.0;  color: Qt.rgba(1, 1, 1, 0.065) }
                GradientStop { position: 0.28; color: Qt.rgba(1, 1, 1, 0.015) }
                GradientStop { position: 1.0;  color: "transparent" }
            }
        }

        // bordo speculare superiore (highlight 1px che sfuma ai lati)
        Rectangle {
            anchors.top: parent.top
            anchors.topMargin: 1
            anchors.horizontalCenter: parent.horizontalCenter
            width: parent.width - parent.radius * 1.6
            height: 1
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: "transparent" }
                GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.22) }
                GradientStop { position: 1.0; color: "transparent" }
            }
        }

        Item {
            id: inner
            anchors.fill: parent
        }
    }
}
