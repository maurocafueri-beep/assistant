// ui/qml/Glass.qml — superficie tonale in filosofia Material (MD3).
// Il nome resta "Glass" per compatibilità con le pagine, ma il linguaggio è
// cambiato: niente trasparenze né bordi speculari — la gerarchia nasce dal
// TONO della superficie e dall'ELEVAZIONE (ombra morbida).

import QtQuick
import QtQuick.Effects
import "."

Item {
    id: glass
    default property alias content: inner.data
    property real  glassRadius: Theme.radius
    property color fill: Theme.card
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

        Item {
            id: inner
            anchors.fill: parent
        }
    }
}
