// ui/qml/ErrorPanel.qml — pannello scorrevole con lo storico degli errori
// di esecuzione (turni LLM, terminale, backend). Si apre dal badge nella
// barra superiore; ogni voce mostra ora, sorgente e messaggio.

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: panel
    property ListModel errors
    property bool open: false

    color: Qt.rgba(0.055, 0.07, 0.10, 0.92)
    border.color: Qt.rgba(1, 1, 1, 0.12)
    radius: 18
    height: open ? Math.min(280, 64 + errors.count * 58) : 0
    opacity: open ? 1.0 : 0.0
    visible: height > 2
    clip: true

    Behavior on height  { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
    Behavior on opacity { NumberAnimation { duration: 180 } }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "Errori di esecuzione"
                color: "#e6edf3"
                font.pixelSize: 13
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "Svuota"
                visible: panel.errors.count > 0
                font.pixelSize: 11
                background: Rectangle {
                    color: parent.hovered ? "#2d333b" : "transparent"
                    border.color: "#2d333b"; radius: 8
                }
                contentItem: Text { text: parent.text; color: "#8b949e"; font: parent.font
                                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                onClicked: panel.errors.clear()
            }
            Button {
                text: "✕"
                font.pixelSize: 12
                background: Rectangle { color: parent.hovered ? "#2d333b" : "transparent"; radius: 8 }
                contentItem: Text { text: parent.text; color: "#8b949e"; font: parent.font
                                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                onClicked: panel.open = false
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: panel.errors
            spacing: 6
            clip: true
            ScrollBar.vertical: ScrollBar {
                contentItem: Rectangle { implicitWidth: 4; radius: 2; color: "#2d333b" }
            }

            Text {
                anchors.centerIn: parent
                visible: panel.errors.count === 0
                text: "Nessun errore — tutto liscio ✓"
                color: "#8b949e"
                font.pixelSize: 12
            }

            delegate: Rectangle {
                width: ListView.view.width
                height: errCol.height + 16
                radius: 10
                color: Qt.rgba(1, 1, 1, 0.05)
                border.color: Qt.rgba(1, 0.42, 0.42, 0.35)

                Column {
                    id: errCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.margins: 10
                    spacing: 2
                    Row {
                        spacing: 8
                        Text { text: model.time;   color: "#8b949e"; font.pixelSize: 10 }
                        Text { text: model.source; color: "#f03e3e"; font.pixelSize: 10
                               font.capitalization: Font.AllUppercase; font.bold: true }
                    }
                    Text {
                        width: errCol.width
                        text: model.message
                        color: "#e6edf3"
                        font.pixelSize: 12
                        wrapMode: Text.Wrap
                        maximumLineCount: 3
                        elide: Text.ElideRight
                    }
                }
            }
        }
    }
}
