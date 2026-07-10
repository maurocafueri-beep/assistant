// ui/qml/ErrorPanel.qml — pannello scorrevole con lo storico degli errori
// di esecuzione (turni LLM, terminale, backend). Si apre dal badge nella
// barra superiore; ogni voce mostra ora, sorgente e messaggio.

import QtQuick
import QtQuick.Controls.Basic
import "."
import QtQuick.Layouts

Rectangle {
    id: panel
    property ListModel errors
    property bool open: false

    color: Theme.cardStrong
    border.color: Theme.cardLine
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
                color: Theme.text
                font.pixelSize: 13
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "Svuota"
                visible: panel.errors.count > 0
                font.pixelSize: 11
                background: Rectangle {
                    color: parent.hovered ? Theme.cardHover : "transparent"
                    border.color: Theme.cardLine; radius: 8
                }
                contentItem: Text { text: parent.text; color: Theme.textDim; font: parent.font
                                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                onClicked: panel.errors.clear()
            }
            Button {
                text: "✕"
                font.pixelSize: 12
                background: Rectangle { color: parent.hovered ? Theme.cardHover : "transparent"; radius: 8 }
                contentItem: Text { text: parent.text; color: Theme.textDim; font: parent.font
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
                contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.scrollBar }
            }

            Text {
                anchors.centerIn: parent
                visible: panel.errors.count === 0
                text: "Nessun errore — tutto liscio ✓"
                color: Theme.textDim
                font.pixelSize: 12
            }

            delegate: Rectangle {
                width: ListView.view.width
                height: errCol.height + 16
                radius: 10
                color: Theme.dark ? Qt.rgba(0.30, 0.16, 0.17, 0.6) : Qt.rgba(0.98, 0.94, 0.94, 0.9)
                border.color: Qt.rgba(0.90, 0.28, 0.30, 0.30)

                Column {
                    id: errCol
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.margins: 10
                    spacing: 2
                    Row {
                        spacing: 8
                        Text { text: model.time;   color: Theme.textDim; font.pixelSize: 10 }
                        Text { text: model.source; color: Theme.danger; font.pixelSize: 10
                               font.capitalization: Font.AllUppercase; font.bold: true }
                    }
                    Text {
                        width: errCol.width
                        text: model.message
                        color: Theme.text
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
