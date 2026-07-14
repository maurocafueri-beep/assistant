// ui/qml/CodeBlock.qml — blocco di codice in stile Claude: contenitore
// scuro arrotondato (anche a tema chiaro), header con linguaggio e pulsante
// copia, testo monospace selezionabile.

import QtQuick
import QtQuick.Controls.Basic
import "."

Rectangle {
    id: block
    property string code: ""
    property string language: ""

    implicitHeight: header.height + codeText.implicitHeight + 24
    radius: 12
    color: Theme.dark ? "#0b0e14" : "#1b1f27"
    border.width: 1
    border.color: Qt.rgba(1, 1, 1, 0.08)

    // ── header: linguaggio + copia ──────────────────────────────────
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 34
        radius: parent.radius
        color: Qt.rgba(1, 1, 1, 0.05)
        // squadra i soli angoli inferiori dell'header
        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: parent.radius
            color: parent.color
        }

        Text {
            anchors.left: parent.left
            anchors.leftMargin: 14
            anchors.verticalCenter: parent.verticalCenter
            text: block.language || "testo"
            color: "#8b95a8"
            font.pixelSize: 11
            font.family: "JetBrainsMono Nerd Font"
        }

        Rectangle {
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.verticalCenter: parent.verticalCenter
            width: copyRow.width + 16
            height: 22
            radius: 7
            color: copyArea.containsMouse ? Qt.rgba(1, 1, 1, 0.12) : "transparent"
            Behavior on color { ColorAnimation { duration: 100 } }

            Row {
                id: copyRow
                anchors.centerIn: parent
                spacing: 5
                Text {
                    text: copyArea.copied ? "✓" : "⧉"
                    color: copyArea.copied ? "#51cf66" : "#8b95a8"
                    font.pixelSize: 11
                    anchors.verticalCenter: parent.verticalCenter
                }
                Text {
                    text: copyArea.copied ? "Copiato" : "Copia"
                    color: copyArea.copied ? "#51cf66" : "#8b95a8"
                    font.pixelSize: 11
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
            MouseArea {
                id: copyArea
                property bool copied: false
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    codeText.selectAll()
                    codeText.copy()
                    codeText.deselect()
                    copied = true
                    resetCopied.restart()
                }
                Timer { id: resetCopied; interval: 1600; onTriggered: copyArea.copied = false }
            }
        }
    }

    TextEdit {
        id: codeText
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 14
        anchors.topMargin: 10
        text: block.code
        color: "#dce3ee"
        font.family: "JetBrainsMono Nerd Font"
        font.pixelSize: 13
        wrapMode: TextEdit.WrapAnywhere
        readOnly: true
        selectByMouse: true
        selectionColor: "#3d7ef7"
    }
}
