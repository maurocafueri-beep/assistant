// ui/qml/ThemedCombo.qml — ComboBox a tema scuro con etichetta piccola.
// Emette picked(name) SOLO su scelta dell'utente (non sugli aggiornamenti
// programmatici di `current` che arrivano dal backend).

import QtQuick
import QtQuick.Controls.Basic
import "."

Item {
    id: wrap
    property var comboModel: []
    property string current: ""
    property string label: ""
    signal picked(string name)

    implicitHeight: 40

    Column {
        anchors.fill: parent
        spacing: 1

        Text {
            text: wrap.label
            color: Theme.textDim
            font.pixelSize: 9
            font.capitalization: Font.AllUppercase
            leftPadding: 8
        }

        ComboBox {
            id: combo
            width: wrap.width
            height: 28
            model: wrap.comboModel
            font.pixelSize: 12

            // Sincronizza dal backend senza scatenare picked()
            property bool syncing: false
            Connections {
                target: wrap
                function onCurrentChanged() { combo.syncTo(wrap.current) }
                function onComboModelChanged() { combo.syncTo(wrap.current) }
            }
            Component.onCompleted: syncTo(wrap.current)
            function syncTo(name) {
                syncing = true
                var i = -1
                for (var k = 0; k < model.length; k++)
                    if (model[k] === name) { i = k; break }
                currentIndex = i
                syncing = false
            }
            onActivated: (index) => {
                if (!syncing && index >= 0)
                    wrap.picked(model[index])
            }

            background: Rectangle {
                color: combo.hovered ? Theme.cardHover : Theme.card
                border.color: Theme.cardLine
                radius: 12
                Behavior on color { ColorAnimation { duration: 120 } }
            }
            contentItem: Text {
                text: combo.currentIndex >= 0 ? combo.displayText : wrap.current
                color: Theme.text
                font: combo.font
                leftPadding: 8
                rightPadding: 22
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideMiddle
            }
            indicator: Text {
                x: combo.width - width - 8
                anchors.verticalCenter: parent.verticalCenter
                text: "▾"
                color: Theme.textDim
                font.pixelSize: 10
            }
            popup: Popup {
                y: combo.height + 4
                width: Math.max(combo.width, 260)
                padding: 4
                background: Rectangle {
                    color: Theme.cardStrong
                    border.color: Theme.cardLine
                    radius: 14
                }
                contentItem: ListView {
                    implicitHeight: Math.min(contentHeight, 320)
                    clip: true
                    model: combo.popup.visible ? combo.delegateModel : null
                    ScrollBar.vertical: ScrollBar {
                        contentItem: Rectangle { implicitWidth: 4; radius: 2; color: Theme.scrollBar }
                    }
                }
            }
            delegate: ItemDelegate {
                width: combo.popup.width - 8
                height: 30
                highlighted: combo.highlightedIndex === index
                background: Rectangle {
                    color: highlighted ? Theme.accentSoft : "transparent"
                    radius: 6
                }
                contentItem: Text {
                    text: modelData
                    color: Theme.text
                    font.pixelSize: 12
                    leftPadding: 6
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideMiddle
                }
            }
        }
    }
}
