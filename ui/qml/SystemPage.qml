// ui/qml/SystemPage.qml — sezione "Sistema": stato dei servizi, GPU/VRAM,
// modelli caricati, statistiche della sessione con sparkline delle latenze
// LLM e regolazione della velocità della voce. Si aggiorna da sola ogni 4s
// mentre è visibile (backend.requestSystemStatus → systemStatus).

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import "."

Item {
    id: page
    property var status: ({})                 // ultimo payload _system
    property ListModel latencyHistory         // {ms} — alimentata da Main
    property bool fresh: false

    function handleStatus(p) { status = p; fresh = true }
    function refreshSpark() { spark.revision++ }

    Timer {
        interval: 4000
        running: page.visible
        repeat: true
        triggeredOnStart: true
        onTriggered: backend.requestSystemStatus()
    }

    component SectionCard: Rectangle {
        default property alias inner: box.data
        property string title: ""
        Layout.fillWidth: true
        implicitHeight: box.implicitHeight + head.height + 30
        radius: Theme.radiusIn
        color: Theme.cardStrong
        border.color: Theme.cardLine

        Text {
            id: head
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.margins: 14
            text: parent.title
            color: Theme.textDim
            font.pixelSize: 11
            font.bold: true
            font.capitalization: Font.AllUppercase
        }
        ColumnLayout {
            id: box
            anchors.top: head.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.margins: 14
            anchors.topMargin: 8
            spacing: 8
        }
    }

    component StatusDot: Row {
        property string label: ""
        property bool ok: false
        property string detail: ""
        spacing: 8
        Rectangle {
            width: 9; height: 9; radius: 4.5
            anchors.verticalCenter: parent.verticalCenter
            color: ok ? "#37b24d" : "#e5484d"
            SequentialAnimation on opacity {
                running: ok
                loops: Animation.Infinite
                NumberAnimation { to: 0.5; duration: 1200; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0; duration: 1200; easing.type: Easing.InOutSine }
            }
        }
        Text { text: label; color: Theme.text; font.pixelSize: 13
               anchors.verticalCenter: parent.verticalCenter }
        Text { text: detail; color: Theme.textDim; font.pixelSize: 11
               anchors.verticalCenter: parent.verticalCenter }
    }

    Flickable {
        anchors.fill: parent
        anchors.leftMargin: 20
        anchors.rightMargin: 20
        anchors.bottomMargin: 16
        contentHeight: grid.implicitHeight
        clip: true
        ScrollBar.vertical: ScrollBar {
            contentItem: Rectangle { implicitWidth: 5; radius: 3; color: Theme.scrollBar }
        }

        GridLayout {
            id: grid
            width: parent.width
            columns: width > 760 ? 2 : 1
            columnSpacing: 12
            rowSpacing: 12

            // ── servizi ─────────────────────────────────────────────
            SectionCard {
                title: "Servizi"
                StatusDot {
                    label: "Ollama"
                    ok: (page.status.ollama || {}).ok === true
                    detail: (page.status.ollama || {}).version
                            ? "v" + page.status.ollama.version : "non raggiungibile"
                }
                StatusDot {
                    label: "Server TTS"
                    ok: (page.status.tts || {}).ok === true
                    detail: (page.status.tts || {}).profile
                            ? "voce " + page.status.tts.profile : "non attivo"
                }
                StatusDot {
                    label: "SearXNG (web)"
                    ok: page.status.web_ok === true
                    detail: page.status.web_ok ? "" : "non raggiungibile"
                }
                StatusDot {
                    label: "Wake word"
                    ok: (page.status.wake || {}).enabled === true
                    detail: (page.status.wake || {}).enabled
                            ? "«" + String((page.status.wake || {}).model || "").replace(/_/g, " ") + "»"
                            : "disattivata"
                }
            }

            // ── GPU ─────────────────────────────────────────────────
            SectionCard {
                title: "GPU · VRAM"
                Repeater {
                    model: page.status.gpus || []
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 3
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                Layout.fillWidth: true
                                text: modelData.name
                                color: Theme.text; font.pixelSize: 12
                                elide: Text.ElideRight
                            }
                            Text {
                                text: (modelData.used / 1024).toFixed(1) + " / "
                                      + (modelData.total / 1024).toFixed(1) + " GB · "
                                      + modelData.util + "%"
                                color: Theme.textDim; font.pixelSize: 11
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 7; radius: 3.5
                            color: Theme.scrollBar
                            Rectangle {
                                width: parent.width * Math.min(1, modelData.used / modelData.total)
                                height: parent.height; radius: 3.5
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0; color: "#4dabf7" }
                                    GradientStop { position: 1; color: "#9775fa" }
                                }
                                Behavior on width { NumberAnimation { duration: 500; easing.type: Easing.OutCubic } }
                            }
                        }
                    }
                }
                Text {
                    visible: !(page.status.gpus || []).length
                    text: "nvidia-smi non disponibile"
                    color: Theme.textDim; font.pixelSize: 12
                }
            }

            // ── modelli in VRAM ─────────────────────────────────────
            SectionCard {
                title: "Modelli caricati"
                Repeater {
                    model: (page.status.ollama || {}).models || []
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        Text {
                            Layout.fillWidth: true
                            text: modelData.name
                            color: Theme.text; font.pixelSize: 12
                            elide: Text.ElideMiddle
                        }
                        Rectangle {
                            width: vramText.width + 14; height: 20; radius: 10
                            color: Theme.accentSoft
                            border.color: Theme.accentLine
                            Text {
                                id: vramText
                                anchors.centerIn: parent
                                text: (modelData.vram_mb / 1024).toFixed(1) + " GB"
                                color: Theme.accent; font.pixelSize: 10; font.bold: true
                            }
                        }
                    }
                }
                Text {
                    visible: !((page.status.ollama || {}).models || []).length
                    text: "Nessun modello in VRAM (si caricano al primo uso)"
                    color: Theme.textDim; font.pixelSize: 12
                }
            }

            // ── statistiche + sparkline ─────────────────────────────
            SectionCard {
                title: "Sessione · latenza LLM"
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 18
                    Repeater {
                        model: [
                            { k: "turni",   v: (page.status.stats || {}).turns },
                            { k: "parole →", v: (page.status.stats || {}).words_in },
                            { k: "parole ←", v: (page.status.stats || {}).words_out },
                            { k: "errori",  v: ((page.status.stats || {}).stt_errors || 0)
                                             + ((page.status.stats || {}).tts_errors || 0) },
                        ]
                        delegate: Column {
                            spacing: 1
                            Text {
                                text: modelData.v === undefined ? "—" : String(modelData.v)
                                color: Theme.text; font.pixelSize: 19; font.bold: true
                            }
                            Text { text: modelData.k; color: Theme.textDim; font.pixelSize: 10 }
                        }
                    }
                    Item { Layout.fillWidth: true }
                }

                // sparkline delle ultime latenze LLM
                Canvas {
                    id: spark
                    Layout.fillWidth: true
                    Layout.preferredHeight: 44
                    property int revision: 0
                    onRevisionChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        var n = page.latencyHistory ? page.latencyHistory.count : 0
                        if (n < 2) return
                        var maxV = 1
                        for (var i = 0; i < n; i++)
                            maxV = Math.max(maxV, page.latencyHistory.get(i).ms)
                        ctx.strokeStyle = Theme.accent
                        ctx.lineWidth = 2
                        ctx.lineJoin = "round"
                        ctx.beginPath()
                        for (i = 0; i < n; i++) {
                            var x = (width - 4) * i / (n - 1) + 2
                            var y = height - 4 - (height - 10) * page.latencyHistory.get(i).ms / maxV
                            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                        }
                        ctx.stroke()
                        // punto finale
                        ctx.fillStyle = Theme.accent
                        ctx.beginPath()
                        ctx.arc((width - 4) + 2 - (width - 4) / (n - 1) * 0, 0, 0, 0, 0) // no-op
                        ctx.closePath()
                    }
                }
                Text {
                    text: page.latencyHistory && page.latencyHistory.count > 0
                          ? "ultimo turno: "
                            + (page.latencyHistory.get(page.latencyHistory.count - 1).ms / 1000).toFixed(1)
                            + "s (llm)"
                          : "nessun turno ancora"
                    color: Theme.textDim; font.pixelSize: 11
                }
            }

            // ── voce ────────────────────────────────────────────────
            SectionCard {
                title: "Voce · velocità parlato"
                Layout.columnSpan: grid.columns
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 14
                    Text { text: "🐢"; font.pixelSize: 14; font.family: "Noto Color Emoji" }
                    Slider {
                        id: speedSlider
                        Layout.fillWidth: true
                        from: 0.8; to: 1.5; stepSize: 0.05
                        value: page.status.tts_speed || 1.0
                        onPressedChanged: {
                            if (!pressed) backend.setTtsSpeed(value)
                        }
                        background: Rectangle {
                            x: speedSlider.leftPadding
                            y: speedSlider.topPadding + speedSlider.availableHeight / 2 - height / 2
                            width: speedSlider.availableWidth; height: 6; radius: 3
                            color: Theme.scrollBar
                            Rectangle {
                                width: speedSlider.visualPosition * parent.width
                                height: parent.height; radius: 3
                                color: Theme.accent
                            }
                        }
                        handle: Rectangle {
                            x: speedSlider.leftPadding + speedSlider.visualPosition
                               * (speedSlider.availableWidth - width)
                            y: speedSlider.topPadding + speedSlider.availableHeight / 2 - height / 2
                            width: 18; height: 18; radius: 9
                            color: "white"
                            border.color: Theme.accent
                            border.width: 2
                            scale: speedSlider.pressed ? 1.2 : 1.0
                            Behavior on scale { SpringAnimation { spring: 4; damping: 0.3 } }
                        }
                    }
                    Text { text: "🐇"; font.pixelSize: 14; font.family: "Noto Color Emoji" }
                    Rectangle {
                        width: speedText.width + 16; height: 24; radius: 12
                        color: Theme.accentSoft
                        border.color: Theme.accentLine
                        Text {
                            id: speedText
                            anchors.centerIn: parent
                            text: speedSlider.value.toFixed(2) + "×"
                            color: Theme.accent; font.pixelSize: 11; font.bold: true
                        }
                    }
                }
                Text {
                    text: "1.00× = voce naturale (nessun processing). Applicata subito e ricordata al riavvio."
                    color: Theme.textDim; font.pixelSize: 11
                }
            }
        }
    }
}
