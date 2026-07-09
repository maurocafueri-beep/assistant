// ui/qml/TerminalPage.qml — modalità terminale agentica.
// Flusso: richiesta in linguaggio naturale → proposta comando (con
// motivazione e livello di rischio) → conferma/annulla → risultato +
// analisi. La voce (wake word/PTT) popola il box input via
// terminal.transcription: Invio per confermare.

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Item {
    id: page
    // feed: kind ∈ request|proposal|result|analysis
    property ListModel feed
    property string cwd: "~"
    property var pendingProposal: null    // proposta in attesa di conferma

    function handleEvent(p) {
        var t = p.type
        if (t === "terminal.proposal") {
            feed.append({ kind: "proposal", data: JSON.stringify(p.proposal) })
            if (p.proposal.needs_confirmation)
                page.pendingProposal = p.proposal
            feedList.positionViewAtEnd()
        } else if (t === "terminal.result") {
            page.pendingProposal = null
            feed.append({ kind: "result", data: JSON.stringify(p.result) })
            if (p.analysis)
                feed.append({ kind: "analysis", data: JSON.stringify({ text: p.analysis }) })
            feedList.positionViewAtEnd()
        } else if (t === "terminal.cancelled") {
            page.pendingProposal = null
        } else if (t === "terminal.cwd" || t === "terminal.reset") {
            page.cwd = p.cwd || "~"
            if (t === "terminal.reset") { feed.clear(); page.pendingProposal = null }
        } else if (t === "terminal.transcription") {
            termInput.text = p.text
            termInput.forceActiveFocus()
        } else if (t === "terminal.state") {
            page.cwd = p.cwd || "~"
        }
    }

    function submit() {
        if (termInput.text.trim().length === 0) return
        feed.append({ kind: "request", data: JSON.stringify({ text: termInput.text.trim() }) })
        backend.terminalPropose(termInput.text.trim())
        termInput.text = ""
        feedList.positionViewAtEnd()
    }

    readonly property color riskLow:  "#40c057"
    readonly property color riskMed:  "#fab005"
    readonly property color riskHigh: "#f03e3e"
    function riskColor(level) {
        return level === "high" ? riskHigh : level === "medium" ? riskMed : riskLow
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ── barra cwd ───────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 36
            color: "transparent"
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 24
                anchors.rightMargin: 24
                Text { text: "📁"; font.pixelSize: 12 }
                Text {
                    Layout.fillWidth: true
                    text: page.cwd
                    color: "#8b949e"
                    font.family: "monospace"
                    font.pixelSize: 11
                    elide: Text.ElideMiddle
                }
                Button {
                    text: "Reset"
                    font.pixelSize: 10
                    background: Rectangle { color: parent.hovered ? "#1f2630" : "transparent"
                                            border.color: "#2d333b"; radius: 6 }
                    contentItem: Text { text: parent.text; color: "#8b949e"; font: parent.font
                                        horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    onClicked: backend.terminalReset()
                }
            }
        }

        // ── feed ────────────────────────────────────────────────────
        ListView {
            id: feedList
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: page.feed
            spacing: 12
            clip: true
            topMargin: 16; bottomMargin: 16; leftMargin: 24; rightMargin: 24
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {
                contentItem: Rectangle { implicitWidth: 5; radius: 3; color: "#2d333b" }
            }

            add: Transition {
                NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 200 }
                NumberAnimation { property: "y"; from: feedList.height; duration: 240
                                  easing.type: Easing.OutCubic }
            }

            delegate: Loader {
                width: feedList.width - feedList.leftMargin - feedList.rightMargin
                property var d: JSON.parse(model.data)
                property string kind: model.kind
                sourceComponent:
                    kind === "request"  ? requestCard
                  : kind === "proposal" ? proposalCard
                  : kind === "result"   ? resultCard
                  : analysisCard

                // ── richiesta utente ───────────────────────────────
                Component {
                    id: requestCard
                    Rectangle {
                        width: parent ? parent.width : 0
                        height: reqText.implicitHeight + 20
                        radius: 16
                        color: Qt.rgba(0.35, 0.72, 1.0, 0.16)
                        border.width: 1
                        border.color: Qt.rgba(0.35, 0.72, 1.0, 0.32)
                        Text {
                            id: reqText
                            anchors.fill: parent; anchors.margins: 10
                            text: d.text
                            color: "#e6edf3"; font.pixelSize: 13; wrapMode: Text.Wrap
                        }
                    }
                }

                // ── proposta comando ───────────────────────────────
                Component {
                    id: proposalCard
                    Rectangle {
                        width: parent ? parent.width : 0
                        height: propCol.height + 24
                        radius: 16
                        color: Qt.rgba(1, 1, 1, 0.055)
                        border.color: Qt.rgba(page.riskColor(d.risk_level).r,
                                              page.riskColor(d.risk_level).g,
                                              page.riskColor(d.risk_level).b, 0.45)
                        border.width: 1

                        Column {
                            id: propCol
                            anchors.left: parent.left; anchors.right: parent.right
                            anchors.top: parent.top; anchors.margins: 12
                            spacing: 8

                            Row {
                                spacing: 8
                                Rectangle {
                                    width: riskText.width + 14; height: 18; radius: 9
                                    color: Qt.rgba(page.riskColor(d.risk_level).r,
                                                   page.riskColor(d.risk_level).g,
                                                   page.riskColor(d.risk_level).b, 0.18)
                                    Text {
                                        id: riskText
                                        anchors.centerIn: parent
                                        text: d.risk_level
                                        color: page.riskColor(d.risk_level)
                                        font.pixelSize: 10; font.bold: true
                                        font.capitalization: Font.AllUppercase
                                    }
                                }
                                Text {
                                    visible: d.search_used === true
                                    text: "🔎 web"
                                    color: "#8b949e"; font.pixelSize: 10
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }

                            Rectangle {
                                width: propCol.width
                                height: cmdText.implicitHeight + 16
                                radius: 10
                                color: Qt.rgba(0, 0, 0, 0.35)
                                TextEdit {
                                    id: cmdText
                                    anchors.fill: parent; anchors.margins: 8
                                    text: "$ " + d.command
                                    color: "#3bc9db"
                                    font.family: "monospace"; font.pixelSize: 12
                                    wrapMode: Text.WrapAnywhere
                                    readOnly: true; selectByMouse: true
                                }
                            }

                            Text {
                                width: propCol.width
                                text: d.rationale || ""
                                visible: text.length > 0
                                color: "#8b949e"; font.pixelSize: 12; wrapMode: Text.Wrap
                            }

                            Row {
                                spacing: 8
                                visible: page.pendingProposal !== null
                                         && page.pendingProposal.proposal_id === d.proposal_id
                                Button {
                                    text: "▶ Esegui"
                                    font.pixelSize: 12
                                    background: Rectangle { color: parent.hovered ? "#37b24d" : "#2f9e44"; radius: 8 }
                                    contentItem: Text { text: parent.text; color: "white"; font: parent.font
                                                        horizontalAlignment: Text.AlignHCenter
                                                        verticalAlignment: Text.AlignVCenter }
                                    onClicked: backend.terminalConfirm(d.proposal_id)
                                }
                                Button {
                                    text: "Annulla"
                                    font.pixelSize: 12
                                    background: Rectangle { color: parent.hovered ? "#2d333b" : "transparent"
                                                            border.color: "#2d333b"; radius: 8 }
                                    contentItem: Text { text: parent.text; color: "#8b949e"; font: parent.font
                                                        horizontalAlignment: Text.AlignHCenter
                                                        verticalAlignment: Text.AlignVCenter }
                                    onClicked: backend.terminalCancel(d.proposal_id)
                                }
                            }
                        }
                    }
                }

                // ── risultato ──────────────────────────────────────
                Component {
                    id: resultCard
                    Rectangle {
                        width: parent ? parent.width : 0
                        height: resCol.height + 24
                        radius: 16
                        color: Qt.rgba(0, 0, 0, 0.30)
                        border.color: d.success ? Qt.rgba(1, 1, 1, 0.10) : Qt.rgba(1, 0.42, 0.42, 0.5)

                        Column {
                            id: resCol
                            anchors.left: parent.left; anchors.right: parent.right
                            anchors.top: parent.top; anchors.margins: 12
                            spacing: 6

                            Row {
                                spacing: 10
                                Text {
                                    text: d.success ? "✓ exit " + d.exit_code : "✗ exit " + d.exit_code
                                    color: d.success ? "#40c057" : "#f03e3e"
                                    font.pixelSize: 11; font.bold: true
                                }
                                Text {
                                    text: Math.round(d.duration_ms) + " ms"
                                    color: "#8b949e"; font.pixelSize: 11
                                }
                                Text {
                                    visible: d.truncated === true
                                    text: "output troncato"
                                    color: "#fab005"; font.pixelSize: 11
                                }
                            }

                            TextEdit {
                                width: resCol.width
                                visible: (d.stdout || "").length > 0
                                text: d.stdout
                                color: "#e6edf3"
                                font.family: "monospace"; font.pixelSize: 11
                                wrapMode: Text.WrapAnywhere
                                readOnly: true; selectByMouse: true
                            }
                            TextEdit {
                                width: resCol.width
                                visible: (d.stderr || "").length > 0
                                text: d.stderr
                                color: "#ff8787"
                                font.family: "monospace"; font.pixelSize: 11
                                wrapMode: Text.WrapAnywhere
                                readOnly: true; selectByMouse: true
                            }
                        }
                    }
                }

                // ── analisi del modello ────────────────────────────
                Component {
                    id: analysisCard
                    Rectangle {
                        width: parent ? parent.width : 0
                        height: anaText.implicitHeight + 20
                        radius: 16
                        color: Qt.rgba(1, 1, 1, 0.055)
                        border.width: 1
                        border.color: Qt.rgba(1, 1, 1, 0.09)
                        TextEdit {
                            id: anaText
                            anchors.fill: parent; anchors.margins: 10
                            text: d.text
                            textFormat: TextEdit.MarkdownText
                            color: "#e6edf3"; font.pixelSize: 13
                            wrapMode: Text.Wrap
                            readOnly: true; selectByMouse: true
                        }
                    }
                }
            }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: Qt.rgba(1, 1, 1, 0.08) }

        // ── input ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 68
            color: "transparent"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12

                TextField {
                    id: termInput
                    Layout.fillWidth: true
                    placeholderText: "Cosa devo fare? (es. \"mostra i processi che usano più RAM\")"
                    placeholderTextColor: "#8b949e"
                    color: "#e6edf3"
                    font.pixelSize: 13
                    background: Rectangle {
                        color: Qt.rgba(0, 0, 0, 0.28); radius: 20
                        border.color: termInput.activeFocus ? Qt.rgba(0.35, 0.72, 1.0, 0.55)
                                                            : Qt.rgba(1, 1, 1, 0.10)
                        Behavior on border.color { ColorAnimation { duration: 140 } }
                    }
                    onAccepted: page.submit()
                }

                RoundButton {
                    text: "➤"
                    font.pixelSize: 15
                    implicitWidth: 40; implicitHeight: 40
                    enabled: termInput.text.trim().length > 0
                    background: Rectangle {
                        color: parent.enabled ? "#5ab7ff" : Qt.rgba(1,1,1,0.07); radius: 20
                    }
                    contentItem: Text { text: parent.text
                                        color: parent.enabled ? "#0d1117" : "#8b949e"
                                        font: parent.font
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter }
                    onClicked: page.submit()
                }
            }
        }
    }
}
