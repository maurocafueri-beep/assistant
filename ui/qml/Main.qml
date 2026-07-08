// ui/qml/Main.qml — finestra principale della UI nativa Qt Quick (v2).
// Parità completa con la vecchia UI web: chat vocale in streaming, modalità
// terminale agentica, upload file (drag&drop + dialog), pannello errori,
// sessioni, selettori modello/profilo/voce, TTS, latenze. Tema scuro custom,
// transizioni fluide. Parla col Python solo tramite `backend`.

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    visible: true
    width: 1100
    height: 740
    minimumWidth: 760
    minimumHeight: 500
    title: "Local Assistant"
    color: theme.bg

    // ── tema ────────────────────────────────────────────────────────
    QtObject {
        id: theme
        readonly property color bg:        "#0d1117"
        readonly property color surface:   "#161b22"
        readonly property color surface2:  "#1f2630"
        readonly property color border:    "#2d333b"
        readonly property color text:      "#e6edf3"
        readonly property color textDim:   "#8b949e"
        readonly property color accent:    "#4dabf7"
        readonly property color accent2:   "#9775fa"
        readonly property color userBubble:"#1c3a5e"
        readonly property color danger:    "#f03e3e"
        readonly property color ok:        "#40c057"
        readonly property int   radius:    12
    }

    // ── stato applicativo ───────────────────────────────────────────
    QtObject {
        id: appState
        property string mode: "chat"              // chat | terminal
        property string voiceState: "loading"
        property string model: ""
        property string personality: ""
        property string voice: ""
        property string sessionId: ""
        property bool   ttsEnabled: true
        property bool   streaming: false
        property var    latency: ({})
        property var    sessions: []
        property var    personalities: []
        property var    voices: []
        property var    models: []
        property var    attachments: []           // path caricati, da inviare col messaggio
        property var    terminalModels: []
        property string terminalModel: ""
    }

    ListModel { id: chatModel }      // {role, text}
    ListModel { id: errorModel }     // {time, source, message}
    ListModel { id: terminalFeed }   // {kind, data(json)}

    function nowTime() {
        return Qt.formatTime(new Date(), "HH:mm:ss")
    }

    function pushError(source, message) {
        errorModel.insert(0, { time: nowTime(), source: source, message: message })
        if (errorModel.count > 50) errorModel.remove(50, errorModel.count - 50)
    }

    function scrollChatToEnd() { chatList.positionViewAtEnd() }

    function appendChunk(text) {
        if (!appState.streaming) {
            chatModel.append({ role: "assistant", text: text })
            appState.streaming = true
        } else {
            var last = chatModel.count - 1
            chatModel.setProperty(last, "text", chatModel.get(last).text + text)
        }
        scrollChatToEnd()
    }

    function loadMessages(msgs) {
        chatModel.clear()
        appState.streaming = false
        if (!msgs) return
        for (var i = 0; i < msgs.length; i++)
            chatModel.append({ role: msgs[i].role, text: msgs[i].text })
        scrollChatToEnd()
    }

    function sendCurrentInput() {
        var text = input.text.trim()
        var paths = appState.attachments.map(a => a.path).join("\n")
        if (paths.length > 0)
            text = (text.length > 0 ? text + "\n" : "") + paths
        if (text.length === 0) return
        backend.sendText(text)
        input.text = ""
        appState.attachments = []
    }

    // ── segnali dal backend ─────────────────────────────────────────
    Connections {
        target: backend

        function onInitReady(p) {
            appState.voiceState  = p.state || "idle"
            appState.model       = p.model || ""
            appState.personality = p.personality || ""
            appState.voice       = p.voice || ""
            appState.sessionId   = p.session || ""
            appState.ttsEnabled  = p.tts_enabled === undefined ? true : p.tts_enabled
            appState.sessions      = p.sessions || []
            appState.personalities = p.personalities || []
            appState.voices        = p.voices || []
            loadMessages(p.active_messages)
            backend.requestModels()
            backend.requestTerminalState()
        }
        function onStateChanged(s)        { appState.voiceState = s }
        function onChunkReceived(t)       { appendChunk(t) }
        function onUserMessage(t) {
            chatModel.append({ role: "user", text: t })
            appState.streaming = false
            scrollChatToEnd()
        }
        function onStatsChanged(p) {
            appState.streaming = false
            if (p.latency) appState.latency = p.latency
        }
        function onSessionsChanged(list)  { appState.sessions = list }
        function onSessionSwitched(p) {
            appState.sessionId = p.session || ""
            appState.attachments = []
            loadMessages(p.messages)
        }
        function onModelChanged(name)     { appState.model = name }
        function onPersonalityChanged(name, display) { appState.personality = name }
        function onVoiceChanged(name)     { appState.voice = name }
        function onTtsChanged(en)         { appState.ttsEnabled = en }
        function onModeChanged(m)         { appState.mode = m }
        function onModelsListed(list)     { appState.models = list }
        function onErrorOccurred(p)       { pushError(p.source || "app", p.message || "") }
        function onBackendError(msg)      { pushError("backend", msg) }
        function onUploadFinished(p) {
            if (p.ok) {
                var atts = appState.attachments.slice()
                atts.push({ name: p.name, path: p.path })
                appState.attachments = atts
            } else {
                pushError("upload", p.error || "upload fallito")
                errorPanel.open = true
            }
        }
        function onTerminalEvent(p) {
            if (p.type === "terminal.state") {
                appState.terminalModels = p.models || []
                appState.terminalModel  = p.current_model || ""
            } else if (p.type === "terminal.model") {
                appState.terminalModel = p.name || ""
            }
            terminalPage.handleEvent(p)
        }
    }

    // ── drag & drop file (in chat) ──────────────────────────────────
    DropArea {
        anchors.fill: parent
        enabled: appState.mode === "chat"
        onDropped: (drop) => {
            if (drop.hasUrls)
                for (var i = 0; i < drop.urls.length; i++)
                    backend.uploadFile(drop.urls[i].toString())
        }
        Rectangle {
            anchors.fill: parent
            visible: parent.containsDrag
            color: Qt.rgba(0.30, 0.67, 0.97, 0.08)
            border.color: theme.accent
            border.width: 2
            radius: 8
            z: 100
            Text {
                anchors.centerIn: parent
                text: "Rilascia per allegare"
                color: theme.accent
                font.pixelSize: 20
                font.bold: true
            }
        }
    }

    FileDialog {
        id: fileDialog
        title: "Allega file"
        fileMode: FileDialog.OpenFiles
        onAccepted: {
            for (var i = 0; i < selectedFiles.length; i++)
                backend.uploadFile(selectedFiles[i].toString())
        }
    }

    // ── layout ──────────────────────────────────────────────────────
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── colonna sessioni ────────────────────────────────────────
        Rectangle {
            id: sidebar
            Layout.fillHeight: true
            Layout.preferredWidth: sidebarOpen && appState.mode === "chat" ? 232 : 0
            property bool sidebarOpen: true
            color: theme.surface
            clip: true
            Behavior on Layout.preferredWidth {
                NumberAnimation { duration: 200; easing.type: Easing.OutCubic }
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 8

                Button {
                    Layout.fillWidth: true
                    text: "+  Nuova chat"
                    font.pixelSize: 13
                    background: Rectangle {
                        color: parent.hovered ? theme.surface2 : "transparent"
                        border.color: theme.border
                        radius: theme.radius
                        Behavior on color { ColorAnimation { duration: 120 } }
                    }
                    contentItem: Text {
                        text: parent.text; color: theme.text; font: parent.font
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: backend.newSession()
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: appState.sessions
                    spacing: 3
                    clip: true

                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 40
                        radius: theme.radius - 4
                        color: modelData.id === appState.sessionId
                               ? theme.surface2
                               : (sessionArea.containsMouse ? Qt.darker(theme.surface2, 1.25) : "transparent")
                        Behavior on color { ColorAnimation { duration: 100 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 10
                            anchors.rightMargin: 6
                            Text {
                                Layout.fillWidth: true
                                text: modelData.name || modelData.id
                                color: modelData.id === appState.sessionId ? theme.text : theme.textDim
                                font.pixelSize: 13
                                elide: Text.ElideRight
                            }
                            Text {
                                visible: sessionArea.containsMouse && appState.sessions.length > 1
                                text: "✕"
                                color: theme.textDim
                                font.pixelSize: 12
                                MouseArea {
                                    anchors.fill: parent
                                    anchors.margins: -6
                                    onClicked: backend.deleteSession(modelData.id)
                                }
                            }
                        }
                        MouseArea {
                            id: sessionArea
                            anchors.fill: parent
                            hoverEnabled: true
                            z: -1
                            onClicked: backend.switchSession(modelData.id)
                        }
                    }
                }
            }
        }

        // ── colonna principale ──────────────────────────────────────
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // ── barra superiore ─────────────────────────────────────
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 52
                color: theme.surface

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 12
                    spacing: 10

                    ToolButton {
                        visible: appState.mode === "chat"
                        text: "☰"
                        font.pixelSize: 16
                        contentItem: Text { text: parent.text; color: theme.textDim; font: parent.font
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter }
                        background: Rectangle { color: parent.hovered ? theme.surface2 : "transparent"; radius: 8 }
                        onClicked: sidebar.sidebarOpen = !sidebar.sidebarOpen
                    }

                    // Selettore modalità chat/terminale
                    Rectangle {
                        Layout.preferredWidth: 108
                        Layout.preferredHeight: 30
                        radius: 15
                        color: theme.bg
                        border.color: theme.border
                        Rectangle {
                            id: modeThumb
                            width: 52; height: 26; radius: 13
                            y: 2
                            x: appState.mode === "chat" ? 2 : parent.width - width - 2
                            color: theme.surface2
                            border.color: theme.accent
                            Behavior on x { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }
                        }
                        Row {
                            anchors.fill: parent
                            Item {
                                width: parent.width / 2; height: parent.height
                                Text { anchors.centerIn: parent; text: "💬"; font.pixelSize: 13
                                       opacity: appState.mode === "chat" ? 1 : 0.45 }
                                MouseArea { anchors.fill: parent; onClicked: backend.setMode("chat") }
                            }
                            Item {
                                width: parent.width / 2; height: parent.height
                                Text { anchors.centerIn: parent; text: "⌨"; font.pixelSize: 13
                                       opacity: appState.mode === "terminal" ? 1 : 0.45 }
                                MouseArea { anchors.fill: parent; onClicked: backend.setMode("terminal") }
                            }
                        }
                    }

                    ThemedCombo {
                        comboModel: appState.mode === "terminal" ? appState.terminalModels
                                                                 : appState.models
                        current: appState.mode === "terminal" ? appState.terminalModel
                                                              : appState.model
                        label: appState.mode === "terminal" ? "modello terminale" : "modello"
                        Layout.preferredWidth: 280
                        onPicked: (name) => appState.mode === "terminal"
                                            ? backend.terminalSwitchModel(name)
                                            : backend.switchModel(name)
                    }
                    ThemedCombo {
                        visible: appState.mode === "chat"
                        comboModel: appState.personalities.map(p => p.name)
                        current: appState.personality
                        label: "profilo"
                        Layout.preferredWidth: 120
                        onPicked: (name) => backend.switchPersonality(name)
                    }
                    ThemedCombo {
                        visible: appState.mode === "chat"
                        comboModel: appState.voices
                        current: appState.voice
                        label: "voce"
                        Layout.preferredWidth: 140
                        onPicked: (name) => backend.switchVoice(name)
                    }

                    Item { Layout.fillWidth: true }

                    // Latenze ultimo turno
                    Row {
                        spacing: 6
                        visible: appState.latency.total_ms !== undefined
                        Repeater {
                            model: [
                                { k: "stt", v: appState.latency.stt_ms },
                                { k: "llm", v: appState.latency.llm_ms },
                                { k: "tts", v: appState.latency.tts_ms },
                            ]
                            delegate: Rectangle {
                                visible: modelData.v !== undefined && modelData.v !== null && modelData.v > 0
                                width: chipText.width + 14; height: 20; radius: 10
                                color: theme.surface2
                                Text {
                                    id: chipText
                                    anchors.centerIn: parent
                                    text: modelData.k + " " + Math.round(modelData.v) + "ms"
                                    color: theme.textDim
                                    font.pixelSize: 10
                                }
                            }
                        }
                    }

                    // Badge errori → apre il pannello
                    ToolButton {
                        id: errBtn
                        text: errorModel.count > 0 ? "⚠ " + errorModel.count : "⚠"
                        font.pixelSize: 12
                        contentItem: Text {
                            text: errBtn.text
                            color: errorModel.count > 0 ? "#fab005" : theme.textDim
                            font: errBtn.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            color: errBtn.hovered ? theme.surface2 : "transparent"; radius: 8
                        }
                        onClicked: errorPanel.open = !errorPanel.open
                    }

                    // Toggle TTS
                    Switch {
                        id: ttsSwitch
                        checked: appState.ttsEnabled
                        onToggled: backend.setTtsEnabled(checked)
                        indicator: Rectangle {
                            implicitWidth: 40; implicitHeight: 20; radius: 10
                            color: ttsSwitch.checked ? theme.accent : theme.surface2
                            border.color: theme.border
                            Behavior on color { ColorAnimation { duration: 120 } }
                            Rectangle {
                                x: ttsSwitch.checked ? parent.width - width - 2 : 2
                                anchors.verticalCenter: parent.verticalCenter
                                width: 16; height: 16; radius: 8
                                color: theme.text
                                Behavior on x { NumberAnimation { duration: 120 } }
                            }
                        }
                        contentItem: Text {
                            text: "🔊"
                            font.pixelSize: 13
                            color: ttsSwitch.checked ? theme.text : theme.textDim
                            leftPadding: ttsSwitch.indicator.width + 6
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: theme.border }

            // ── pagine ──────────────────────────────────────────────
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: appState.mode === "terminal" ? 1 : 0

                // ── pagina chat ─────────────────────────────────────
                ColumnLayout {
                    spacing: 0

                    ListView {
                        id: chatList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        model: chatModel
                        spacing: 14
                        clip: true
                        topMargin: 18; bottomMargin: 18; leftMargin: 24; rightMargin: 24
                        boundsBehavior: Flickable.StopAtBounds

                        add: Transition {
                            NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 220 }
                            NumberAnimation { property: "scale"; from: 0.96; to: 1; duration: 220
                                              easing.type: Easing.OutCubic }
                        }

                        ScrollBar.vertical: ScrollBar {
                            contentItem: Rectangle { implicitWidth: 5; radius: 3; color: theme.border }
                        }

                        delegate: Item {
                            width: chatList.width - chatList.leftMargin - chatList.rightMargin
                            height: bubble.height

                            Rectangle {
                                id: bubble
                                anchors.right: role === "user" ? parent.right : undefined
                                anchors.left:  role === "user" ? undefined : parent.left
                                width: Math.min(msgText.implicitWidth + 30, parent.width * 0.78)
                                height: msgText.implicitHeight + 22
                                radius: theme.radius + 2
                                color: role === "user" ? theme.userBubble : theme.surface

                                TextEdit {
                                    id: msgText
                                    anchors.fill: parent
                                    anchors.margins: 11
                                    text: model.text
                                    textFormat: TextEdit.MarkdownText
                                    color: theme.text
                                    font.pixelSize: 14
                                    wrapMode: Text.Wrap
                                    readOnly: true
                                    selectByMouse: true
                                    selectionColor: theme.accent
                                }
                            }
                        }

                        footer: Item {
                            width: 1
                            height: typing.visible ? 34 : 0
                            TypingDots {
                                id: typing
                                visible: appState.voiceState === "thinking" && !appState.streaming
                                anchors.left: parent.left
                                anchors.verticalCenter: parent.verticalCenter
                                dotColor: theme.accent2
                            }
                        }
                    }

                    Rectangle { Layout.fillWidth: true; height: 1; color: theme.border }

                    // Chip allegati in attesa di invio
                    Flow {
                        Layout.fillWidth: true
                        Layout.leftMargin: 16
                        Layout.rightMargin: 16
                        Layout.topMargin: appState.attachments.length > 0 ? 8 : 0
                        spacing: 6
                        visible: appState.attachments.length > 0

                        Repeater {
                            model: appState.attachments
                            delegate: Rectangle {
                                width: attRow.width + 18; height: 26; radius: 13
                                color: theme.surface2
                                border.color: theme.accent
                                Row {
                                    id: attRow
                                    anchors.centerIn: parent
                                    spacing: 6
                                    Text { text: "📄"; font.pixelSize: 11
                                           anchors.verticalCenter: parent.verticalCenter }
                                    Text { text: modelData.name; color: theme.text; font.pixelSize: 11
                                           anchors.verticalCenter: parent.verticalCenter }
                                    Text {
                                        text: "✕"; color: theme.textDim; font.pixelSize: 11
                                        anchors.verticalCenter: parent.verticalCenter
                                        MouseArea {
                                            anchors.fill: parent; anchors.margins: -5
                                            onClicked: {
                                                var atts = appState.attachments.slice()
                                                atts.splice(index, 1)
                                                appState.attachments = atts
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ── input chat ──────────────────────────────────
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 76
                        color: theme.surface

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 12

                            StateOrb {
                                voiceState: appState.voiceState
                                onPressedChanged: pressed ? backend.pttDown() : backend.pttUp()
                            }

                            ToolButton {
                                text: "📎"
                                font.pixelSize: 15
                                contentItem: Text { text: parent.text; font: parent.font
                                                    color: theme.textDim
                                                    horizontalAlignment: Text.AlignHCenter
                                                    verticalAlignment: Text.AlignVCenter }
                                background: Rectangle {
                                    color: parent.hovered ? theme.surface2 : "transparent"; radius: 8
                                }
                                onClicked: fileDialog.open()
                            }

                            TextField {
                                id: input
                                Layout.fillWidth: true
                                placeholderText: appState.voiceState === "loading"
                                                 ? "Caricamento assistente…"
                                                 : "Scrivi, trascina un file, tieni premuto l'orb o di' la wake word…"
                                placeholderTextColor: theme.textDim
                                enabled: appState.voiceState !== "loading"
                                color: theme.text
                                font.pixelSize: 14
                                background: Rectangle {
                                    color: theme.bg
                                    radius: theme.radius
                                    border.color: input.activeFocus ? theme.accent : theme.border
                                    Behavior on border.color { ColorAnimation { duration: 120 } }
                                }
                                onAccepted: sendCurrentInput()
                            }

                            RoundButton {
                                visible: appState.voiceState === "thinking" || appState.voiceState === "speaking"
                                text: "◼"
                                font.pixelSize: 12
                                implicitWidth: 42; implicitHeight: 42
                                background: Rectangle {
                                    color: theme.danger; radius: 21
                                    opacity: parent.hovered ? 1.0 : 0.85
                                }
                                contentItem: Text { text: parent.text; color: "white"; font: parent.font
                                                    horizontalAlignment: Text.AlignHCenter
                                                    verticalAlignment: Text.AlignVCenter }
                                onClicked: backend.cancelTurn()
                            }

                            RoundButton {
                                text: "➤"
                                font.pixelSize: 15
                                implicitWidth: 42; implicitHeight: 42
                                enabled: input.text.trim().length > 0 || appState.attachments.length > 0
                                scale: enabled ? 1.0 : 0.92
                                Behavior on scale { NumberAnimation { duration: 120 } }
                                background: Rectangle {
                                    color: parent.enabled ? theme.accent : theme.surface2
                                    radius: 21
                                    Behavior on color { ColorAnimation { duration: 120 } }
                                }
                                contentItem: Text { text: parent.text
                                                    color: parent.enabled ? "#0d1117" : theme.textDim
                                                    font: parent.font
                                                    horizontalAlignment: Text.AlignHCenter
                                                    verticalAlignment: Text.AlignVCenter }
                                onClicked: sendCurrentInput()
                            }
                        }
                    }
                }

                // ── pagina terminale ────────────────────────────────
                TerminalPage {
                    id: terminalPage
                    feed: terminalFeed
                }
            }
        }
    }

    // ── pannello errori (overlay in basso a destra) ─────────────────
    ErrorPanel {
        id: errorPanel
        errors: errorModel
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 16
        width: Math.min(460, parent.width - 32)
        z: 50
    }
}
