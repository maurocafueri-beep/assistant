// ui/qml/Main.qml — finestra principale della UI nativa Qt Quick (v3).
// Design "Liquid Glass" in stile macOS Tahoe: sfondo profondo con blob
// colorati sfocati, pannelli di vetro flottanti (Glass.qml) con bordo
// speculare e ombre morbide, raggi concentrici, animazioni elastiche.
// Funzioni: chat vocale streaming, terminale agentico, upload, pannello
// errori, sessioni, selettori, TTS, latenze. Parla solo con `backend`.

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    visible: true
    width: 1100
    height: 740
    minimumWidth: 780
    minimumHeight: 520
    title: "Local Assistant"
    // Trasparenza reale: il desktop si vede attraverso (compositor Wayland).
    color: "transparent"

    // ── tema ────────────────────────────────────────────────────────
    QtObject {
        id: theme
        readonly property color text:      "#f2f5f8"
        readonly property color textDim:   "#9aa4b2"
        readonly property color accent:    "#5ab7ff"
        readonly property color accent2:   "#a78bfa"
        readonly property color danger:    "#ff6b6b"
        readonly property color ok:        "#51cf66"
        readonly property color warn:      "#ffd43b"
        readonly property int   radius:    22     // raggio pannelli
        readonly property int   radiusIn:  16     // raggio elementi interni (concentrico)
        readonly property color glassLine: Qt.rgba(1, 1, 1, 0.10)
        readonly property color glassFill: Qt.rgba(1, 1, 1, 0.055)
    }

    // ── stato applicativo ───────────────────────────────────────────
    QtObject {
        id: appState
        property string mode: "chat"
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
        property var    attachments: []
        property var    terminalModels: []
        property string terminalModel: ""
    }

    ListModel { id: chatModel }
    ListModel { id: errorModel }
    ListModel { id: terminalFeed }

    function nowTime() { return Qt.formatTime(new Date(), "HH:mm:ss") }

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

    // ── sfondo: velo neutro traslucido (60% opaco) ──────────────────
    // Il desktop filtra attraverso; con una windowrule Hyprland di blur
    // si ottiene il vero effetto "frosted" dietro la finestra.
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0.075, 0.08, 0.09, 0.60)
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
            anchors.margins: 10
            visible: parent.containsDrag
            color: Qt.rgba(0.35, 0.72, 1.0, 0.07)
            border.color: theme.accent
            border.width: 2
            radius: theme.radius
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

    // ── layout: isole di vetro flottanti ────────────────────────────
    RowLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 12

        // ── isola sessioni ──────────────────────────────────────────
        Glass {
            id: sidebar
            Layout.fillHeight: true
            Layout.preferredWidth: sidebarOpen && appState.mode === "chat" ? 230 : 0
            property bool sidebarOpen: true
            glassRadius: theme.radius
            clip: true
            visible: Layout.preferredWidth > 0
            Behavior on Layout.preferredWidth {
                NumberAnimation { duration: 260; easing.type: Easing.OutCubic }
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10

                Button {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 38
                    text: "＋  Nuova chat"
                    font.pixelSize: 13
                    scale: pressed ? 0.97 : 1.0
                    Behavior on scale { SpringAnimation { spring: 4; damping: 0.3 } }
                    background: Rectangle {
                        color: parent.hovered ? Qt.rgba(1,1,1,0.10) : theme.glassFill
                        border.color: theme.glassLine
                        radius: theme.radiusIn
                        Behavior on color { ColorAnimation { duration: 140 } }
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
                    spacing: 4
                    clip: true

                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 40
                        radius: theme.radiusIn - 4
                        color: modelData.id === appState.sessionId
                               ? Qt.rgba(0.35, 0.72, 1.0, 0.14)
                               : (sessionArea.containsMouse ? Qt.rgba(1,1,1,0.06) : "transparent")
                        border.color: modelData.id === appState.sessionId
                                      ? Qt.rgba(0.35, 0.72, 1.0, 0.30) : "transparent"
                        Behavior on color { ColorAnimation { duration: 120 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 8
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
            spacing: 12

            // ── isola barra superiore ───────────────────────────────
            Glass {
                Layout.fillWidth: true
                Layout.preferredHeight: 58
                glassRadius: theme.radius

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10
                    anchors.rightMargin: 14
                    spacing: 10

                    ToolButton {
                        visible: appState.mode === "chat"
                        text: "☰"
                        font.pixelSize: 15
                        contentItem: Text { text: parent.text; color: theme.textDim; font: parent.font
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter }
                        background: Rectangle {
                            color: parent.hovered ? Qt.rgba(1,1,1,0.08) : "transparent"; radius: 10
                        }
                        onClicked: sidebar.sidebarOpen = !sidebar.sidebarOpen
                    }

                    // Selettore modalità (capsula con cursore elastico)
                    Rectangle {
                        Layout.preferredWidth: 110
                        Layout.preferredHeight: 32
                        radius: 16
                        color: Qt.rgba(0, 0, 0, 0.25)
                        border.color: theme.glassLine
                        Rectangle {
                            id: modeThumb
                            width: 52; height: 28; radius: 14
                            y: 2
                            x: appState.mode === "chat" ? 2 : parent.width - width - 2
                            color: Qt.rgba(1, 1, 1, 0.12)
                            border.color: Qt.rgba(0.35, 0.72, 1.0, 0.45)
                            Behavior on x { SpringAnimation { spring: 3.5; damping: 0.28 } }
                        }
                        Row {
                            anchors.fill: parent
                            Item {
                                width: parent.width / 2; height: parent.height
                                Text { anchors.centerIn: parent; text: "💬"; font.pixelSize: 13; font.family: "Noto Color Emoji"
                                       opacity: appState.mode === "chat" ? 1 : 0.4 }
                                MouseArea { anchors.fill: parent; onClicked: backend.setMode("chat") }
                            }
                            Item {
                                width: parent.width / 2; height: parent.height
                                Text { anchors.centerIn: parent; text: ">_"; font.pixelSize: 11; font.bold: true; font.family: "monospace"; color: theme.text
                                       opacity: appState.mode === "terminal" ? 1 : 0.4 }
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
                        Layout.fillWidth: true
                        Layout.maximumWidth: 238
                        Layout.minimumWidth: 130
                        onPicked: (name) => appState.mode === "terminal"
                                            ? backend.terminalSwitchModel(name)
                                            : backend.switchModel(name)
                    }
                    ThemedCombo {
                        visible: appState.mode === "chat"
                        comboModel: appState.personalities.map(p => p.name)
                        current: appState.personality
                        label: "profilo"
                        Layout.preferredWidth: 104
                        onPicked: (name) => backend.switchPersonality(name)
                    }
                    ThemedCombo {
                        visible: appState.mode === "chat"
                        comboModel: appState.voices
                        current: appState.voice
                        label: "voce"
                        Layout.preferredWidth: 120
                        onPicked: (name) => backend.switchVoice(name)
                    }

                    Item { Layout.fillWidth: true }

                    Rectangle {
                        visible: appState.latency.total_ms !== undefined
                        width: latText.width + 18; height: 24; radius: 12
                        color: Qt.rgba(0, 0, 0, 0.25)
                        border.color: theme.glassLine
                        Text {
                            id: latText
                            anchors.centerIn: parent
                            text: "⏱ " + (appState.latency.total_ms / 1000).toFixed(1) + "s"
                            color: theme.textDim
                            font.pixelSize: 11
                        }
                        MouseArea {
                            id: latArea
                            anchors.fill: parent
                            hoverEnabled: true
                        }
                        ToolTip.visible: latArea.containsMouse
                        ToolTip.text: "stt " + Math.round(appState.latency.stt_ms || 0)
                                      + "ms · llm " + Math.round(appState.latency.llm_ms || 0)
                                      + "ms · tts " + Math.round(appState.latency.tts_ms || 0) + "ms"
                    }

                    ToolButton {
                        id: errBtn
                        text: errorModel.count > 0 ? "⚠ " + errorModel.count : "⚠"
                        font.pixelSize: 12
                        contentItem: Text {
                            text: errBtn.text
                            color: errorModel.count > 0 ? theme.warn : theme.textDim
                            font: errBtn.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            color: errBtn.hovered ? Qt.rgba(1,1,1,0.08) : "transparent"; radius: 10
                        }
                        onClicked: errorPanel.open = !errorPanel.open
                    }

                    Switch {
                        id: ttsSwitch
                        checked: appState.ttsEnabled
                        onToggled: backend.setTtsEnabled(checked)
                        indicator: Rectangle {
                            implicitWidth: 42; implicitHeight: 22; radius: 11
                            color: ttsSwitch.checked ? Qt.rgba(0.35, 0.72, 1.0, 0.55)
                                                     : Qt.rgba(0, 0, 0, 0.30)
                            border.color: theme.glassLine
                            Behavior on color { ColorAnimation { duration: 140 } }
                            Rectangle {
                                x: ttsSwitch.checked ? parent.width - width - 2 : 2
                                anchors.verticalCenter: parent.verticalCenter
                                width: 18; height: 18; radius: 9
                                color: "#ffffff"
                                Behavior on x { SpringAnimation { spring: 4; damping: 0.32 } }
                            }
                        }
                        contentItem: Text {
                            text: "🔊"
                            font.pixelSize: 13
                            font.family: "Noto Color Emoji"
                            color: ttsSwitch.checked ? theme.text : theme.textDim
                            leftPadding: ttsSwitch.indicator.width + 6
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }

            // ── isola contenuto ─────────────────────────────────────
            Glass {
                Layout.fillWidth: true
                Layout.fillHeight: true
                glassRadius: theme.radius
                fillOpacity: 0.26
                clip: true

                StackLayout {
                    anchors.fill: parent
                    currentIndex: appState.mode === "terminal" ? 1 : 0

                    // ── pagina chat ─────────────────────────────────
                    ListView {
                        id: chatList
                        model: chatModel
                        spacing: 14
                        clip: true
                        topMargin: 20; bottomMargin: 20; leftMargin: 22; rightMargin: 22
                        boundsBehavior: Flickable.StopAtBounds

                        add: Transition {
                            NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 240 }
                            NumberAnimation { property: "scale"; from: 0.92; to: 1; duration: 320
                                              easing.type: Easing.OutBack; easing.overshoot: 1.2 }
                        }

                        ScrollBar.vertical: ScrollBar {
                            contentItem: Rectangle { implicitWidth: 5; radius: 3
                                                     color: Qt.rgba(1,1,1,0.15) }
                        }

                        delegate: Item {
                            width: chatList.width - chatList.leftMargin - chatList.rightMargin
                            height: bubble.height

                            Rectangle {
                                id: bubble
                                anchors.right: role === "user" ? parent.right : undefined
                                anchors.left:  role === "user" ? undefined : parent.left
                                width: Math.min(msgText.implicitWidth + 32, parent.width * 0.78)
                                height: msgText.implicitHeight + 24
                                radius: theme.radiusIn
                                color: role === "user" ? Qt.rgba(0.35, 0.72, 1.0, 0.16)
                                                       : Qt.rgba(1, 1, 1, 0.055)
                                border.width: 1
                                border.color: role === "user" ? Qt.rgba(0.35, 0.72, 1.0, 0.32)
                                                              : Qt.rgba(1, 1, 1, 0.09)

                                TextEdit {
                                    id: msgText
                                    anchors.fill: parent
                                    anchors.margins: 12
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
                            height: typing.visible ? 36 : 0
                            TypingDots {
                                id: typing
                                visible: appState.voiceState === "thinking" && !appState.streaming
                                anchors.left: parent.left
                                anchors.verticalCenter: parent.verticalCenter
                                dotColor: theme.accent2
                            }
                        }
                    }

                    // ── pagina terminale ────────────────────────────
                    TerminalPage {
                        id: terminalPage
                        feed: terminalFeed
                    }
                }
            }

            // Chip allegati (flottanti tra pannello e dock)
            Flow {
                Layout.fillWidth: true
                Layout.leftMargin: 6
                spacing: 6
                visible: appState.mode === "chat" && appState.attachments.length > 0

                Repeater {
                    model: appState.attachments
                    delegate: Rectangle {
                        width: attRow.width + 20; height: 28; radius: 14
                        color: Qt.rgba(0.35, 0.72, 1.0, 0.12)
                        border.color: Qt.rgba(0.35, 0.72, 1.0, 0.35)
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

            // ── dock input (capsula flottante, solo chat) ───────────
            Glass {
                visible: appState.mode === "chat"
                Layout.fillWidth: true
                Layout.preferredHeight: 66
                glassRadius: 33

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 12
                    spacing: 10

                    StateOrb {
                        voiceState: appState.voiceState
                        onPressedChanged: pressed ? backend.pttDown() : backend.pttUp()
                    }

                    ToolButton {
                        text: "📎"
                        font.pixelSize: 15
                        font.family: "Noto Color Emoji"
                        contentItem: Text { text: parent.text; font: parent.font
                                            color: theme.textDim
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter }
                        background: Rectangle {
                            color: parent.hovered ? Qt.rgba(1,1,1,0.08) : "transparent"; radius: 10
                        }
                        onClicked: fileDialog.open()
                    }

                    TextField {
                        id: input
                        Layout.fillWidth: true
                        Layout.preferredHeight: 42
                        placeholderText: appState.voiceState === "loading"
                                         ? "Caricamento assistente…"
                                         : "Scrivi, trascina un file, tieni premuto l'orb o di' la wake word…"
                        placeholderTextColor: theme.textDim
                        enabled: appState.voiceState !== "loading"
                        color: theme.text
                        font.pixelSize: 14
                        leftPadding: 16
                        background: Rectangle {
                            color: Qt.rgba(0, 0, 0, 0.28)
                            radius: 21
                            border.color: input.activeFocus ? Qt.rgba(0.35, 0.72, 1.0, 0.55)
                                                            : theme.glassLine
                            Behavior on border.color { ColorAnimation { duration: 140 } }
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
                        scale: enabled ? 1.0 : 0.9
                        Behavior on scale { SpringAnimation { spring: 3.5; damping: 0.3 } }
                        background: Rectangle {
                            color: parent.enabled ? theme.accent : Qt.rgba(1,1,1,0.07)
                            radius: 21
                            Behavior on color { ColorAnimation { duration: 140 } }
                        }
                        contentItem: Text { text: parent.text
                                            color: parent.enabled ? "#0a0d12" : theme.textDim
                                            font: parent.font
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter }
                        onClicked: sendCurrentInput()
                    }
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
        anchors.margins: 20
        width: Math.min(460, parent.width - 40)
        z: 50
    }
}
