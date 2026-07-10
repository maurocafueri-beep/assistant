// ui/qml/Main.qml — finestra principale della UI nativa Qt Quick (v5).
// Design chiaro pastello sul reference con TEMA SCURO commutabile (singleton
// Theme, cross-fade animato). Rail di icone, sessioni a card Oggi/Ieri con
// anteprima, saluto grande, chips strumenti, input a pillola. Upload via
// pulsante 📎, chip File e drag&drop sull'intera finestra. Parla solo con
// `backend`.

import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts
import "."

ApplicationWindow {
    id: root
    visible: true
    width: 1180
    height: 760
    minimumWidth: 860
    minimumHeight: 560
    title: "Local Assistant"
    color: "transparent"

    Component.onCompleted: Theme.dark = backend.uiSetting("ui_theme") === "dark"

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

    function sessionGroups() {
        var groups = [{ label: "Oggi", items: [] },
                      { label: "Ieri", items: [] },
                      { label: "Precedenti", items: [] }]
        var now = new Date()
        var today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000
        var yesterday = today - 86400
        for (var i = 0; i < appState.sessions.length; i++) {
            var s = appState.sessions[i]
            var t = s.last_active || 0
            if (t >= today)          groups[0].items.push(s)
            else if (t >= yesterday) groups[1].items.push(s)
            else                     groups[2].items.push(s)
        }
        return groups.filter(g => g.items.length > 0)
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
            // il bridge emette "id" (protocollo storico della UI web)
            appState.sessionId = p.id || p.session || ""
            appState.attachments = []
            if (p.sessions) appState.sessions = p.sessions
            loadMessages(p.messages)
        }
        function onModelChanged(name)     { appState.model = name }
        function onPersonalityChanged(name, display) { appState.personality = name }
        function onVoiceChanged(name)     { appState.voice = name }
        function onTtsChanged(en)         { appState.ttsEnabled = en }
        function onModeChanged(m)         { appState.mode = m; pageFade.restart() }
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

    // ── sfondo: gradiente pastello + aloni (tema-reattivo) ──────────
    Rectangle {
        anchors.fill: parent
        opacity: 0.96
        gradient: Gradient {
            GradientStop { position: 0.0;  color: Theme.bg0 }
            GradientStop { position: 0.45; color: Theme.bg1 }
            GradientStop { position: 1.0;  color: Theme.bg2 }
        }
    }
    Item {
        anchors.fill: parent
        layer.enabled: true
        layer.effect: MultiEffect { blurEnabled: true; blur: 1.0; blurMax: 64 }
        opacity: 0.5
        Rectangle { x: -root.width*0.1; y: -root.height*0.2
                    width: root.width*0.5; height: width; radius: width/2; color: Theme.blob1 }
        Rectangle { x: root.width*0.65; y: root.height*0.55
                    width: root.width*0.5; height: width; radius: width/2; color: Theme.blob2 }
    }

    // ── layout ──────────────────────────────────────────────────────
    RowLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 16

        // ── rail icone ──────────────────────────────────────────────
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 52
            spacing: 12

            component RailButton: AbstractButton {
                id: rb
                property string glyph: ""
                property bool   active: false
                property color  glyphColor: Theme.textDim
                property bool   emoji: false
                Layout.preferredWidth: 44
                Layout.preferredHeight: 44
                scale: pressed ? 0.90 : (hovered ? 1.08 : 1.0)
                Behavior on scale { SpringAnimation { spring: 3.5; damping: 0.28 } }
                background: Rectangle {
                    radius: 22
                    color: rb.active ? Theme.accent
                                     : (rb.hovered ? Theme.cardHover : Theme.card)
                    border.color: Theme.cardLine
                    Behavior on color { ColorAnimation { duration: 140 } }
                }
                contentItem: Text {
                    text: rb.glyph
                    color: rb.active ? "white" : rb.glyphColor
                    font.pixelSize: rb.emoji ? 15 : 17
                    font.bold: !rb.emoji
                    font.family: rb.emoji ? "Noto Color Emoji" : Qt.application.font.family
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            Item { Layout.preferredHeight: 4 }

            RailButton {
                glyph: "＋"
                onClicked: backend.newSession()
                ToolTip.visible: hovered; ToolTip.text: "Nuova chat"
            }
            RailButton {
                glyph: "💬"; emoji: true
                active: appState.mode === "chat"
                onClicked: backend.setMode("chat")
                ToolTip.visible: hovered; ToolTip.text: "Chat"
            }
            RailButton {
                glyph: ">_"
                active: appState.mode === "terminal"
                onClicked: backend.setMode("terminal")
                ToolTip.visible: hovered; ToolTip.text: "Terminale"
            }

            Item { Layout.fillHeight: true }

            RailButton {
                glyph: Theme.dark ? "☀" : "☾"
                onClicked: {
                    Theme.dark = !Theme.dark
                    backend.saveUiSetting("ui_theme", Theme.dark ? "dark" : "light")
                }
                ToolTip.visible: hovered
                ToolTip.text: Theme.dark ? "Tema chiaro" : "Tema scuro"
            }
            RailButton {
                glyph: errorModel.count > 0 ? String(errorModel.count) : "✓"
                glyphColor: errorModel.count > 0 ? Theme.warn : Theme.textDim
                onClicked: errorPanel.open = !errorPanel.open
                ToolTip.visible: hovered
                ToolTip.text: errorModel.count > 0
                              ? errorModel.count + " errori — clic per aprire"
                              : "Nessun errore"
            }
            RailButton {
                glyph: "🔊"; emoji: true
                active: appState.ttsEnabled
                onClicked: backend.setTtsEnabled(!appState.ttsEnabled)
                ToolTip.visible: hovered
                ToolTip.text: appState.ttsEnabled ? "Voce attiva" : "Voce disattivata"
            }

            Rectangle {
                Layout.preferredWidth: 44
                Layout.preferredHeight: 44
                radius: 22
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#8fb8fe" }
                    GradientStop { position: 1.0; color: "#c39efb" }
                }
                border.color: Theme.cardLine
                Text {
                    anchors.centerIn: parent
                    text: (appState.personality || "?").charAt(0).toUpperCase()
                    color: "white"
                    font.pixelSize: 17
                    font.bold: true
                }
                MouseArea { id: avatarArea; anchors.fill: parent; hoverEnabled: true }
                ToolTip.visible: avatarArea.containsMouse
                ToolTip.text: "Profilo: " + appState.personality + " · Voce: " + appState.voice
            }
        }

        // ── pannello sessioni ───────────────────────────────────────
        Glass {
            Layout.fillHeight: true
            Layout.preferredWidth: appState.mode === "chat" ? 268 : 0
            glassRadius: Theme.radius
            fill: Theme.cardSoft
            clip: true
            visible: Layout.preferredWidth > 0
            Behavior on Layout.preferredWidth {
                NumberAnimation { duration: 240; easing.type: Easing.OutCubic }
            }

            Flickable {
                anchors.fill: parent
                anchors.margins: 16
                contentHeight: sessCol.height
                clip: true

                Column {
                    id: sessCol
                    width: parent.width
                    spacing: 10

                    Text {
                        text: "Chat"
                        color: Theme.text
                        font.pixelSize: 21
                        font.bold: true
                        bottomPadding: 4
                    }

                    Repeater {
                        model: sessionGroups()
                        delegate: Column {
                            width: sessCol.width
                            spacing: 8
                            Text {
                                text: modelData.label
                                color: Theme.textDim
                                font.pixelSize: 12
                                font.bold: true
                                topPadding: 6
                            }
                            Repeater {
                                model: modelData.items
                                delegate: Rectangle {
                                    width: sessCol.width
                                    height: 62
                                    radius: Theme.radiusIn
                                    color: modelData.id === appState.sessionId
                                           ? Theme.cardStrong
                                           : (cardArea.containsMouse ? Theme.cardHover : Theme.card)
                                    border.color: modelData.id === appState.sessionId
                                                  ? Theme.accentLine : Theme.cardLine
                                    Behavior on color { ColorAnimation { duration: 120 } }

                                    Column {
                                        anchors.left: parent.left
                                        anchors.right: parent.right
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.leftMargin: 14
                                        anchors.rightMargin: 26
                                        spacing: 3
                                        Text {
                                            width: parent.width
                                            text: modelData.name || modelData.id
                                            color: Theme.text
                                            font.pixelSize: 13
                                            font.bold: true
                                            elide: Text.ElideRight
                                        }
                                        Text {
                                            width: parent.width
                                            text: modelData.preview || "chat vuota"
                                            color: Theme.textDim
                                            font.pixelSize: 11
                                            elide: Text.ElideRight
                                        }
                                    }
                                    Text {
                                        visible: cardArea.containsMouse && appState.sessions.length > 1
                                        anchors.right: parent.right
                                        anchors.top: parent.top
                                        anchors.margins: 8
                                        text: "✕"
                                        color: Theme.textDim
                                        font.pixelSize: 11
                                        MouseArea {
                                            anchors.fill: parent
                                            anchors.margins: -6
                                            onClicked: backend.deleteSession(modelData.id)
                                        }
                                    }
                                    MouseArea {
                                        id: cardArea
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        z: -1
                                        onClicked: backend.switchSession(modelData.id)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── pannello principale ─────────────────────────────────────
        Glass {
            Layout.fillWidth: true
            Layout.fillHeight: true
            glassRadius: Theme.radius
            fill: Theme.cardSoft
            clip: true

            StackLayout {
                id: pages
                anchors.fill: parent
                currentIndex: appState.mode === "terminal" ? 1 : 0

                // dissolvenza al cambio pagina
                NumberAnimation {
                    id: pageFade
                    target: pages
                    property: "opacity"
                    from: 0.25; to: 1.0
                    duration: 260
                    easing.type: Easing.OutCubic
                }

                // ── pagina chat ─────────────────────────────────────
                ColumnLayout {
                    spacing: 0

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 54
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        spacing: 8

                        Text {
                            text: appState.streaming || appState.voiceState === "thinking"
                                  ? "Sto pensando…" : ""
                            color: Theme.textDim
                            font.pixelSize: 12
                        }
                        Item { Layout.fillWidth: true }

                        Rectangle {
                            visible: appState.latency.total_ms !== undefined
                            width: latText.width + 18; height: 24; radius: 12
                            color: Theme.card
                            border.color: Theme.cardLine
                            Text {
                                id: latText
                                anchors.centerIn: parent
                                text: "⏱ " + (appState.latency.total_ms / 1000).toFixed(1) + "s"
                                color: Theme.textDim
                                font.pixelSize: 11
                            }
                            MouseArea { id: latArea; anchors.fill: parent; hoverEnabled: true }
                            ToolTip.visible: latArea.containsMouse
                            ToolTip.text: "stt " + Math.round(appState.latency.stt_ms || 0)
                                          + "ms · llm " + Math.round(appState.latency.llm_ms || 0)
                                          + "ms · tts " + Math.round(appState.latency.tts_ms || 0) + "ms"
                        }

                        ThemedCombo {
                            comboModel: appState.models
                            current: appState.model
                            label: "modello"
                            Layout.preferredWidth: 210
                            onPicked: (name) => backend.switchModel(name)
                        }
                        ThemedCombo {
                            comboModel: appState.personalities.map(p => p.name)
                            current: appState.personality
                            label: "profilo"
                            Layout.preferredWidth: 100
                            onPicked: (name) => backend.switchPersonality(name)
                        }
                        ThemedCombo {
                            comboModel: appState.voices
                            current: appState.voice
                            label: "voce"
                            Layout.preferredWidth: 118
                            onPicked: (name) => backend.switchVoice(name)
                        }
                    }

                    // saluto (con ingresso animato)
                    Column {
                        id: greeting
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 40
                        visible: chatModel.count === 0 && appState.voiceState !== "loading"
                        spacing: 6
                        opacity: visible ? 1 : 0
                        scale: visible ? 1 : 0.95
                        Behavior on opacity { NumberAnimation { duration: 400; easing.type: Easing.OutCubic } }
                        Behavior on scale   { NumberAnimation { duration: 400; easing.type: Easing.OutBack } }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "Ciao, Mauro!"
                            color: Theme.textDim
                            font.pixelSize: 22
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "Come posso aiutarti?"
                            color: Theme.text
                            font.pixelSize: 28
                            font.bold: true
                        }
                    }

                    ListView {
                        id: chatList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        model: chatModel
                        spacing: 14
                        clip: true
                        topMargin: 10; bottomMargin: 16; leftMargin: 24; rightMargin: 24
                        boundsBehavior: Flickable.StopAtBounds

                        add: Transition {
                            NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 220 }
                            NumberAnimation { property: "y"; from: chatList.height; duration: 260
                                              easing.type: Easing.OutCubic }
                        }
                        displaced: Transition {
                            NumberAnimation { property: "y"; duration: 200
                                              easing.type: Easing.OutCubic }
                        }

                        ScrollBar.vertical: ScrollBar {
                            contentItem: Rectangle { implicitWidth: 5; radius: 3
                                                     color: Theme.scrollBar }
                        }

                        delegate: Item {
                            width: chatList.width - chatList.leftMargin - chatList.rightMargin
                            height: bubble.height

                            Rectangle {
                                id: bubble
                                anchors.right: role === "user" ? parent.right : undefined
                                anchors.left:  role === "user" ? undefined : parent.left
                                width: Math.min(msgText.implicitWidth + 34, parent.width * 0.80)
                                height: msgText.implicitHeight + 26
                                radius: Theme.radiusIn
                                color: role === "user" ? Theme.userFill : Theme.mintFill
                                border.width: 1
                                border.color: role === "user" ? Theme.userLine : Theme.mintLine

                                TextEdit {
                                    id: msgText
                                    anchors.fill: parent
                                    anchors.margins: 13
                                    text: model.text
                                    textFormat: TextEdit.MarkdownText
                                    color: Theme.text
                                    font.pixelSize: 14
                                    wrapMode: Text.Wrap
                                    readOnly: true
                                    selectByMouse: true
                                    selectionColor: Theme.accent
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
                                dotColor: Theme.accent
                            }
                        }
                    }

                    Flow {
                        Layout.fillWidth: true
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        spacing: 6
                        visible: appState.attachments.length > 0

                        Repeater {
                            model: appState.attachments
                            delegate: Rectangle {
                                width: attRow.width + 20; height: 28; radius: 14
                                color: Theme.cardStrong
                                border.color: Theme.accentLine
                                Row {
                                    id: attRow
                                    anchors.centerIn: parent
                                    spacing: 6
                                    Text { text: "📄"; font.pixelSize: 11; font.family: "Noto Color Emoji"
                                           anchors.verticalCenter: parent.verticalCenter }
                                    Text { text: modelData.name; color: Theme.text; font.pixelSize: 11
                                           anchors.verticalCenter: parent.verticalCenter }
                                    Text {
                                        text: "✕"; color: Theme.textDim; font.pixelSize: 11
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

                    // chips strumenti
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        Layout.topMargin: 8
                        spacing: 10

                        component ToolChip: AbstractButton {
                            id: chip
                            property string emoji: ""
                            property string label: ""
                            Layout.fillWidth: true
                            Layout.preferredHeight: 58
                            scale: pressed ? 0.97 : (hovered ? 1.02 : 1.0)
                            Behavior on scale { SpringAnimation { spring: 3.8; damping: 0.28 } }
                            background: Rectangle {
                                radius: Theme.radiusIn
                                color: chip.hovered ? Theme.cardStrong : Theme.card
                                border.color: Theme.cardLine
                                Behavior on color { ColorAnimation { duration: 140 } }
                            }
                            contentItem: Row {
                                spacing: 8
                                leftPadding: 14
                                Text {
                                    text: chip.emoji
                                    font.pixelSize: 18
                                    font.family: "Noto Color Emoji"
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    text: chip.label
                                    color: Theme.text
                                    font.pixelSize: 13
                                    font.bold: true
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                        }

                        ToolChip {
                            emoji: "📂"; label: "File"
                            onClicked: backend.pickFiles()
                        }
                        ToolChip {
                            emoji: "🖥️"; label: "Schermo"
                            onClicked: {
                                input.text = "Guarda lo schermo e "
                                input.forceActiveFocus()
                                input.cursorPosition = input.text.length
                            }
                        }
                        ToolChip {
                            emoji: "🌐"; label: "Web"
                            onClicked: {
                                input.text = "Cerca online "
                                input.forceActiveFocus()
                                input.cursorPosition = input.text.length
                            }
                        }
                        ToolChip {
                            emoji: "⚡"; label: "Terminale"
                            onClicked: backend.setMode("terminal")
                        }
                    }

                    // input a pillola
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.margins: 20
                        Layout.topMargin: 12
                        spacing: 10

                        StateOrb {
                            voiceState: appState.voiceState
                            onPressedChanged: pressed ? backend.pttDown() : backend.pttUp()
                        }

                        RoundButton {
                            text: "📎"
                            font.pixelSize: 14
                            font.family: "Noto Color Emoji"
                            implicitWidth: 40; implicitHeight: 40
                            scale: pressed ? 0.9 : (hovered ? 1.08 : 1.0)
                            Behavior on scale { SpringAnimation { spring: 3.8; damping: 0.28 } }
                            background: Rectangle {
                                color: parent.hovered ? Theme.cardHover : Theme.card
                                border.color: Theme.cardLine
                                radius: 20
                            }
                            contentItem: Text { text: parent.text; font: parent.font
                                                horizontalAlignment: Text.AlignHCenter
                                                verticalAlignment: Text.AlignVCenter }
                            onClicked: backend.pickFiles()
                            ToolTip.visible: hovered; ToolTip.text: "Allega file (o trascinali qui)"
                        }

                        TextField {
                            id: input
                            Layout.fillWidth: true
                            Layout.preferredHeight: 46
                            placeholderText: appState.voiceState === "loading"
                                             ? "Caricamento assistente…"
                                             : "Chiedimi qualsiasi cosa…"
                            placeholderTextColor: Theme.textDim
                            enabled: appState.voiceState !== "loading"
                            color: Theme.text
                            font.pixelSize: 14
                            leftPadding: 18
                            background: Rectangle {
                                color: Theme.inputFill
                                radius: 23
                                border.color: input.activeFocus ? Theme.accentLine : Theme.cardLine
                                Behavior on border.color { ColorAnimation { duration: 140 } }
                            }
                            onAccepted: sendCurrentInput()
                        }

                        RoundButton {
                            visible: appState.voiceState === "thinking" || appState.voiceState === "speaking"
                            text: "◼"
                            font.pixelSize: 12
                            implicitWidth: 44; implicitHeight: 44
                            background: Rectangle {
                                color: Theme.danger; radius: 22
                                opacity: parent.hovered ? 1.0 : 0.85
                            }
                            contentItem: Text { text: parent.text; color: "white"; font: parent.font
                                                horizontalAlignment: Text.AlignHCenter
                                                verticalAlignment: Text.AlignVCenter }
                            onClicked: backend.cancelTurn()
                        }

                        RoundButton {
                            text: "↑"
                            font.pixelSize: 17
                            font.bold: true
                            implicitWidth: 44; implicitHeight: 44
                            enabled: input.text.trim().length > 0 || appState.attachments.length > 0
                            scale: enabled ? (hovered ? 1.08 : 1.0) : 0.9
                            Behavior on scale { SpringAnimation { spring: 3.5; damping: 0.3 } }
                            background: Rectangle {
                                color: parent.enabled ? Theme.accent : Theme.card
                                border.color: Theme.cardLine
                                radius: 22
                                Behavior on color { ColorAnimation { duration: 140 } }
                            }
                            contentItem: Text { text: parent.text
                                                color: parent.enabled ? "white" : Theme.textDim
                                                font: parent.font
                                                horizontalAlignment: Text.AlignHCenter
                                                verticalAlignment: Text.AlignVCenter }
                            onClicked: sendCurrentInput()
                        }
                    }
                }

                // ── pagina terminale ────────────────────────────────
                ColumnLayout {
                    spacing: 0

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 54
                        Layout.leftMargin: 20
                        Layout.rightMargin: 20
                        Text {
                            text: "Terminale"
                            color: Theme.text
                            font.pixelSize: 18
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        ThemedCombo {
                            comboModel: appState.terminalModels
                            current: appState.terminalModel
                            label: "modello terminale"
                            Layout.preferredWidth: 220
                            onPicked: (name) => backend.terminalSwitchModel(name)
                        }
                    }

                    TerminalPage {
                        id: terminalPage
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        feed: terminalFeed
                    }
                }
            }
        }
    }

    // ── drag & drop (sopra i pannelli, così l'overlay è visibile) ───
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
            anchors.margins: 12
            visible: parent.containsDrag
            color: Qt.rgba(0.24, 0.49, 0.97, 0.08)
            border.color: Theme.accent
            border.width: 2
            radius: Theme.radius
            Text {
                anchors.centerIn: parent
                text: "Rilascia per allegare"
                color: Theme.accent
                font.pixelSize: 20
                font.bold: true
            }
        }
    }

    // ── pannello errori (overlay) ───────────────────────────────────
    ErrorPanel {
        id: errorPanel
        errors: errorModel
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 22
        width: Math.min(460, parent.width - 44)
        z: 50
    }
}
