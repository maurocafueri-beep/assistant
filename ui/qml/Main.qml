// ui/qml/Main.qml — finestra principale della UI nativa Qt Quick (v9).
// Riorganizzazione per CHIAREZZA: un solo header contestuale (titolo della
// pagina/sessione + stato + latenza + un pulsante Impostazioni che apre un
// popover con modello/profilo/voce), chips di suggerimento solo a chat
// vuota, input essenziale (orb · campo con 📎 integrato · invio), pannelli
// tonali MD3 dai bordi morbidi, transizioni slide+fade tra le pagine.
// Testo in stile Claude (MessageView/CodeBlock). Parla solo con `backend`.

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
        property string mode: "chat"          // modalità del bridge (chat|terminal)
        property string page: "chat"          // pagina UI (chat|terminal|system)
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

        readonly property string sessionName: {
            for (var i = 0; i < sessions.length; i++)
                if (sessions[i].id === sessionId)
                    return sessions[i].name || "Chat"
            return "Chat"
        }
        readonly property string stateLabel:
            voiceState === "loading"   ? "caricamento…"
          : voiceState === "thinking"  ? "sta pensando…"
          : voiceState === "speaking"  ? "sta parlando…"
          : voiceState === "recording" ? "in ascolto…"
          : voiceState === "warmup"    ? "riscaldamento…"
          : "pronto"
    }

    ListModel { id: chatModel }
    ListModel { id: errorModel }
    ListModel { id: terminalFeed }
    ListModel { id: latencyHistory }

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

    function goPage(p) {
        if (appState.page === p) return
        if (p === "chat" || p === "terminal") backend.setMode(p)
        appState.page = p
        pageTrans.restart()
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
            if (p.latency) {
                appState.latency = p.latency
                if (p.latency.llm_ms) {
                    latencyHistory.append({ ms: p.latency.llm_ms })
                    if (latencyHistory.count > 30) latencyHistory.remove(0)
                    systemPage.refreshSpark()
                }
            }
        }
        function onSessionsChanged(list)  { appState.sessions = list }
        function onSessionSwitched(p) {
            appState.sessionId = p.id || p.session || ""
            appState.attachments = []
            if (p.sessions) appState.sessions = p.sessions
            loadMessages(p.messages)
        }
        function onModelChanged(name)     { appState.model = name }
        function onPersonalityChanged(name, display) { appState.personality = name }
        function onVoiceChanged(name)     { appState.voice = name }
        function onTtsChanged(en)         { appState.ttsEnabled = en }
        function onModeChanged(m) {
            appState.mode = m
            if (appState.page !== "system") {
                appState.page = m
                pageTrans.restart()
            }
        }
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
        function onSystemStatus(p)        { systemPage.handleStatus(p) }
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

    // ── sfondo piatto ───────────────────────────────────────────────
    Rectangle {
        anchors.fill: parent
        color: Theme.bg0
        opacity: 0.86
    }

    // ── layout ──────────────────────────────────────────────────────
    RowLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 14

        // ── navigation rail ─────────────────────────────────────────
        ColumnLayout {
            Layout.fillHeight: true
            Layout.preferredWidth: 56
            spacing: 10

            component RailButton: AbstractButton {
                id: rb
                property string glyph: ""
                property bool   active: false
                property color  glyphColor: Theme.textDim
                property bool   emoji: false
                Layout.preferredWidth: 48
                Layout.preferredHeight: 48
                scale: pressed ? 0.94 : 1.0
                Behavior on scale { SpringAnimation { spring: 3.5; damping: 0.28 } }
                onPressed: railRip.trigger(pressX, pressY)
                background: Rectangle {
                    radius: 24
                    color: rb.active ? Theme.accentSoft
                                     : (rb.hovered ? Theme.cardHover : "transparent")
                    Behavior on color { ColorAnimation { duration: 140 } }
                    Ripple {
                        id: railRip
                        rippleColor: Qt.rgba(Theme.accent.r, Theme.accent.g,
                                             Theme.accent.b, 0.16)
                    }
                }
                contentItem: Text {
                    text: rb.glyph
                    color: rb.active ? Theme.accent : rb.glyphColor
                    font.pixelSize: rb.emoji ? 15 : 17
                    font.bold: !rb.emoji
                    font.family: rb.emoji ? "Noto Color Emoji" : Qt.application.font.family
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    Behavior on color { ColorAnimation { duration: 140 } }
                }
            }

            Item { Layout.preferredHeight: 2 }

            RailButton {
                glyph: "＋"
                onClicked: { goPage("chat"); backend.newSession() }
                ToolTip.visible: hovered; ToolTip.text: "Nuova chat"
            }
            RailButton {
                glyph: "💬"; emoji: true
                active: appState.page === "chat"
                onClicked: goPage("chat")
                ToolTip.visible: hovered; ToolTip.text: "Chat"
            }
            RailButton {
                glyph: ">_"
                active: appState.page === "terminal"
                onClicked: goPage("terminal")
                ToolTip.visible: hovered; ToolTip.text: "Terminale"
            }
            RailButton {
                glyph: "📊"; emoji: true
                active: appState.page === "system"
                onClicked: goPage("system")
                ToolTip.visible: hovered; ToolTip.text: "Sistema"
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
                Layout.alignment: Qt.AlignHCenter
                radius: 22
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#8fb8fe" }
                    GradientStop { position: 1.0; color: "#c39efb" }
                }
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

        // ── pannello sessioni (solo chat) ───────────────────────────
        Glass {
            id: sidebar
            Layout.fillHeight: true
            Layout.preferredWidth: appState.page === "chat" ? 264 : 0
            glassRadius: Theme.radius
            fill: Theme.panelGlass
            clip: true
            visible: Layout.preferredWidth > 0
            Behavior on Layout.preferredWidth {
                NumberAnimation {
                    duration: Theme.durMed
                    easing.type: Easing.BezierSpline
                    easing.bezierCurve: Theme.emphasized
                }
            }

            Flickable {
                anchors.fill: parent
                anchors.margins: 18
                contentHeight: sessCol.height
                clip: true

                Column {
                    id: sessCol
                    width: parent.width
                    spacing: 12

                    Text {
                        text: "Conversazioni"
                        color: Theme.text
                        font.pixelSize: 19
                        font.bold: true
                        bottomPadding: 2
                    }

                    Repeater {
                        model: sessionGroups()
                        delegate: Column {
                            width: sessCol.width
                            spacing: 6
                            Text {
                                text: modelData.label
                                color: Theme.textDim
                                font.pixelSize: 11
                                font.bold: true
                                font.capitalization: Font.AllUppercase
                                font.letterSpacing: 0.5
                                topPadding: 8
                            }
                            Repeater {
                                model: modelData.items
                                delegate: Rectangle {
                                    width: sessCol.width
                                    height: 58
                                    radius: Theme.radiusIn
                                    property bool editing: false
                                    color: modelData.id === appState.sessionId
                                           ? Theme.accentSoft
                                           : (cardArea.containsMouse ? Theme.cardHover : "transparent")
                                    Behavior on color { ColorAnimation { duration: 120 } }

                                    Column {
                                        visible: !editing
                                        anchors.left: parent.left
                                        anchors.right: parent.right
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.leftMargin: 14
                                        anchors.rightMargin: 26
                                        spacing: 3
                                        Text {
                                            width: parent.width
                                            text: modelData.name || modelData.id
                                            color: modelData.id === appState.sessionId
                                                   ? Theme.accent : Theme.text
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

                                    TextField {
                                        visible: editing
                                        anchors.fill: parent
                                        anchors.margins: 8
                                        text: modelData.name || ""
                                        color: Theme.text
                                        font.pixelSize: 13
                                        background: Rectangle {
                                            color: Theme.cardStrong; radius: 10
                                            border.color: Theme.accentLine
                                        }
                                        onVisibleChanged: if (visible) { forceActiveFocus(); selectAll() }
                                        onAccepted: {
                                            if (text.trim().length > 0)
                                                backend.renameSession(modelData.id, text.trim())
                                            editing = false
                                        }
                                        onActiveFocusChanged: if (!activeFocus) editing = false
                                        Keys.onEscapePressed: editing = false
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
                                        onDoubleClicked: editing = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── colonna principale ──────────────────────────────────────
        Glass {
            Layout.fillWidth: true
            Layout.fillHeight: true
            glassRadius: Theme.radius
            fill: Theme.panelGlass
            clip: true

            Item {
                anchors.fill: parent

                // ── pagine (scorrono SOTTO l'header frosted) ────────
                Item {
                    anchors.fill: parent

                    StackLayout {
                        id: pages
                        anchors.fill: parent
                        currentIndex: appState.page === "terminal" ? 1
                                    : appState.page === "system"   ? 2 : 0

                        // ── pagina chat ─────────────────────────────
                        ColumnLayout {
                            spacing: 0

                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.topMargin: 124
                                spacing: 24
                                visible: chatModel.count === 0 && appState.voiceState !== "loading"

                                Column {
                                    Layout.alignment: Qt.AlignHCenter
                                    spacing: 6
                                    Text {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        text: "Ciao, Mauro!"
                                        color: Theme.textDim
                                        font.pixelSize: 21
                                    }
                                    Text {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        text: "Come posso aiutarti?"
                                        color: Theme.text
                                        font.pixelSize: 28
                                        font.bold: true
                                    }
                                }

                                RowLayout {
                                    Layout.alignment: Qt.AlignHCenter
                                    spacing: 10

                                    component SuggestChip: AbstractButton {
                                        id: chip
                                        property string emoji: ""
                                        property string label: ""
                                        implicitWidth: chipRow.implicitWidth + 30
                                        implicitHeight: 46
                                        scale: pressed ? 0.96 : (hovered ? 1.05 : 1.0)
                                        Behavior on scale { SpringAnimation { spring: 3.6; damping: 0.26 } }
                                        onPressed: chipRip.trigger(pressX, pressY)
                                        background: Rectangle {
                                            radius: 23
                                            color: chip.hovered ? Theme.cardHover : Theme.card
                                            Behavior on color { ColorAnimation { duration: 140 } }
                                            Ripple { id: chipRip }
                                        }
                                        contentItem: Row {
                                            id: chipRow
                                            spacing: 8
                                            leftPadding: 15
                                            Text {
                                                text: chip.emoji
                                                font.pixelSize: 15
                                                font.family: "Noto Color Emoji"
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                            Text {
                                                text: chip.label
                                                color: Theme.text
                                                font.pixelSize: 13
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                        }
                                    }

                                    SuggestChip {
                                        emoji: "📂"; label: "Allega un file"
                                        onClicked: backend.pickFiles()
                                    }
                                    SuggestChip {
                                        emoji: "🖥️"; label: "Guarda lo schermo"
                                        onClicked: {
                                            input.text = "Guarda lo schermo e "
                                            input.forceActiveFocus()
                                            input.cursorPosition = input.text.length
                                        }
                                    }
                                    SuggestChip {
                                        emoji: "🌐"; label: "Cerca sul web"
                                        onClicked: {
                                            input.text = "Cerca online "
                                            input.forceActiveFocus()
                                            input.cursorPosition = input.text.length
                                        }
                                    }
                                    SuggestChip {
                                        emoji: "⚡"; label: "Apri il terminale"
                                        onClicked: goPage("terminal")
                                    }
                                }
                            }

                            ListView {
                                id: chatList
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                model: chatModel
                                spacing: 18
                                clip: true
                                topMargin: 84; bottomMargin: 20; leftMargin: 26; rightMargin: 26
                                boundsBehavior: Flickable.StopAtBounds

                                add: Transition {
                                    NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 220 }
                                    NumberAnimation {
                                        property: "y"; from: chatList.height
                                        duration: Theme.durMed
                                        easing.type: Easing.BezierSpline
                                        easing.bezierCurve: Theme.emphasized
                                    }
                                }
                                populate: Transition {
                                    id: popTrans
                                    SequentialAnimation {
                                        PauseAnimation {
                                            duration: Math.min(popTrans.ViewTransition.index * 45, 360)
                                        }
                                        ParallelAnimation {
                                            NumberAnimation { property: "opacity"; from: 0; to: 1
                                                              duration: 260 }
                                            NumberAnimation { property: "y"
                                                              from: popTrans.ViewTransition.destination.y + 24
                                                              duration: 320
                                                              easing.type: Easing.BezierSpline
                                                              easing.bezierCurve: Theme.emphasized }
                                        }
                                    }
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
                                    id: msgRow
                                    width: chatList.width - chatList.leftMargin - chatList.rightMargin
                                    height: role === "user" ? bubbleLoader.height : flowLoader.height
                                    readonly property real colW: Math.min(width, 780)
                                    readonly property real colX: (width - colW) / 2
                                    readonly property bool isLast: index === chatModel.count - 1

                                    MouseArea {
                                        id: rowHover
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        acceptedButtons: Qt.NoButton
                                    }

                                    TextEdit { id: copyHelper; visible: false; text: model.text }

                                    Loader {
                                        id: bubbleLoader
                                        active: role === "user"
                                        x: msgRow.colX
                                        width: msgRow.colW
                                        sourceComponent: Item {
                                            implicitHeight: uBubble.height
                                            Rectangle {
                                                id: uBubble
                                                anchors.right: parent.right
                                                width: Math.min(uText.implicitWidth + 34,
                                                                parent.width * 0.72)
                                                height: uText.implicitHeight + 26
                                                topLeftRadius: 18
                                                topRightRadius: 18
                                                bottomLeftRadius: 18
                                                bottomRightRadius: 6
                                                border.width: 1
                                                border.color: Theme.accentLine
                                                gradient: Gradient {
                                                    GradientStop {
                                                        position: 0.0
                                                        color: Qt.lighter(Theme.userFill,
                                                                          Theme.dark ? 1.12 : 1.03)
                                                    }
                                                    GradientStop {
                                                        position: 1.0
                                                        color: Theme.userFill
                                                    }
                                                }
                                                TextEdit {
                                                    id: uText
                                                    anchors.fill: parent
                                                    anchors.margins: 12
                                                    text: model.text
                                                    textFormat: TextEdit.MarkdownText
                                                    color: Theme.text
                                                    font.pixelSize: 14
                                                    font.family: "Noto Sans"
                                                    wrapMode: Text.Wrap
                                                    readOnly: true
                                                    selectByMouse: true
                                                    selectionColor: Theme.accent
                                                }
                                            }
                                        }
                                    }

                                    Loader {
                                        id: flowLoader
                                        active: role !== "user"
                                        x: msgRow.colX
                                        width: msgRow.colW
                                        sourceComponent: Item {
                                            implicitHeight: msgView.implicitHeight + 6
                                            Rectangle {
                                                width: 22; height: 22; radius: 11
                                                y: 2
                                                gradient: Gradient {
                                                    GradientStop { position: 0.0; color: "#8fb8fe" }
                                                    GradientStop { position: 1.0; color: "#c39efb" }
                                                }
                                                Text {
                                                    anchors.centerIn: parent
                                                    text: "✦"; color: "white"; font.pixelSize: 10
                                                }
                                            }
                                            MessageView {
                                                id: msgView
                                                x: 34
                                                width: parent.width - 34
                                                content: model.text
                                                showCaret: msgRow.isLast && appState.streaming
                                            }
                                        }
                                    }

                                    Rectangle {
                                        visible: rowHover.containsMouse || copyArea.containsMouse
                                        width: 26; height: 26; radius: 13
                                        x: msgRow.colX + msgRow.colW - width + 8
                                        y: -6
                                        color: copyArea.containsMouse ? Theme.accent : Theme.cardStrong
                                        z: 5
                                        Text {
                                            anchors.centerIn: parent
                                            text: "⧉"
                                            color: copyArea.containsMouse ? "white" : Theme.textDim
                                            font.pixelSize: 12
                                        }
                                        MouseArea {
                                            id: copyArea
                                            anchors.fill: parent
                                            hoverEnabled: true
                                            onClicked: {
                                                copyHelper.selectAll()
                                                copyHelper.copy()
                                                copyHelper.deselect()
                                            }
                                        }
                                        ToolTip.visible: copyArea.containsMouse
                                        ToolTip.text: "Copia messaggio"
                                    }
                                }

                                footer: Item {
                                    width: chatList.width - chatList.leftMargin - chatList.rightMargin
                                    height: typing.visible ? 36 : 0
                                    TypingDots {
                                        id: typing
                                        visible: appState.voiceState === "thinking" && !appState.streaming
                                        x: (parent.width - Math.min(parent.width, 780)) / 2 + 34
                                        anchors.verticalCenter: parent.verticalCenter
                                        dotColor: Theme.accent
                                    }
                                }
                            }

                            Item {
                                Layout.fillWidth: true
                                Layout.leftMargin: 24
                                Layout.rightMargin: 24
                                Layout.preferredHeight: appState.attachments.length > 0
                                                        ? attFlow.implicitHeight + 4 : 0
                                clip: true
                                Behavior on Layout.preferredHeight {
                                    NumberAnimation {
                                        duration: Theme.durShort
                                        easing.type: Easing.BezierSpline
                                        easing.bezierCurve: Theme.emphasized
                                    }
                                }

                                Flow {
                                id: attFlow
                                width: parent.width
                                spacing: 6
                                opacity: appState.attachments.length > 0 ? 1 : 0
                                Behavior on opacity { NumberAnimation { duration: 180 } }

                                Repeater {
                                    model: appState.attachments
                                    delegate: Rectangle {
                                        width: attRow.width + 20; height: 28; radius: 14
                                        color: Theme.accentSoft
                                        Row {
                                            id: attRow
                                            anchors.centerIn: parent
                                            spacing: 6
                                            Text { text: "📄"; font.pixelSize: 11
                                                   font.family: "Noto Color Emoji"
                                                   anchors.verticalCenter: parent.verticalCenter }
                                            Text { text: modelData.name; color: Theme.accent
                                                   font.pixelSize: 11
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
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                Layout.margins: 18
                                Layout.topMargin: 10
                                spacing: 10

                                StateOrb {
                                    voiceState: appState.voiceState
                                    onPressedChanged: pressed ? backend.pttDown() : backend.pttUp()
                                }

                                Item {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 48

                                    TextField {
                                        id: input
                                        anchors.fill: parent
                                        rightPadding: 46
                                        leftPadding: 18
                                        placeholderText: appState.voiceState === "loading"
                                                         ? "Caricamento assistente…"
                                                         : "Chiedimi qualsiasi cosa…"
                                        placeholderTextColor: Theme.textDim
                                        enabled: appState.voiceState !== "loading"
                                        color: Theme.text
                                        font.pixelSize: 14
                                        font.family: "Noto Sans"
                                        background: Rectangle {
                                            color: input.activeFocus ? Theme.cardHover : Theme.inputFill
                                            radius: 24
                                            border.width: input.activeFocus ? 2 : 0
                                            border.color: Theme.accent
                                            Behavior on color { ColorAnimation { duration: 140 } }
                                        }
                                        onAccepted: sendCurrentInput()
                                    }

                                    AbstractButton {
                                        id: attachBtn
                                        width: 36; height: 36
                                        anchors.right: parent.right
                                        anchors.rightMargin: 6
                                        anchors.verticalCenter: parent.verticalCenter
                                        onPressed: attachRip.trigger(pressX, pressY)
                                        onClicked: backend.pickFiles()
                                        ToolTip.visible: hovered
                                        ToolTip.text: "Allega file (o trascinali qui)"
                                        background: Rectangle {
                                            radius: 18
                                            color: attachBtn.hovered ? Theme.cardHover : "transparent"
                                            Behavior on color { ColorAnimation { duration: 120 } }
                                            Ripple { id: attachRip }
                                        }
                                        contentItem: Text {
                                            text: "📎"
                                            font.pixelSize: 14
                                            font.family: "Noto Color Emoji"
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter
                                        }
                                    }
                                }

                                AbstractButton {
                                    id: stopBtn
                                    visible: appState.voiceState === "thinking"
                                             || appState.voiceState === "speaking"
                                    implicitWidth: 46; implicitHeight: 46
                                    onPressed: stopRip.trigger(pressX, pressY)
                                    onClicked: backend.cancelTurn()
                                    background: Rectangle {
                                        radius: 23
                                        color: Theme.danger
                                        opacity: stopBtn.hovered ? 1.0 : 0.88
                                        Ripple { id: stopRip; rippleColor: Qt.rgba(1, 1, 1, 0.25) }
                                    }
                                    contentItem: Text {
                                        text: "◼"; color: "white"; font.pixelSize: 12
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                }

                                AbstractButton {
                                    id: sendBtn
                                    implicitWidth: 48; implicitHeight: 48
                                    enabled: input.text.trim().length > 0
                                             || appState.attachments.length > 0
                                    scale: enabled ? 1.0 : 0.92
                                    Behavior on scale { SpringAnimation { spring: 3.5; damping: 0.3 } }
                                    onPressed: sendRip.trigger(pressX, pressY)
                                    onClicked: sendCurrentInput()
                                    background: Rectangle {
                                        radius: 24
                                        color: sendBtn.enabled ? Theme.accent : Theme.cardHover
                                        Behavior on color { ColorAnimation { duration: 140 } }
                                        Ripple { id: sendRip; rippleColor: Qt.rgba(1, 1, 1, 0.28) }
                                    }
                                    contentItem: Text {
                                        text: "↑"
                                        color: sendBtn.enabled ? "white" : Theme.textDim
                                        font.pixelSize: 17
                                        font.bold: true
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                }
                            }
                        }

                        // ── pagina terminale ────────────────────────
                        ColumnLayout {
                            spacing: 0
                            Item { Layout.preferredHeight: 64 }
                            TerminalPage {
                                id: terminalPage
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                feed: terminalFeed
                            }
                        }

                        // ── pagina sistema ──────────────────────────
                        ColumnLayout {
                            spacing: 0
                            Item { Layout.preferredHeight: 64 }
                            SystemPage {
                                id: systemPage
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                latencyHistory: latencyHistory
                            }
                        }
                    }

                    ParallelAnimation {
                        id: pageTrans
                        NumberAnimation {
                            target: pages; property: "opacity"
                            from: 0; to: 1
                            duration: Theme.durMed
                            easing.type: Easing.BezierSpline
                            easing.bezierCurve: Theme.emphasized
                        }
                        NumberAnimation {
                            target: pages; property: "y"
                            from: 14; to: 0
                            duration: Theme.durMed
                            easing.type: Easing.BezierSpline
                            easing.bezierCurve: Theme.emphasized
                        }
                    }

                // ── header frosted (macOS): sfoca ciò che scorre sotto ──
                Item {
                    id: headerZone
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 64

                    // NB: niente maschera arrotondata sul frost — MultiEffect
                    // maskSource non dipinge in questo scenario (testato: layer
                    // e ShaderEffectSource producono texture vuota) e il velo è
                    // così vicino al tono del pannello che gli angoli quadrati
                    // sono invisibili.
                    Item {
                        anchors.fill: parent

                        // campione live del contenuto sottostante, sfocato
                        ShaderEffectSource {
                            id: headerSample
                            anchors.fill: parent
                            anchors.leftMargin: Theme.radius
                            anchors.rightMargin: Theme.radius
                            sourceItem: pages
                            sourceRect: Qt.rect(Theme.radius, 0,
                                                headerZone.width - Theme.radius * 2,
                                                headerZone.height)
                            live: true
                            visible: false
                        }
                        MultiEffect {
                            anchors.fill: parent
                            // il blur è rettangolare: lo si tiene DENTRO la
                            // curva degli angoli del pannello
                            anchors.leftMargin: Theme.radius
                            anchors.rightMargin: Theme.radius
                            source: headerSample
                            blurEnabled: true
                            blur: 1.0
                            blurMax: 32
                        }
                        // velo di vetro con i soli angoli superiori arrotondati
                        Rectangle {
                            anchors.fill: parent
                            color: Theme.headerTint
                            topLeftRadius: Theme.radius
                            topRightRadius: Theme.radius
                            bottomLeftRadius: 0
                            bottomRightRadius: 0
                        }
                    }

                    // hairline: appare quando il contenuto scorre sotto
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 18
                        anchors.rightMargin: 18
                        height: 1
                        color: Theme.cardLine
                        opacity: appState.page !== "chat"
                                 || chatList.contentY > (-chatList.topMargin + 4) ? 1 : 0
                        Behavior on opacity { NumberAnimation { duration: 180 } }
                    }

                    // contenuto dell'header
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 24
                        anchors.rightMargin: 20
                        spacing: 12

                        ColumnLayout {
                            spacing: 1
                            Text {
                                text: appState.page === "terminal" ? "Terminale"
                                    : appState.page === "system"   ? "Sistema"
                                    : appState.sessionName
                                color: Theme.text
                                font.pixelSize: 19
                                font.bold: true
                                elide: Text.ElideRight
                                Layout.maximumWidth: 380
                            }
                            Row {
                                spacing: 6
                                Rectangle {
                                    width: 7; height: 7; radius: 3.5
                                    anchors.verticalCenter: parent.verticalCenter
                                    color: appState.voiceState === "idle"
                                           || appState.voiceState === "listening"
                                           ? Theme.okCol : Theme.accent
                                    SequentialAnimation on opacity {
                                        running: appState.voiceState !== "idle"
                                        loops: Animation.Infinite
                                        NumberAnimation { to: 0.4; duration: 700 }
                                        NumberAnimation { to: 1.0; duration: 700 }
                                    }
                                }
                                Text {
                                    text: appState.stateLabel
                                    color: Theme.textDim
                                    font.pixelSize: 11
                                }
                            }
                        }

                        Item { Layout.fillWidth: true }

                        Rectangle {
                            visible: appState.latency.total_ms !== undefined
                            width: latText.width + 18; height: 26; radius: 13
                            color: Theme.card
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

                        AbstractButton {
                            id: setupBtn
                            implicitWidth: setupRow.implicitWidth + 26
                            implicitHeight: 34
                            onPressed: setupRip.trigger(pressX, pressY)
                            onClicked: setupPopup.open()
                            background: Rectangle {
                                radius: 17
                                color: setupBtn.hovered || setupPopup.visible
                                       ? Theme.cardHover : Theme.card
                                Behavior on color { ColorAnimation { duration: 140 } }
                                Ripple { id: setupRip }
                            }
                            contentItem: Row {
                                id: setupRow
                                spacing: 7
                                leftPadding: 13
                                Text {
                                    text: "⚙"
                                    color: Theme.textDim
                                    font.pixelSize: 13
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    text: appState.page === "terminal"
                                          ? (appState.terminalModel || "modello")
                                          : appState.model.split("/").pop().split(":")[0] || "modello"
                                    color: Theme.text
                                    font.pixelSize: 12
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                            ToolTip.visible: hovered && !setupPopup.visible
                            ToolTip.text: "Modello, profilo e voce"
                        }
                    }
                }
            }
        }
    }

    }

    // ── popover impostazioni (modello / profilo / voce) ─────────────
    Popup {
        id: setupPopup
        x: parent.width - width - 34
        y: 74
        width: 300
        padding: 16
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        enter: Transition {
            ParallelAnimation {
                NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 180 }
                NumberAnimation { property: "scale"; from: 0.94; to: 1; duration: 220
                                  easing.type: Easing.BezierSpline
                                  easing.bezierCurve: Theme.emphasized }
            }
        }
        exit: Transition {
            NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 120 }
        }

        background: Rectangle {
            color: Theme.cardStrong
            radius: 20
            border.width: 1
            border.color: Theme.cardLine
        }

        contentItem: ColumnLayout {
            spacing: 12

            Text {
                text: "Impostazioni rapide"
                color: Theme.text
                font.pixelSize: 14
                font.bold: true
            }

            ThemedCombo {
                Layout.fillWidth: true
                comboModel: appState.page === "terminal" ? appState.terminalModels
                                                         : appState.models
                current: appState.page === "terminal" ? appState.terminalModel
                                                      : appState.model
                label: appState.page === "terminal" ? "modello terminale" : "modello"
                onPicked: (name) => appState.page === "terminal"
                                    ? backend.terminalSwitchModel(name)
                                    : backend.switchModel(name)
            }
            ThemedCombo {
                Layout.fillWidth: true
                visible: appState.page !== "terminal"
                comboModel: appState.personalities.map(p => p.name)
                current: appState.personality
                label: "profilo"
                onPicked: (name) => backend.switchPersonality(name)
            }
            ThemedCombo {
                Layout.fillWidth: true
                visible: appState.page !== "terminal"
                comboModel: appState.voices
                current: appState.voice
                label: "voce"
                onPicked: (name) => backend.switchVoice(name)
            }
        }
    }

    // ── drag & drop (sopra tutto) ───────────────────────────────────
    DropArea {
        anchors.fill: parent
        enabled: appState.page === "chat"
        onDropped: (drop) => {
            if (drop.hasUrls)
                for (var i = 0; i < drop.urls.length; i++)
                    backend.uploadFile(drop.urls[i].toString())
        }
        Rectangle {
            anchors.fill: parent
            anchors.margins: 12
            visible: parent.containsDrag
            color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.07)
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

    // ── pannello errori ─────────────────────────────────────────────
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
