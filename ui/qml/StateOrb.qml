// ui/qml/StateOrb.qml — indicatore animato dello stato vocale che fa anche
// da pulsante microfono (tieni premuto per parlare, come il PTT).
//
// Stati: loading/warmup (viola, respiro lento) · idle (ciano, respiro tenue)
//        recording (rosso, pulsazione) · thinking (arco che ruota)
//        speaking (barre audio animate)

import QtQuick

Item {
    id: orb
    property string voiceState: "idle"
    property alias pressed: tapArea.pressed

    implicitWidth: 46
    implicitHeight: 46

    readonly property color cIdle:     "#4dabf7"
    readonly property color cRec:      "#f03e3e"
    readonly property color cThink:    "#9775fa"
    readonly property color cSpeak:    "#3bc9db"
    readonly property color cLoad:     "#6c757d"

    readonly property color stateColor:
        voiceState === "recording" ? cRec
      : voiceState === "thinking"  ? cThink
      : voiceState === "speaking"  ? cSpeak
      : (voiceState === "loading" || voiceState === "warmup") ? cLoad
      : cIdle

    // ── disco di base ───────────────────────────────────────────────
    Rectangle {
        id: disc
        anchors.centerIn: parent
        width: parent.width
        height: width
        radius: width / 2
        color: Qt.rgba(orb.stateColor.r, orb.stateColor.g, orb.stateColor.b, 0.16)
        border.color: orb.stateColor
        border.width: 2

        // respiro (idle/loading) e pulsazione (recording)
        SequentialAnimation on scale {
            running: orb.voiceState !== "thinking" && orb.voiceState !== "speaking"
            loops: Animation.Infinite
            NumberAnimation {
                to: orb.voiceState === "recording" ? 1.14 : 1.05
                duration: orb.voiceState === "recording" ? 420 : 1400
                easing.type: Easing.InOutSine
            }
            NumberAnimation {
                to: 1.0
                duration: orb.voiceState === "recording" ? 420 : 1400
                easing.type: Easing.InOutSine
            }
        }
    }

    // ── arco rotante (thinking) ─────────────────────────────────────
    Canvas {
        id: arc
        anchors.fill: parent
        visible: orb.voiceState === "thinking"
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            ctx.strokeStyle = orb.cThink
            ctx.lineWidth = 3
            ctx.lineCap = "round"
            ctx.beginPath()
            ctx.arc(width / 2, height / 2, width / 2 - 3, 0, Math.PI * 1.25)
            ctx.stroke()
        }
        RotationAnimation on rotation {
            running: arc.visible
            loops: Animation.Infinite
            from: 0; to: 360
            duration: 900
        }
    }

    // ── barre audio (speaking) ──────────────────────────────────────
    Row {
        anchors.centerIn: parent
        spacing: 3
        visible: orb.voiceState === "speaking"
        Repeater {
            model: 4
            delegate: Rectangle {
                width: 4
                radius: 2
                color: orb.cSpeak
                anchors.verticalCenter: parent.verticalCenter
                height: 8
                SequentialAnimation on height {
                    running: orb.voiceState === "speaking"
                    loops: Animation.Infinite
                    NumberAnimation {
                        to: 18 + (index % 2) * 6
                        duration: 260 + index * 70
                        easing.type: Easing.InOutSine
                    }
                    NumberAnimation {
                        to: 6 + index * 2
                        duration: 260 + index * 70
                        easing.type: Easing.InOutSine
                    }
                }
            }
        }
    }

    // ── icona microfono (idle/recording) ────────────────────────────
    Text {
        anchors.centerIn: parent
        visible: orb.voiceState !== "thinking" && orb.voiceState !== "speaking"
        text: "🎙"
        font.pixelSize: 17
        opacity: (orb.voiceState === "loading" || orb.voiceState === "warmup") ? 0.4 : 0.95
    }

    MouseArea {
        id: tapArea
        anchors.fill: parent
        enabled: orb.voiceState === "idle" || orb.voiceState === "listening"
                 || orb.voiceState === "recording"
        cursorShape: Qt.PointingHandCursor
    }
}
