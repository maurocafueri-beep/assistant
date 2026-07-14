// ui/qml/MessageView.qml — renderer del messaggio in stile Claude: la prosa
// scorre come testo sul fondo (markdown), i code-fence ``` diventano blocchi
// scuri (CodeBlock) con linguaggio e pulsante copia. Regge lo streaming:
// un fence non ancora chiuso viene mostrato come codice "in crescita".

import QtQuick
import "."

Column {
    id: view
    property string content: ""
    property bool showCaret: false     // cursore pulsante a fine stream
    spacing: 10

    // [{kind:"prose"|"code", text, lang}]
    property var blocks: parseBlocks(content)

    function parseBlocks(text) {
        var out = []
        var rest = text
        while (true) {
            var open = rest.indexOf("```")
            if (open < 0) break
            if (open > 0)
                out.push({ kind: "prose", text: rest.slice(0, open), lang: "" })
            var nl = rest.indexOf("\n", open + 3)
            var lang = nl > 0 ? rest.slice(open + 3, nl).trim() : ""
            var bodyStart = nl > 0 ? nl + 1 : open + 3
            var close = rest.indexOf("```", bodyStart)
            if (close < 0) {
                // fence aperto (streaming): tutto il resto è codice
                out.push({ kind: "code", text: rest.slice(bodyStart), lang: lang })
                return out
            }
            out.push({ kind: "code",
                       text: rest.slice(bodyStart, close).replace(/\n$/, ""),
                       lang: lang })
            rest = rest.slice(close + 3)
        }
        if (rest.trim().length > 0)
            out.push({ kind: "prose", text: rest, lang: "" })
        return out
    }

    Repeater {
        model: view.blocks
        delegate: Loader {
            width: view.width
            sourceComponent: modelData.kind === "code" ? codeComp : proseComp

            Component {
                id: proseComp
                TextEdit {
                    width: view.width
                    text: modelData.text
                    textFormat: TextEdit.MarkdownText
                    color: Theme.text
                    font.pixelSize: 15
                    font.family: "Noto Sans"
                    wrapMode: Text.Wrap
                    readOnly: true
                    selectByMouse: true
                    selectionColor: Theme.accent
                }
            }
            Component {
                id: codeComp
                CodeBlock {
                    width: view.width
                    code: modelData.text
                    language: modelData.lang
                }
            }
        }
    }

    // cursore di streaming (blocco pulsante, come Claude)
    Rectangle {
        visible: view.showCaret
        width: 9; height: 17; radius: 2
        color: Theme.accent
        SequentialAnimation on opacity {
            running: view.showCaret
            loops: Animation.Infinite
            NumberAnimation { to: 0.15; duration: 500; easing.type: Easing.InOutSine }
            NumberAnimation { to: 1.0;  duration: 500; easing.type: Easing.InOutSine }
        }
    }
}
