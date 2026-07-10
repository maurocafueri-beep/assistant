// ui/qml/Theme.qml — singleton del tema (chiaro pastello / scuro).
// Ogni colore ha un Behavior: al toggle l'intera UI sfuma con un
// cross-fade di ~280ms invece di scattare.

pragma Singleton
import QtQuick

Item {
    id: t
    property bool dark: false

    readonly property int radius:   26
    readonly property int radiusIn: 18

    property color text:    dark ? "#e8ecf4" : "#2b3040"
    property color textDim: dark ? "#98a2b8" : "#8b93a7"
    property color accent:  dark ? "#5b96ff" : "#3d7ef7"
    property color danger:  dark ? "#ff6b6b" : "#e5484d"
    property color warn:    dark ? "#ffd43b" : "#d9a300"
    property color okCol:   dark ? "#51cf66" : "#2f9e44"

    // sfondo (gradiente + aloni)
    property color bg0:   dark ? "#151823" : "#faf8fd"
    property color bg1:   dark ? "#101219" : "#f3edf9"
    property color bg2:   dark ? "#171523" : "#f9eef3"
    property color blob1: dark ? "#2c3766" : "#dcd2f5"
    property color blob2: dark ? "#4a2e57" : "#f7d9e4"

    // card di vetro
    property color card:       dark ? Qt.rgba(0.11, 0.125, 0.175, 0.66) : Qt.rgba(1, 1, 1, 0.62)
    property color cardSoft:   dark ? Qt.rgba(0.11, 0.125, 0.175, 0.45) : Qt.rgba(1, 1, 1, 0.45)
    property color cardHover:  dark ? Qt.rgba(0.16, 0.18, 0.25, 0.85)   : Qt.rgba(1, 1, 1, 0.85)
    property color cardStrong: dark ? Qt.rgba(0.18, 0.20, 0.28, 0.97)   : Qt.rgba(1, 1, 1, 0.95)
    property color cardLine:   dark ? Qt.rgba(1, 1, 1, 0.09)            : Qt.rgba(1, 1, 1, 0.90)
    property color shadowCol:  dark ? "#000000" : "#3a3550"

    // bolle
    property color mintFill:   dark ? Qt.rgba(0.14, 0.24, 0.17, 0.80) : Qt.rgba(0.914, 0.965, 0.925, 0.92)
    property color mintLine:   dark ? Qt.rgba(0.35, 0.62, 0.42, 0.45) : Qt.rgba(0.55, 0.78, 0.60, 0.35)
    property color userFill:   dark ? Qt.rgba(0.17, 0.22, 0.34, 0.85) : Qt.rgba(1, 1, 1, 0.92)
    property color userLine:   dark ? Qt.rgba(0.36, 0.55, 0.95, 0.35) : Qt.rgba(1, 1, 1, 0.90)

    // input / codice / accenti tenui
    property color inputFill:  dark ? Qt.rgba(0, 0, 0, 0.30)  : Qt.rgba(1, 1, 1, 0.90)
    property color codeBg:     dark ? Qt.rgba(0, 0, 0, 0.35)  : "#eef1f7"
    property color codeText:   dark ? "#7fb4ff" : "#1d4ed8"
    property color accentSoft: dark ? Qt.rgba(0.36, 0.55, 0.95, 0.16) : Qt.rgba(0.24, 0.49, 0.97, 0.10)
    property color accentLine: dark ? Qt.rgba(0.36, 0.55, 0.95, 0.40) : Qt.rgba(0.24, 0.49, 0.97, 0.30)
    property color scrollBar:  dark ? Qt.rgba(1, 1, 1, 0.15)  : Qt.rgba(0, 0, 0, 0.12)

    Behavior on text       { ColorAnimation { duration: 280 } }
    Behavior on textDim    { ColorAnimation { duration: 280 } }
    Behavior on accent     { ColorAnimation { duration: 280 } }
    Behavior on bg0        { ColorAnimation { duration: 280 } }
    Behavior on bg1        { ColorAnimation { duration: 280 } }
    Behavior on bg2        { ColorAnimation { duration: 280 } }
    Behavior on blob1      { ColorAnimation { duration: 280 } }
    Behavior on blob2      { ColorAnimation { duration: 280 } }
    Behavior on card       { ColorAnimation { duration: 280 } }
    Behavior on cardSoft   { ColorAnimation { duration: 280 } }
    Behavior on cardHover  { ColorAnimation { duration: 280 } }
    Behavior on cardStrong { ColorAnimation { duration: 280 } }
    Behavior on cardLine   { ColorAnimation { duration: 280 } }
    Behavior on mintFill   { ColorAnimation { duration: 280 } }
    Behavior on mintLine   { ColorAnimation { duration: 280 } }
    Behavior on userFill   { ColorAnimation { duration: 280 } }
    Behavior on userLine   { ColorAnimation { duration: 280 } }
    Behavior on inputFill  { ColorAnimation { duration: 280 } }
    Behavior on codeBg     { ColorAnimation { duration: 280 } }
    Behavior on codeText   { ColorAnimation { duration: 280 } }
    Behavior on accentSoft { ColorAnimation { duration: 280 } }
    Behavior on accentLine { ColorAnimation { duration: 280 } }
    Behavior on scrollBar  { ColorAnimation { duration: 280 } }
    Behavior on shadowCol  { ColorAnimation { duration: 280 } }
}
