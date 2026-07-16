// ui/qml/Theme.qml — singleton del tema su filosofia Material You (MD3):
// superfici TONALI (la gerarchia nasce dal tono, non dai bordi), colore
// primario indigo con container, elevazioni delegate alle ombre, motion
// con easing "emphasized". Ogni colore ha un Behavior: il toggle
// chiaro/scuro sfuma in ~280ms.
//
// I nomi delle property restano quelli storici della UI (card, cardStrong,
// accentSoft, …) mappati sui token MD3, così tutte le pagine ereditano il
// nuovo linguaggio senza modifiche.

pragma Singleton
import QtQuick

Item {
    id: t
    property bool dark: false

    // shape MD3
    readonly property int radius:   24    // superfici grandi
    readonly property int radiusIn: 16    // card interne / elementi

    // motion MD3 — curva "emphasized decelerate"
    readonly property var emphasized: [0.05, 0.7, 0.1, 1.0, 1.0, 1.0]
    readonly property int durShort: 200
    readonly property int durMed:   350

    // ── ruoli colore ────────────────────────────────────────────────
    property color text:    dark ? "#E4E1E9" : "#1B1B21"   // onSurface
    property color textDim: dark ? "#A5A3B0" : "#46464F"   // onSurfaceVariant
    property color accent:  dark ? "#6674E8" : "#4C5BD4"   // primary (bianco sopra ok)
    property color danger:  dark ? "#FF8A80" : "#BA1A1A"
    property color warn:    dark ? "#FFD54F" : "#8F6D00"
    property color okCol:   dark ? "#7BD88A" : "#2E7D32"

    // sfondo piatto (surface) — bg0..2 identici: MD3 non usa gradienti
    property color bg0:   dark ? "#131318" : "#FBF8FF"
    property color bg1:   dark ? "#131318" : "#FBF8FF"
    property color bg2:   dark ? "#131318" : "#FBF8FF"
    property color blob1: dark ? "#131318" : "#FBF8FF"   // legacy, non usati
    property color blob2: dark ? "#131318" : "#FBF8FF"

    // superfici tonali (surfaceContainer*)
    property color card:       dark ? "#1E1E25" : "#F2EFF7"   // container-low
    property color cardSoft:   dark ? "#1A1A21" : "#F7F4FB"   // container-lowest
    property color cardHover:  dark ? "#27272F" : "#EAE7F0"   // container-high
    property color cardStrong: dark ? "#2B2B33" : "#FFFFFF"   // container / bright
    property color cardLine:   dark ? Qt.rgba(1,1,1,0.06) : Qt.rgba(0,0,0,0.05) // outlineVariant tenue
    property color shadowCol:  "#000000"

    // bolle e accenti di contenuto
    property color userFill:   dark ? "#2E3357" : "#DFE0FF"   // primaryContainer
    property color userLine:   "transparent"
    property color mintFill:   dark ? "#22372A" : "#DFF2E1"   // tertiaryContainer (verde)
    property color mintLine:   "transparent"

    // input / codice / stati
    property color inputFill:  dark ? "#23232B" : "#ECE9F3"   // container-high (search bar)
    property color codeBg:     dark ? "#0E1016" : "#1B1F27"   // il codice resta scuro
    property color codeText:   dark ? "#A9C7FF" : "#DCE3EE"
    property color accentSoft: dark ? "#2E3163" : "#DFE0FF"   // primaryContainer
    property color accentLine: dark ? Qt.rgba(0.55, 0.62, 1, 0.45) : Qt.rgba(0.30, 0.36, 0.83, 0.40)
    property color scrollBar:  dark ? Qt.rgba(1,1,1,0.16) : Qt.rgba(0,0,0,0.14)

    // vetro (traslucenza in stile macOS: il desktop filtra attraverso)
    property color panelGlass: dark ? Qt.rgba(0.094, 0.10, 0.129, 0.62) : Qt.rgba(1, 1, 1, 0.60)
    property color headerTint: dark ? Qt.rgba(0.075, 0.078, 0.10, 0.55) : Qt.rgba(0.99, 0.98, 1, 0.55)
    property color specular:   dark ? Qt.rgba(1, 1, 1, 0.07)            : Qt.rgba(1, 1, 1, 0.75)

    Behavior on text       { ColorAnimation { duration: 280 } }
    Behavior on textDim    { ColorAnimation { duration: 280 } }
    Behavior on accent     { ColorAnimation { duration: 280 } }
    Behavior on bg0        { ColorAnimation { duration: 280 } }
    Behavior on bg1        { ColorAnimation { duration: 280 } }
    Behavior on bg2        { ColorAnimation { duration: 280 } }
    Behavior on card       { ColorAnimation { duration: 280 } }
    Behavior on cardSoft   { ColorAnimation { duration: 280 } }
    Behavior on cardHover  { ColorAnimation { duration: 280 } }
    Behavior on cardStrong { ColorAnimation { duration: 280 } }
    Behavior on cardLine   { ColorAnimation { duration: 280 } }
    Behavior on userFill   { ColorAnimation { duration: 280 } }
    Behavior on mintFill   { ColorAnimation { duration: 280 } }
    Behavior on inputFill  { ColorAnimation { duration: 280 } }
    Behavior on codeBg     { ColorAnimation { duration: 280 } }
    Behavior on codeText   { ColorAnimation { duration: 280 } }
    Behavior on accentSoft { ColorAnimation { duration: 280 } }
    Behavior on accentLine { ColorAnimation { duration: 280 } }
    Behavior on scrollBar  { ColorAnimation { duration: 280 } }
    Behavior on panelGlass { ColorAnimation { duration: 280 } }
    Behavior on headerTint { ColorAnimation { duration: 280 } }
    Behavior on specular   { ColorAnimation { duration: 280 } }
}
