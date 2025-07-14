import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    width: Screen.width * 0.15
    color: "white"
    border.color: "#888"
    radius: 10
    border.width: 1

    ColumnLayout {
        id: contentLayout
        anchors.centerIn: parent
        spacing: 6
        width: parent.width * 0.9
        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Достоверность:"
                Layout.fillWidth: true
                Layout.preferredWidth: 2
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            TextField {
                id: conf_threshold
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                placeholderText: ""
                text : Number(onnxloaderModel.confThreshold).toFixed(3).replace(".", ",")

                validator: DoubleValidator {
                    locale: Qt.locale("C")
                    bottom: 0.0
                    top: 1.0
                    decimals: 3
                    notation: DoubleValidator.StandardNotation
                }
                onTextChanged: {
                    if (acceptableInput) {
                        let normalized = text.replace(",", ".")
                        onnxloaderModel.confThreshold = parseFloat(normalized)
                    }
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Порог IoU:"
                Layout.fillWidth: true
                Layout.preferredWidth: 2
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            TextField {
                id: nms_threshold
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                placeholderText: ""
                text : Number(onnxloaderModel.nmsThreshold).toFixed(3).replace(".", ",")

                validator: DoubleValidator {
                    bottom: 0.0
                    top: 1.0
                    decimals: 3
                    notation: DoubleValidator.StandardNotation
                }
                onTextChanged: {
                    if (acceptableInput) {
                        let normalized = text.replace(",", ".")
                        onnxloaderModel.nmsThreshold = parseFloat(normalized)
                    }
                }
            }
        }
        GroupBox {
            Layout.fillWidth: true
            title: "Сохранение результатов"

            ColumnLayout {
                anchors.fill: parent
                spacing: 4

                CheckBox {
                    id: enableDir
                    text: "Сохранять в каталог"
                    checked: onnxloaderModel.saveResult
                    onCheckedChanged: onnxloaderModel.saveResult = checked
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    TextField {
                        id: dirPathField
                        Layout.fillWidth: true
                        readOnly: true
                        enabled: enableDir.checked
                        text: onnxloaderModel.resultFolder
                    }

                    Button {
                        icon.source: "qrc:/images/folder.png"
                        icon.width: 16
                        icon.height: 16
                        Layout.preferredWidth: 30
                        enabled: enableDir.checked
                        onClicked: onnxloaderModel.selectResultFolder()
                    }
                }
            }
        }
        Button {
            text: "Закрыть"
            Layout.alignment: Qt.AlignHCenter
            onClicked: onnxloaderModel.closeWindowInfo()
        }
    }
    height: contentLayout.implicitHeight + 20
}
