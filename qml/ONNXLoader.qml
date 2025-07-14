import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    width: Screen.width * 0.2
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
                text: "Наличие CUDA:"
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            Label {
                text: onnxloaderModel.cudaDeviceNumber
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                font.pointSize: 11
            }
        }

        Label {
            text: onnxloaderModel.deviceName
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            font.pointSize: 11
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Память:"
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            Label {
                text: onnxloaderModel.globalMem
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                font.pointSize: 11
            }
        }

        Label {
            text: "Доступные провайдеры:"
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignLeft
            font.pointSize: 11
            font.bold: true
        }

        Text {
            text: onnxloaderModel.providers
            wrapMode: Text.WordWrap
            font.pointSize: 11
            horizontalAlignment: Text.AlignLeft
            width: parent.width - 20
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Входы сети:"
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            Label {
                text: onnxloaderModel.inputShapeInfo
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                font.pointSize: 11
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Выходы сети:"
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignLeft
                font.pointSize: 11
                font.bold: true
            }
            Label {
                text: onnxloaderModel.outputShapeInfo
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                font.pointSize: 11
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
