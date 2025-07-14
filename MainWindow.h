#pragma once
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSettings>
#include "QONNXLoader.h"
#include "QCaptureReader.h"
#include "QVideoPlayer.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	MainWindow(QWidget *parent = nullptr);
	~MainWindow();
private:
	void enableButtons();
	void removeCaptureReader();
private:
	Ui::MainWindow *ui;
	QONNXLoader *onnxLoader{nullptr};
	QCaptureReader *captureReader{nullptr};
	QVideoPlayer *player{nullptr};
	QSettings settings{"copa","detectOnVideo"};
	const char *lastPathKey = "lastPath";
	const char *lastVideoPathKey = "lastVideoPath";
};
#endif // MAINWINDOW_H
