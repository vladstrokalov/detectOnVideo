#include <QObject>
#include <QPushButton>
#include <QFileInfo>
#include <QFileDialog>
#include <QMessageBox>
#include "MainWindow.h"
#include "./ui_MainWindow.h"

MainWindow::MainWindow(
	QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
	ui->setupUi(this);

	player = new QVideoPlayer(ui->centralwidget);
	ui->centralwidget->layout()->addWidget(player);

	connect(ui->openNeural,&QPushButton::clicked,[&](bool){
		QString lastDir = settings.value(lastPathKey, QDir::homePath()).toString();
		QString fileName = QFileDialog::getOpenFileName(
			this,"Выберите лог файл",
			lastDir,"ONNX файлы (*.onnx)");
		if (!fileName.isEmpty()){
			QFileInfo fileInfo(fileName);
			settings.setValue(lastPathKey, fileInfo.absolutePath());
			if (!fileName.endsWith(".onnx", Qt::CaseInsensitive)) {
				QMessageBox::critical(this, "Ошибка", "Вы должны выбрать файл в формате onnx.");
			}
			else {
				if(onnxLoader != nullptr){
					delete onnxLoader;
					onnxLoader = nullptr;
				}
				onnxLoader = new QONNXLoader(&settings);
				if(!onnxLoader->load(fileName,this)){
					delete onnxLoader;
					onnxLoader = nullptr;
				}
				if(onnxLoader != nullptr) {
					onnxLoader->info(this);
				}
				enableButtons();
			}
			if(captureReader != nullptr){
				captureReader->setONNX(onnxLoader);
			}
		}
	});
	connect(ui->openVideo,&QPushButton::clicked,[&](bool){
		QString lastDir = settings.value(lastVideoPathKey, QDir::homePath()).toString();
		QString fileName = QFileDialog::getOpenFileName(
			this,"Файлы видео",
			lastDir,"Файлы видео (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)");
		if (!fileName.isEmpty()){
			QFileInfo fileInfo(fileName);
			settings.setValue(lastVideoPathKey, fileInfo.absolutePath());
			removeCaptureReader();
			captureReader = new QCaptureReader();
			if(!captureReader->open(fileName)) {
				delete captureReader;
				captureReader = nullptr;
				QString errorMsg = QString("Ошибка открытия видео детекции %1").arg(fileName);
				QMessageBox::critical(this,"Ошибка",errorMsg);
			}
			else {
				connect(captureReader,&QCaptureReader::onEof,[&](){
					enableButtons();
				});
				player->setCaptureReader(captureReader);
				captureReader->setONNX(onnxLoader);

			}
			enableButtons();
		}
	});
	connect(ui->play,&QPushButton::clicked,[&](bool){
		if(onnxLoader  != nullptr && captureReader != nullptr &&
			captureReader->isRunning()){
			captureReader->play();
		}
		enableButtons();
	});
	connect(ui->stop,&QPushButton::clicked,[&](bool){
		if(onnxLoader  != nullptr && captureReader != nullptr){
			captureReader->pause();
		}
		enableButtons();
	});
	connect(ui->setting,&QPushButton::clicked,[&](bool){
		if(onnxLoader != nullptr){
			onnxLoader->config(this);
		}
	});
	enableButtons();
}

MainWindow::~MainWindow() {
	removeCaptureReader();
	if(onnxLoader != nullptr){
		delete onnxLoader;
		onnxLoader = nullptr;
	}
	delete ui;
}

void MainWindow::removeCaptureReader(){
	player->setCaptureReader(nullptr);
	if(captureReader != nullptr) {
		delete captureReader;
		captureReader = nullptr;
	}
}

void MainWindow::enableButtons(){
	ui->play->setEnabled(onnxLoader != nullptr && captureReader != nullptr &&
						 captureReader->isRunning() && !captureReader->isPlay());
	ui->stop->setEnabled(captureReader != nullptr && captureReader->isPlay());
	ui->setting->setEnabled(onnxLoader != nullptr);
}
