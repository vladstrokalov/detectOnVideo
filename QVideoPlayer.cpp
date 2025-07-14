#include <QPainter>
#include <QTimer>
#include <QImage>
#include "QVideoPlayer.h"
using namespace std;;
QVideoPlayer::QVideoPlayer(QWidget *parent):QOpenGLWidget(parent) {
	QTimer *timer = new QTimer(this);
	connect(timer, &QTimer::timeout, this, QOverload<>::of(&QWidget::update));
	timer->start(30);
}

QVideoPlayer::~QVideoPlayer(){

}

void QVideoPlayer::setCaptureReader(QCaptureReader * rdr){
	std::lock_guard<mutex> lock(mtx);
	reader = rdr;
}

void QVideoPlayer::resizeEvent(QResizeEvent *event){
	QOpenGLWidget::resizeEvent(event);
}
void QVideoPlayer::paintEvent(QPaintEvent *event){
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);
	painter.fillRect(rect(),QBrush(Qt::blue));
	QImage image;
	{
		std::lock_guard<mutex> lock(mtx);
		if(reader != nullptr){
			reader->currentFrame(image);
		}
	}
	if(image.isNull()){
		return;
	}
	QImage scaledImage = image.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	int x = (width() - scaledImage.width()) / 2;
	int y = (height() - scaledImage.height()) / 2;
	imageBounds = QRect(x,y,scaledImage.width(),scaledImage.height());
	painter.drawImage(x, y, scaledImage);
}
