#ifndef QVIDEOPLAYER_H
#define QVIDEOPLAYER_H

#include <mutex>
#include <QObject>
#include <QRect>
#include <QOpenGLWidget>
#include "QCaptureReader.h"


class QVideoPlayer : public QOpenGLWidget {
	Q_OBJECT
public:
	QVideoPlayer(QWidget *parent = nullptr);
	~QVideoPlayer();
	void setCaptureReader(QCaptureReader * rdr);
protected:
	virtual void resizeEvent(QResizeEvent *event)override;
	virtual void paintEvent(QPaintEvent *event) override;
private:
	std::mutex mtx;
	QCaptureReader *reader;
	QRect imageBounds;;
};

#endif // QVIDEOPLAYER_H
