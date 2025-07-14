#pragma once
#ifndef QCAPTUREREADER_H
#define QCAPTUREREADER_H
#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <QObject>
#include <QImage>
#include "QONNXLoader.h"

class QCaptureReader : public QObject {
	Q_OBJECT
public:
	explicit QCaptureReader(QObject *parent = nullptr);
	virtual ~QCaptureReader();
	void stop();
	bool open(const QString& fileName);
	bool isRunning(){return running.load();}
	void play(){playback.store(running.load());}
	void pause(){playback.store(false);}
	bool isPlay(){return playback.load();}
	bool QimageToMat(const QImage& src, cv::Mat& dest);
	bool MatToQImage(const cv::Mat& src, QImage& dest);
	bool currentFrame(QImage& dest);
	void setONNX(QONNXLoader *loader){
		std::lock_guard<std::mutex> lock(mtxONNX);
		onnx = loader;
		if(onnx){
			connect(onnx, &QObject::destroyed, [&](QObject* o){
				std::lock_guard<std::mutex> lock(mtxONNX);
				onnx = nullptr;
			});
		}
	}
signals:
	void onEof();
private:
	void processing(cv::Mat &frame);
	void putSharedFrame(const cv::Mat& mat);
	void getSharedFrame(cv::Mat& mat);
	void applyCLAHE(const cv::Mat& frame,cv::Mat& out);
private:
	std::atomic<bool> running{false};
	std::thread player;
	std::mutex mtx;
	std::mutex mtxSwap;
	std::mutex mtxONNX;
	std::condition_variable cv;
	bool playerInited = false;
	std::atomic<bool> playback{false};
	cv::Mat sharedFrame;
	QONNXLoader *onnx{nullptr};
	cv::VideoWriter videoWriter;
	double fps{30};
	QString fileName{""};
};

#endif // QCAPTUREREADER_H
