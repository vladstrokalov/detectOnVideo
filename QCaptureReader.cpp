#include <future>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudafilters.hpp>
#include "QCaptureReader.h"

using namespace std;
QCaptureReader::QCaptureReader(QObject *parent)
	: QObject{parent}{
}
QCaptureReader::~QCaptureReader(){
	stop();
}
void QCaptureReader::stop(){
	running.store(false);
	if(player.joinable()){
		player.join();
	}
	playback.store(false);
	std::lock_guard<std::mutex> lock(mtx);
	playerInited = false;
}
bool QCaptureReader::open(const QString& fileName){
	stop();
	this->fileName = fileName;
	string fName = fileName.toStdString();
	std::packaged_task<int()> task([this,fName](){
		 cv::VideoCapture cap(fName);
		{
			std::lock_guard<std::mutex> lock(mtx);
			playerInited = true;
		}
		running.store(cap.isOpened());
		cv.notify_one();
		if(!cap.isOpened()){
			return 0;
		}
		cv::Mat frame;
		cap >> frame;
		fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0.0) fps = 30.0;
		int delayMs = static_cast<int>(1000.0 / fps);
		while(running.load() && !frame.empty()){
			if(!playback.load()){
				std::this_thread::sleep_for(std::chrono::milliseconds(30));
				continue;
			}
			processing(frame);
			std::this_thread::yield();
			std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
			cap >> frame;
		}
		running.store(false);
		playback.store(false);
		if(videoWriter.isOpened()){
			videoWriter.release();
		}
		emit onEof();
		return 0;
	});
	player = std::thread(std::move(task));
	{
		std::unique_lock<std::mutex> lock(mtx);
		cv.wait(lock, [this]() { return playerInited; });
	}
	return running.load();
}

void QCaptureReader::putSharedFrame(const cv::Mat& mat){
	std::lock_guard<std::mutex> lock(mtxSwap);
	sharedFrame = cv::Mat();
	if(!mat.empty()){
		mat.copyTo(sharedFrame);
	}
}

void QCaptureReader::getSharedFrame(cv::Mat& mat){
	std::lock_guard<std::mutex> lock(mtxSwap);
	if(!sharedFrame.empty()){
		sharedFrame.copyTo(mat);
	}
}

bool QCaptureReader::QimageToMat(const QImage& src, cv::Mat& dest) {
	switch (src.format()) {
		// 8-bit, 4 channel
	case QImage::Format_ARGB32:
	case QImage::Format_ARGB32_Premultiplied: {
		dest = cv::Mat(src.height(), src.width(), CV_8UC4, const_cast<uchar*>(src.bits()), static_cast<size_t>(src.bytesPerLine())).clone();
	}
	break;
		// 8-bit, 3 channel
	case QImage::Format_RGB32: {
		cv::Mat mat(src.height(), src.width(), CV_8UC4, const_cast<uchar*>(src.bits()), static_cast<size_t>(src.bytesPerLine()));
		cv::cvtColor(mat, dest, cv::COLOR_BGRA2BGR);   // drop the all-white alpha channel
	}
	break;
		// 8-bit, 3 channel
	case QImage::Format_RGB888: {
		QImage swapped = src.rgbSwapped();
		dest = cv::Mat(swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()),
					   static_cast<size_t>(swapped.bytesPerLine())).clone();
	}
	break;
	case QImage::Format_RGBX8888: {
		QImage swapped = src.rgbSwapped();
		dest = cv::Mat(swapped.height(), swapped.width(), CV_8UC4, const_cast<uchar*>(swapped.bits()),
					   static_cast<size_t>(swapped.bytesPerLine())).clone();
	}
	break;
		// 8-bit, 1 channel
	case QImage::Format_Indexed8: {
		dest = cv::Mat(src.height(), src.width(), CV_8UC1, const_cast<uchar*>(src.bits()), static_cast<size_t>(src.bytesPerLine())).clone();
	}
	break;
	default: return false;
	}
	return true;
}
bool QCaptureReader::MatToQImage(const cv::Mat& src, QImage& dest) {
	switch (src.type()) {
		// 8-bit, 4 channel
	case CV_8UC4: {
		dest = QImage(src.cols, src.rows, QImage::Format_ARGB32);
		memcpy(dest.bits(), src.data, src.cols * src.rows * 4);
	}
	break;
		// 8-bit, 3 channel
	case CV_8UC3: {
		dest = QImage(src.data, src.cols, src.rows, static_cast<int>(src.step), QImage::Format_RGB888).rgbSwapped();
	}
	break;
		// 8-bit, 1 channel
	case CV_8UC1: {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
		dest = QImage(src.data, src.cols, src.rows, static_cast<int>(src.step), QImage::Format_Grayscale8);
#else
		static QVector<QRgb>  sColorTable;
		// only create our color table the first time
		if (sColorTable.isEmpty()) {
			sColorTable.resize(256);
			for (int i = 0; i < 256; ++i) {
				sColorTable[i] = qRgb(i, i, i);
			}
		}
		dest = QImage(src.data, src.cols, src.rows, static_cast<int>(src.step), QImage::Format_Indexed8);
		dest.setColorTable(sColorTable);
#endif
	}
	break;
	default: return false;
	}
	return true;
}

bool QCaptureReader::currentFrame(QImage& dest){
	cv::Mat frame;
	getSharedFrame(frame);
	if(frame.empty()){
		return false;
	}
	MatToQImage(frame,dest);
	return true;
}

void QCaptureReader::applyCLAHE(const cv::Mat& frame,cv::Mat& out){
	cv::cuda::GpuMat gpuFrame, labGpu, blurred,hsv;
	gpuFrame.upload(frame);

	cv::cuda::cvtColor(gpuFrame, labGpu, cv::COLOR_RGB2Lab);

	std::vector<cv::cuda::GpuMat> labChannels,hsvChannels;
	cv::cuda::split(labGpu, labChannels);

	cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(4.0);
	clahe->apply(labChannels[0], labChannels[0]);

	cv::cuda::merge(labChannels, labGpu);

	cv::cuda::cvtColor(labGpu, gpuFrame, cv::COLOR_Lab2RGB);

	cv::Ptr<cv::cuda::Filter> gaussFilter =
		cv::cuda::createGaussianFilter(CV_8UC3,CV_8UC3,cv::Size(5, 5),1.5);
	gaussFilter->apply(gpuFrame, blurred);
	blurred.download(out);
}
void QCaptureReader::processing(cv::Mat &frame){
	if(onnx){
		cv::Mat pframe;
		applyCLAHE(frame,pframe);
		std::vector<DetectionClass> detection;
		onnx->detect(pframe,detection);
		cv::Scalar color = cv::Scalar(0, 255, 255);
		for(auto d : detection){
			cv::rectangle(frame, d.rect, color, 2);
			std::string label = "Class " + std::to_string(d.classId) + "(" + d.className + ") : " +
								std::to_string(d.scope);
			cv::putText(frame, label, cv::Point(d.rect.x, d.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
		}
		if(onnx->isSaveResult()){
			if(!videoWriter.isOpened()){
				QFileInfo fileInfo(fileName);
				QString fileOut = onnx->getResultFolder() + QDir::separator() + fileInfo.baseName()+"_out.mp4";
				if (QFile::exists(fileOut)) {
					QFile::remove(fileOut);
				}
				videoWriter = cv::VideoWriter();
				int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // Альтернатива: 'H', '2', '6', '4'
				videoWriter.open(fileOut.toStdString().c_str(), fourcc, fps, cv::Size(frame.cols,frame.rows));
			}
			if(videoWriter.isOpened()){
				videoWriter.write(frame);
			}
		}
	}
	putSharedFrame(frame);
}
