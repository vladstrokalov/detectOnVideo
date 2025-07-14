#pragma once
#ifndef QONNXLOADER_H
#define QONNXLOADER_H
#include <QWidget>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <QObject>
#include <QQuickView>
#include <QSettings>
#include <QFileDialog>
struct DetectionClass {
	cv::Rect		rect;
	float		scope;
	int			classId;
	std::string className;
	DetectionClass(){}
	DetectionClass(cv::Rect r,float s,int c,const std::string &n):
		rect(r),scope(s),classId(c),className(n){}
};

class QONNXLoader : public QObject {
	Q_OBJECT
	Q_PROPERTY(QString cudaDeviceNumber READ cudaDeviceNumber CONSTANT)
	Q_PROPERTY(QString deviceID READ deviceID CONSTANT)
	Q_PROPERTY(QString deviceName READ deviceName CONSTANT)
	Q_PROPERTY(QString globalMem READ globalMem CONSTANT)
	Q_PROPERTY(QString providers READ getProviders CONSTANT)
	Q_PROPERTY(QString inputShapeInfo READ getInputShapeInfo CONSTANT)
	Q_PROPERTY(QString outputShapeInfo READ getOutputShapeInfo CONSTANT)
	Q_PROPERTY(float confThreshold READ getConfThreshold WRITE setConfThreshold )
	Q_PROPERTY(float nmsThreshold READ getNMSThreshold WRITE setNMSThreshold )
	Q_PROPERTY(bool saveResult READ isSaveResult WRITE setSsaveResult)
	Q_PROPERTY(QString resultFolder READ getResultFolder WRITE setResultFolder NOTIFY resultFolderChanged)
public:
	explicit QONNXLoader(QSettings* _settings,QObject *parent = nullptr);
	virtual ~QONNXLoader();
	QString cudaDeviceNumber()const {return QString::number(cuda_devices_number);}
	QString deviceID()const{return QString::number(deviceInfo.deviceID());}
	QString deviceName()const{return deviceInfo.isCompatible() ? deviceInfo.name() : "недоступно";}
	QString globalMem()const{return deviceInfo.isCompatible() ? QString::number(deviceInfo.totalGlobalMem()/1024/1024) + "Gb" : "нет";}
	QString getProviders()const {
		QString res = "";
		for(auto it : providers){
			if(!res.isEmpty()) {
				res += "\n";
			}
			res += it;
		}
		return res;
	}
	QString getInputShapeInfo()const{return input_shape_info;}
	QString getOutputShapeInfo()const{return output_shape_info;}
	bool load(const QString& fileName,QWidget *parent);
	void info(QWidget *parent);
	Q_INVOKABLE void closeWindowInfo() {
		if (viewInfo)
			viewInfo->close();
	}
	Q_INVOKABLE void selectResultFolder(){
		QString dir = QFileDialog::getExistingDirectory(nullptr, "Каталог результатов",
			resultFolder);
		if(!dir.isEmpty()){
			setResultFolder(dir);
		}
	}
	float getConfThreshold()const{
		return conf_threshold;
	}
	void setConfThreshold(float val){
		conf_threshold = val;
		if(settings){
			settings->setValue(conf_thresholdKey,conf_threshold);
		}
	}
	float getNMSThreshold()const{return nms_threshold;}
	void setNMSThreshold(float val){
		nms_threshold = val;
		if(settings){
			settings->setValue(nms_thresholdKey,nms_threshold);
		}
	}
	bool isSaveResult(){
		if(settings){
			return settings->value(saveResultKey,false).toBool();
		}
		return false;
	}
	void setSsaveResult(bool val){
		save_result = val;
		if(settings){
			settings->setValue(saveResultKey,val);
		}
	}
	QString getResultFolder() {
		return resultFolder;
	}
	void setResultFolder(const QString& val){
		resultFolder = val;
		if(settings){
			settings->setValue(resultFolderKey,val);
		}
		emit resultFolderChanged();
	}
	void config(QWidget *parent);
	void detect(const cv::Mat& mat, std::vector<DetectionClass>& detection);
signals:
	 void resultFolderChanged();
private:
	void closeInfo();
private:
	QSettings *settings{nullptr};
	int cuda_devices_number{0};
	cv::cuda::DeviceInfo deviceInfo;
	Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"copa_onnx_loader"};
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo memoryInfo{nullptr};
	std::shared_ptr<Ort::Session> session{nullptr};
	std::map<int,std::string> labels;
	std::vector<std::string> providers;
	QQuickView *viewInfo{nullptr};
	QString input_shape_info{"[]"};
	QString output_shape_info{"[]"};
	cv::Size inputSize{0,0};
	int version{0};
	float conf_threshold { 0.3f};
	float nms_threshold { 0.4f };
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> classesid;
	bool save_result{false};
	QString resultFolder{""};
private:
	const char *conf_thresholdKey = "conf_threshold";
	const char *nms_thresholdKey = "nms_threshold";
	const char *saveResultKey = "save_result";
	const char *resultFolderKey = "result_folder";
};

#endif // QONNXLOADER_H
