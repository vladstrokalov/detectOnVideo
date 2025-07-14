#include <QQmlContext>
#include <QQuickView>
#include <QMessageBox>
#include "QONNXLoader.h"
#if ORT_API_VERSION < 20
#include <cuda_provider_factory.h>
#endif
using namespace std;
using namespace cv;

QONNXLoader::QONNXLoader(QSettings* _settings,QObject *parent)
	: QObject{parent}{
	cuda_devices_number = cv::cuda::getCudaEnabledDeviceCount();
	providers = Ort::GetAvailableProviders();
	bool pCUDA = false;
	for (const auto& p : providers) {
		if(p.find("CUDA") != std::string::npos) {
			pCUDA = true;
		}
	}
	cuda_devices_number = (cuda_devices_number != 0 && pCUDA) ? cuda_devices_number : 0;
	settings = _settings;
	if(settings){
		conf_threshold = settings->value(conf_thresholdKey,conf_threshold).toFloat();
		nms_threshold = settings->value(nms_thresholdKey,nms_threshold).toFloat();
		save_result = settings->value(saveResultKey,false).toBool();
		resultFolder = settings->value(resultFolderKey, QDir::homePath()).toString();
	}
}

QONNXLoader::~QONNXLoader(){
	closeInfo();
}
void QONNXLoader::closeInfo(){
	if(viewInfo != nullptr && viewInfo->isVisible()){
		viewInfo->close();
		delete viewInfo;
	}
	viewInfo = nullptr;
}
void QONNXLoader::info(QWidget *parent){
	closeInfo();
	viewInfo = new QQuickView();
	viewInfo->setResizeMode(QQuickView::SizeRootObjectToView);
	viewInfo->rootContext()->setContextProperty("onnxloaderModel", this);
	viewInfo->setFlags(Qt::FramelessWindowHint | Qt::Window | Qt::WindowStaysOnTopHint);
	viewInfo->setModality(Qt::ApplicationModal);
	viewInfo->setSource(QUrl("qrc:/qml/ONNXLoader.qml"));
	viewInfo->setColor(Qt::transparent);
	viewInfo->show();
}
void QONNXLoader::config(QWidget *parent){
	closeInfo();
	viewInfo = new QQuickView();
	viewInfo->setResizeMode(QQuickView::SizeRootObjectToView);
	viewInfo->rootContext()->setContextProperty("onnxloaderModel", this);
	viewInfo->setFlags(Qt::FramelessWindowHint | Qt::Window | Qt::WindowStaysOnTopHint);
	viewInfo->setModality(Qt::ApplicationModal);
	viewInfo->setSource(QUrl("qrc:/qml/Setting.qml"));
	viewInfo->setColor(Qt::transparent);
	viewInfo->show();
}

bool QONNXLoader::load(const QString& fileName,QWidget *parent){
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "copa_drone_onnx");
	Ort::SessionOptions session_options;
	OrtCUDAProviderOptions cuda_options;
	memset(&cuda_options,0,sizeof(OrtCUDAProviderOptions));
	cuda_options.arena_extend_strategy = 1;
	cuda_options.device_id = 0;
	if(!cuda_devices_number) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
	}
	session_options.AppendExecutionProvider_CUDA(cuda_options);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	try {
		session = std::make_shared<Ort::Session>(env,fileName.toStdString().c_str(), session_options);
	} catch (const Ort::Exception& e) {
		QString err = e.what();
		QString errorMsg = QString("Ошибка загрузки модели детекции %1 (%2)").arg(fileName, err);
		QMessageBox::critical(parent,"Ошибка",errorMsg);
		return false;
	}


	auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	input_shape_info = "[";
	for(auto inf : input_shape){
		if(input_shape_info.size() != 1) {
			input_shape_info += ",";
		}
		input_shape_info += std::to_string(inf);
	}
	input_shape_info += "]";
	inputSize = cv::Size(input_shape[2],input_shape[3]);
	auto output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	if(output_shape.size() == 3 && output_shape[0] == 1 &&
		(output_shape[2] == 84 || output_shape[2] == 85 || output_shape[2] == 6)){
		output_shape_info = "[";
		for(auto inf : output_shape){
			if(output_shape_info.size() != 1) {
				output_shape_info += ",";
			}
			output_shape_info += std::to_string(inf);
		}
		output_shape_info += "]";
		if(output_shape[1] == 82500){
			version = 5;
		}
		else {
			version = 8;
		}
		memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		std::set<std::string> yolo_hints = {"output0", "detect", "yolo", "output", "prob"};
		size_t num_outputs = session->GetOutputCount();
		for (size_t i = 0; i < num_outputs; ++i) {
#if ORT_API_VERSION >= 10
			auto output_name_alloc = session->GetOutputNameAllocated(0, allocator);
			const char* name = output_name_alloc.get();
			std::string out_name = {name};
#else
			const char *name = session->GetOutputName(0, allocator);
			string out_name = {name};
#endif
			for (const auto& hint : yolo_hints) {
				if (out_name.find(hint) != std::string::npos) {
					return true;
				}
			}
		}
		return false;
	}
	QMessageBox::critical(parent,"Ошибка","Выбранная сеть имеет структуру отличную от yolo!");
	return false;
}

void QONNXLoader::detect(const cv::Mat& frame, std::vector<DetectionClass>& detection){
	float scale_x = static_cast<float>(inputSize.width) / static_cast<float>(frame.cols);
	float scale_y = static_cast<float>(inputSize.height) / static_cast<float>(frame.rows);
	float scale = std::min(scale_x,scale_y);
	int resized_w;
	int resized_h;
	if(frame.cols <= inputSize.width && frame.rows <= inputSize.height){
		scale = 1.0f;
		resized_w = frame.cols;
		resized_h = frame.rows;
	}
	else {
		resized_w = int(frame.cols * scale);
		resized_h = int(frame.rows * scale);
	}
	int pad_x = inputSize.width - resized_w;
	int pad_y = inputSize.height - resized_h;

	int pad_left   = pad_x / 2;
	int pad_right  = pad_x - pad_left;
	int pad_top    = pad_y / 2;
	int pad_bottom = pad_y - pad_top;
	cv::Mat resize_frame,input_frame;
	if(frame.cols <= inputSize.width && frame.rows <= inputSize.height){
		cv::copyMakeBorder(frame, input_frame,
						   pad_top, pad_bottom, pad_left, pad_right,
						   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	else {
		cv::resize(frame, resize_frame, cv::Size(resized_w, resized_h));
		cv::copyMakeBorder(resize_frame, input_frame,
						   pad_top, pad_bottom, pad_left, pad_right,
						   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}


	input_frame.convertTo(input_frame, CV_32F, 1.0 / 255);
	std::vector<float> input_tensor_values;
	std::vector<int64_t> input_dims = {1, 3, (int64_t)inputSize.width, (int64_t)inputSize.height};
	std::vector<cv::Mat> channels(3);
	cv::split(input_frame, channels);
	for (auto& c : channels)
		input_tensor_values.insert(input_tensor_values.end(), (float*)c.datastart, (float*)c.dataend);
	size_t input_tensor_size = input_tensor_values.size();
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memoryInfo, input_tensor_values.data(), input_tensor_size, input_dims.data(), input_dims.size());
#if ORT_API_VERSION >= 10
	auto input_name_alloc = session->GetInputNameAllocated(0, allocator);
	const char* input_name = input_name_alloc.get();
#else
	const char* input_name = session->GetInputName(0, allocator);
#endif
	std::vector<const char *> inputNames = {input_name};
	//
#if ORT_API_VERSION >= 10
	auto output_name_alloc = session->GetOutputNameAllocated(0, allocator);
	const char* output_name = output_name_alloc.get();
	std::vector<const char*> output_names_c = {output_name};
#else
	const char *output_names = session->GetOutputName(0, allocator);
	std::vector<const char*> output_names_c = {output_names};
#endif
	auto output_tensors = session->Run(
		Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, output_names_c.data(), 1);
	float* output_data = output_tensors[0].GetTensorMutableData<float>();
	auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

	int num_boxes = output_shape[1];
	int num_attrs = output_shape[2];
	if(boxes.capacity() < num_boxes){
		boxes.reserve(num_boxes);
		scores.reserve(num_boxes);
		classesid.reserve(num_boxes);
	}
	boxes.clear();
	scores.clear();
	classesid.clear();
	for (int i = 0; i < num_boxes; ++i) {
		float* det = output_data + i * num_attrs;

		float max_class_score = 0.0f;
		int class_id = -1;
		float final_score = 0.0f;
		if(version == 5){
			float objectness = det[4];
			if(objectness < conf_threshold){
				continue;
			}
			for (int j = 5; j < num_attrs; ++j) {
				float score = det[j];
				if (score > max_class_score) {
					max_class_score = score;
					class_id = j - 5;
				}
			}
			final_score = max_class_score * objectness;
		}
		else {
			if(version == 8){
				final_score = det[4];
				class_id = static_cast<int>(det[5]);
			}
		}

		if (final_score < conf_threshold)
			continue;
		int left = 0,top = 0,right = 0,bottom = 0;
		if(version == 5){
			float x_center = det[0];
			float y_center = det[1];
			float width = det[2];
			float height = det[3];
			float x1 = x_center - width / 2.0f;
			float y1 = y_center - height / 2.0f;
			float x2 = x_center + width / 2.0f;
			float y2 = y_center + height / 2.0f;
			left = static_cast<int>((x1 - pad_left) / scale);
			top = static_cast<int>((y1 - pad_top) / scale);
			right = static_cast<int>((x2- pad_left) / scale);
			bottom = static_cast<int>((y2 - pad_top) / scale);
		}
		else {
			float x1 = det[0];
			float y1 = det[1];
			float x2 = det[2];
			float y2 = det[3];
			left = static_cast<int>((x1 - pad_left) / scale);
			top = static_cast<int>((y1 - pad_top) / scale);
			right = static_cast<int>((x2 - pad_left) / scale);
			bottom = static_cast<int>((y2 - pad_top) / scale);
		}
		cv::Rect rect = cv::Rect(left,top,right - left,bottom - top);
		boxes.emplace_back(rect);
		scores.emplace_back(final_score);
		classesid.emplace_back(class_id);
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);
	for (int idx : indices) {
		const auto& box = boxes[idx];
		std::string className = "";
		detection.emplace_back(DetectionClass(box,classesid[idx],scores[idx],className));
	}
}
