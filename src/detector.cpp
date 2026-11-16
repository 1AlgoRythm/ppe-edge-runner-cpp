#include "detector.hpp"

#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

Detector::Detector(const std::string& model_path, std::vector<std::string> class_names, int threads, int size,
                   const std::vector<std::pair<std::string, std::string>>& ort_config)
    : path_(model_path), names_(std::move(class_names)), size_(size),
      env_(ORT_LOGGING_LEVEL_WARNING, "ppe_runner"),
      input_(static_cast<size_t>(3) * size * size) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(threads);                       // 0 = ORT default
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  for (const auto& kv : ort_config) opts.AddConfigEntry(kv.first.c_str(), kv.second.c_str());
  session_ = Ort::Session(env_, model_path.c_str(), opts);  // CPU EP is the default; no other EP is appended

  Ort::AllocatorWithDefaultOptions alloc;
  in_name_ = session_.GetInputNameAllocated(0, alloc).get();
  out_name_ = session_.GetOutputNameAllocated(0, alloc).get();

  Ort::TypeInfo in_type = session_.GetInputTypeInfo(0);   // keep alive: the shape info below only borrows from it
  auto in_info = in_type.GetTensorTypeAndShapeInfo();
  if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error(model_path + ": only float32 inputs are supported (fp16 models are not)");
  Ort::TypeInfo out_type = session_.GetOutputTypeInfo(0);
  out_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
  // YOLOv8 head: (1, 4 + nc, 8400) at 640x640. Fail loudly if a model does not look like that.
  if (out_shape_.size() != 3 || out_shape_[0] != 1 || out_shape_[1] != 4 + static_cast<int64_t>(names_.size()))
    throw std::runtime_error(model_path + ": unexpected output shape; expected (1, 4+" +
                             std::to_string(names_.size()) + ", N)");
}

std::vector<Det> Detector::detect(const cv::Mat& bgr, float conf, float iou, StageMs* t, const std::string& save_lb_path) {
  auto t0 = Clock::now();
  Letterboxed lb = letterbox(bgr, size_);                   // (640, 640, 3) uint8
  to_tensor(lb.img, input_.data());                         // (1, 3, 640, 640) float32
  const double pre = ms_since(t0);
  if (!save_lb_path.empty()) cv::imwrite(save_lb_path, lb.img);

  t0 = Clock::now();
  const std::array<int64_t, 4> in_shape{1, 3, size_, size_};
  Ort::Value in = Ort::Value::CreateTensor<float>(mem_, input_.data(), input_.size(), in_shape.data(), in_shape.size());
  const char* in_names[] = {in_name_.c_str()};
  const char* out_names[] = {out_name_.c_str()};
  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
  const double inf = ms_since(t0);

  t0 = Clock::now();
  const std::vector<int64_t> shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();   // (1, 4 + nc, 8400)
  const float* data = outs[0].GetTensorData<float>();
  std::vector<Det> dets = decode(data, static_cast<int>(shape[1]) - 4, static_cast<int>(shape[2]), lb,
                                 bgr.cols, bgr.rows, conf, iou);
  const double post = ms_since(t0);
  if (t) { t->pre = pre; t->inf = inf; t->post = post; }
  return dets;
}
