// One YOLOv8 ONNX model on the ONNX Runtime CPU execution provider.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "yolo.hpp"

// Wall-clock milliseconds spent in each stage of one detect() call.
struct StageMs {
  double pre = 0, inf = 0, post = 0;
  StageMs& operator+=(const StageMs& o) { pre += o.pre; inf += o.inf; post += o.post; return *this; }
};

class Detector {
 public:
  // threads = 0 lets ONNX Runtime choose (all cores), same default as the Python reference.
  // ort_config: extra ONNX Runtime session config entries, e.g. {"mlas.disable_kleidiai", "1"}.
  Detector(const std::string& model_path, std::vector<std::string> class_names, int threads, int size = 640,
           const std::vector<std::pair<std::string, std::string>>& ort_config = {});
  // Letterbox -> tensor -> Run -> decode + NMS. Boxes come back in the coordinates of `bgr`.
  // If save_lb_path is non-empty the letterboxed network input is written there (lossless PNG) - used only by the
  // correctness test, to feed the Python reference the very same pixels. Never set in --bench.
  std::vector<Det> detect(const cv::Mat& bgr, float conf, float iou, StageMs* t = nullptr,
                          const std::string& save_lb_path = "");
  const std::vector<std::string>& names() const { return names_; }
  const std::string& path() const { return path_; }
  std::vector<int64_t> output_shape() const { return out_shape_; }

 private:
  std::string path_;
  std::vector<std::string> names_;
  int size_;
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::string in_name_, out_name_;
  std::vector<int64_t> out_shape_;   // (1, 4 + nc, n_anchors), read from the model, not assumed
  std::vector<float> input_;         // (1, 3, size, size) reused across calls
};
