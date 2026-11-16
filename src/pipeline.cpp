#include "pipeline.hpp"

#include <chrono>

const std::vector<std::string>& person_names() {
  static const std::vector<std::string> n{"person"};
  return n;
}
// ppe.pt has 7 classes (checked against the model metadata and ppe.yaml of the quantization lab).
const std::vector<std::string>& ppe_names() {
  static const std::vector<std::string> n{"hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit"};
  return n;
}

FrameResult Pipeline::process(const cv::Mat& bgr, const std::string& frame_key) {
  auto lb_path = [&](const std::string& stage) {
    return save_lb_dir.empty() ? std::string() : save_lb_dir + "/" + frame_key + "_" + stage + ".png";
  };
  const auto t0 = std::chrono::steady_clock::now();
  FrameResult r;
  if (!cascade) {
    Detector& d = person ? *person : *ppe;
    r.items = d.detect(bgr, conf, iou, &r.person_t, lb_path(person ? "person" : "ppe"));
    if (person) r.persons = r.items;
  } else {
    r.persons = person->detect(bgr, conf, iou, &r.person_t, lb_path("person"));
    for (const Det& p : r.persons) {
      const int x1 = static_cast<int>(p.x1), y1 = static_cast<int>(p.y1);   // truncate like int() in Python
      const int x2 = static_cast<int>(p.x2), y2 = static_cast<int>(p.y2);
      if (x2 - x1 < 10 || y2 - y1 < 10) continue;                            // skip slivers
      cv::Mat crop = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1));                // (h, w, 3) view, no copy
      StageMs t;
      for (Det d : ppe->detect(crop, conf, iou, &t, lb_path("ppe_" + std::to_string(r.n_crops)))) {
        d.x1 += x1; d.x2 += x1; d.y1 += y1; d.y2 += y1;                      // crop coords -> full image
        r.items.push_back(d);
      }
      r.ppe_t += t;
      ++r.n_crops;
    }
  }
  r.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return r;
}
