// Single-model and two-stage (person -> crop -> PPE) detection on one frame.
#pragma once
#include <memory>

#include "detector.hpp"

struct FrameResult {
  std::vector<Det> persons;   // empty in single-model PPE mode
  std::vector<Det> items;     // single mode: the model's detections; cascade: PPE boxes in full-image coords
  StageMs person_t, ppe_t;    // stage times of the person model and the (summed) PPE crops
  double total_ms = 0;        // wall time of the whole process() call, crops included
  int n_crops = 0;
};

struct Pipeline {
  std::unique_ptr<Detector> person;   // may be null (PPE-only single mode)
  std::unique_ptr<Detector> ppe;      // may be null (person-only single mode)
  bool cascade = false;
  float conf = 0.25f, iou = 0.45f;    // same thresholds as the Python reference
  // Test hook: when save_lb_dir is set, every letterboxed network input is saved as <frame_key>_person.png / _ppe_<k>.png.
  std::string save_lb_dir;
  FrameResult process(const cv::Mat& bgr, const std::string& frame_key = "");
};

const std::vector<std::string>& person_names();
const std::vector<std::string>& ppe_names();
