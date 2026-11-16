// Pure-C++ pieces of a YOLOv8 pipeline: letterbox, tensor packing, decode + NMS.
// No ONNX Runtime in here, so it can be unit-tested on its own.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct Det {
  float x1, y1, x2, y2;  // corners in ORIGINAL image pixels
  float score;
  int cls;
};

struct Letterboxed {
  cv::Mat img;  // (size, size, 3) uint8 BGR, grey padded
  double scale; // original -> letterboxed scale factor
  int pad_x;    // left padding in letterboxed pixels
  int pad_y;    // top padding
};

// Resize keeping aspect ratio, pad to size x size with grey 114. Mirrors the Python reference exactly,
// including Python's round-half-to-even (std::nearbyint under the default rounding mode).
Letterboxed letterbox(const cv::Mat& bgr, int size = 640);

// BGR HWC uint8 (size,size,3) -> RGB CHW float32 in [0,1], written to dst of length 3*size*size.
void to_tensor(const cv::Mat& lb, float* dst);

// Class-aware greedy NMS. boxes are x,y,w,h; returns indices of kept boxes, highest score first.
std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     const std::vector<int>& classes, float iou_thr);

// YOLOv8 head output (nc+4, n_anchors), row-major as ORT returns it (batch dim already stripped):
//   rows 0..3 = cx, cy, w, h in letterboxed pixels; rows 4.. = per-class sigmoid scores.
std::vector<Det> decode(const float* out, int nc, int n_anchors, const Letterboxed& lb,
                        int orig_w, int orig_h, float conf, float iou);
