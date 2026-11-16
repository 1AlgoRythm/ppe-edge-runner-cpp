#include "yolo.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

Letterboxed letterbox(const cv::Mat& bgr, int size) {
  const int h = bgr.rows, w = bgr.cols;                       // (H, W, 3)
  const double scale = std::min(static_cast<double>(size) / h, static_cast<double>(size) / w);
  const int nh = static_cast<int>(std::nearbyint(h * scale));
  const int nw = static_cast<int>(std::nearbyint(w * scale));
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
  const double pad_x = (size - nw) / 2.0, pad_y = (size - nh) / 2.0;
  const int top = static_cast<int>(std::nearbyint(pad_y - 0.1));
  const int left = static_cast<int>(std::nearbyint(pad_x - 0.1));
  const int bottom = size - nh - top, right = size - nw - left;
  Letterboxed out;
  cv::copyMakeBorder(resized, out.img, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));                // (size, size, 3)
  out.scale = scale;
  out.pad_x = left;
  out.pad_y = top;
  return out;
}

void to_tensor(const cv::Mat& lb, float* dst) {
  const int H = lb.rows, W = lb.cols;
  const size_t plane = static_cast<size_t>(H) * W;
  float* r = dst;              // CHW: three planes, R then G then B
  float* g = dst + plane;
  float* b = dst + 2 * plane;
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = lb.ptr<uint8_t>(y);                    // interleaved B,G,R
    const size_t o = static_cast<size_t>(y) * W;
    for (int x = 0; x < W; ++x) {
      b[o + x] = row[3 * x + 0] / 255.0f;                       // divide (not multiply by 1/255) to match numpy bit for bit
      g[o + x] = row[3 * x + 1] / 255.0f;
      r[o + x] = row[3 * x + 2] / 255.0f;
    }
  }
}

static float iou_xywh(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float ix = std::max(0.f, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
  const float iy = std::max(0.f, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
  const float inter = ix * iy;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     const std::vector<int>& classes, float iou_thr) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  std::vector<int> keep;
  for (int i : order) {
    bool suppressed = false;
    for (int k : keep)
      if (classes[k] == classes[i] && iou_xywh(boxes[k], boxes[i]) > iou_thr) { suppressed = true; break; }
    if (!suppressed) keep.push_back(i);
  }
  return keep;
}

std::vector<Det> decode(const float* out, int nc, int n_anchors, const Letterboxed& lb,
                        int orig_w, int orig_h, float conf, float iou) {
  // out is (4 + nc, n_anchors): row r, anchor a -> out[r * n_anchors + a].
  std::vector<cv::Rect2f> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  const float scale = static_cast<float>(lb.scale);
  for (int a = 0; a < n_anchors; ++a) {
    int best = 0;                                               // argmax over the nc class rows
    float s = out[4 * n_anchors + a];
    for (int c = 1; c < nc; ++c) {
      const float v = out[(4 + c) * n_anchors + a];
      if (v > s) { s = v; best = c; }
    }
    if (s < conf) continue;
    const float cx = out[0 * n_anchors + a], cy = out[1 * n_anchors + a];
    const float w = out[2 * n_anchors + a], h = out[3 * n_anchors + a];
    const float x1 = (cx - w / 2 - lb.pad_x) / scale;           // undo letterbox: remove padding, divide by scale
    const float y1 = (cy - h / 2 - lb.pad_y) / scale;
    boxes.emplace_back(x1, y1, w / scale, h / scale);
    scores.push_back(s);
    classes.push_back(best);
  }
  std::vector<Det> dets;
  for (int i : nms(boxes, scores, classes, iou)) {
    const cv::Rect2f& b = boxes[i];
    dets.push_back({std::max(0.f, b.x), std::max(0.f, b.y),
                    std::min(static_cast<float>(orig_w), b.x + b.width),
                    std::min(static_cast<float>(orig_h), b.y + b.height), scores[i], classes[i]});
  }
  return dets;
}
