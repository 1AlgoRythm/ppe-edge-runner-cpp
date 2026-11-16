// Small checks of the pure C++ pieces. Exit code 0 = pass.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "yolo.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main() {
  // Letterbox of a 1280x720 (WxH) image: scale 0.5 -> 640x360, 140 px grey bars top and bottom.
  cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
  Letterboxed lb = letterbox(img, 640);
  CHECK(lb.img.rows == 640 && lb.img.cols == 640);
  CHECK(lb.scale == 0.5 && lb.pad_x == 0 && lb.pad_y == 140);
  CHECK(lb.img.at<cv::Vec3b>(10, 10) == cv::Vec3b(114, 114, 114));
  CHECK(lb.img.at<cv::Vec3b>(320, 320) == cv::Vec3b(10, 20, 30));

  // to_tensor: BGR (10,20,30) -> planes R=30/255, G=20/255, B=10/255.
  std::vector<float> t(3 * 640 * 640);
  to_tensor(lb.img, t.data());
  CHECK(t[320 * 640 + 320] == 30 / 255.0f);
  CHECK(t[640 * 640 + 320 * 640 + 320] == 20 / 255.0f);
  CHECK(t[2 * 640 * 640 + 320 * 640 + 320] == 10 / 255.0f);

  // NMS: two overlapping same-class boxes -> keep the better one; a different class survives; a far box survives.
  std::vector<cv::Rect2f> b{{0, 0, 100, 100}, {5, 5, 100, 100}, {5, 5, 100, 100}, {300, 300, 50, 50}};
  std::vector<float> s{0.6f, 0.9f, 0.8f, 0.7f};
  std::vector<int> c{0, 0, 1, 0};
  auto keep = nms(b, s, c, 0.45f);
  CHECK(keep.size() == 3 && keep[0] == 1 && keep[1] == 2 && keep[2] == 3);

  // decode: one anchor out of three above threshold, centred at (320,320) size 100 in a 640 letterbox of the image above.
  const int A = 3, nc = 1;
  std::vector<float> o((4 + nc) * A, 0.f);
  o[0 * A + 1] = 320; o[1 * A + 1] = 320; o[2 * A + 1] = 100; o[3 * A + 1] = 100; o[4 * A + 1] = 0.9f;
  o[4 * A + 0] = 0.1f;
  auto d = decode(o.data(), nc, A, lb, 1280, 720, 0.25f, 0.45f);
  CHECK(d.size() == 1);
  if (d.size() == 1) {   // x1 = (320-50-0)/0.5 = 540, y1 = (320-50-140)/0.5 = 260, w = 200
    CHECK(std::abs(d[0].x1 - 540) < 1e-3 && std::abs(d[0].y1 - 260) < 1e-3);
    CHECK(std::abs(d[0].x2 - 740) < 1e-3 && std::abs(d[0].y2 - 460) < 1e-3);
  }
  std::printf(failures ? "unit tests FAILED (%d)\n" : "unit tests passed\n", failures);
  return failures ? 1 : 0;
}
