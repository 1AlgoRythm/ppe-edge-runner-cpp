// ppe_runner: YOLOv8 ONNX inference on the CPU with ONNX Runtime, single model or person -> PPE cascade.
//   live / files : ppe_runner --variant fp32 --cascade --source data/person_test
//   benchmark    : ppe_runner --variant int8 --cascade --bench --out results/x.json
//   detections   : ppe_runner --variant fp32 --cascade --dump results/dets.json   (used by the correctness test)
#include <dlfcn.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>

#include "pipeline.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------- command line
struct Args {
  std::string variant, person_path, ppe_path, source, bench_out, dump_out, save_frame, save_lb, images = "data/person_test";
  bool cascade = false, bench = false, no_display = false;
  int threads = 0, warmup = 1, passes = 5, save_index = 0, max_frames = 0;
  float conf = 0.25f, iou = 0.45f;
  std::vector<std::pair<std::string, std::string>> ort_config;   // --ort-config KEY=VALUE (repeatable)
};

static void usage() {
  std::cout <<
    "usage: ppe_runner [model] [mode] [options]\n"
    "model:   --variant fp32|int8   use models/{person,ppe}_<variant>.onnx\n"
    "         --person FILE --ppe FILE   explicit model paths\n"
    "mode:    --cascade             person -> crop -> PPE (needs both models); default is the single person model\n"
    "         --source S            image directory, image file, video file, or webcam index (default: UI on --images)\n"
    "         --bench               headless timing over --images (default data/person_test)\n"
    "         --dump FILE           write detections of every --images image to JSON (no timing)\n"
    "options: --out FILE            bench result JSON\n"
    "         --warmup N --passes N bench passes over the image set (default 1 and 5)\n"
    "         --threads N           ORT intra-op threads, 0 = ORT default\n"
    "         --ort-config K=V      extra ONNX Runtime session config entry (repeatable), e.g. mlas.disable_kleidiai=1\n"
    "         --conf X --iou X      thresholds (0.25 / 0.45)\n"
    "         --save-letterbox DIR  with --dump: also save every letterboxed network input (test hook)\n"
    "         --no-display          do not open a window\n"
    "         --save-frame FILE [--save-index N]   write rendered frame N (default 0) as an image\n"
    "         --max-frames N        stop after N frames\n";
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (k == "--variant") a.variant = val();
    else if (k == "--person") a.person_path = val();
    else if (k == "--ppe") a.ppe_path = val();
    else if (k == "--cascade") a.cascade = true;
    else if (k == "--source") a.source = val();
    else if (k == "--images") a.images = val();
    else if (k == "--bench") a.bench = true;
    else if (k == "--out") a.bench_out = val();
    else if (k == "--dump") a.dump_out = val();
    else if (k == "--warmup") a.warmup = std::stoi(val());
    else if (k == "--passes") a.passes = std::stoi(val());
    else if (k == "--threads") a.threads = std::stoi(val());
    else if (k == "--conf") a.conf = std::stof(val());
    else if (k == "--iou") a.iou = std::stof(val());
    else if (k == "--no-display") a.no_display = true;
    else if (k == "--ort-config") {
      std::string kv = val();
      auto eq = kv.find('=');
      if (eq == std::string::npos) { std::cerr << "--ort-config needs KEY=VALUE\n"; std::exit(2); }
      a.ort_config.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    else if (k == "--save-letterbox") a.save_lb = val();
    else if (k == "--save-frame") a.save_frame = val();
    else if (k == "--save-index") a.save_index = std::stoi(val());
    else if (k == "--max-frames") a.max_frames = std::stoi(val());
    else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
    else { std::cerr << "unknown option " << k << "\n"; usage(); std::exit(2); }
  }
  return a;
}

// ---------------------------------------------------------------- helpers
static std::vector<fs::path> list_images(const std::string& dir) {
  std::vector<fs::path> v;
  for (auto& e : fs::directory_iterator(dir)) {
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") v.push_back(e.path());
  }
  std::sort(v.begin(), v.end());                                  // same order as sorted() in the Python reference
  return v;
}

static double percentile(std::vector<double> v, double p) {       // linear interpolation, like numpy's default
  std::sort(v.begin(), v.end());
  const double pos = (v.size() - 1) * p / 100.0;
  const size_t lo = static_cast<size_t>(std::floor(pos)), hi = static_cast<size_t>(std::ceil(pos));
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

static std::string stat_json(const std::vector<double>& v) {
  char b[200];
  std::snprintf(b, sizeof b, "{\"median\": %.4f, \"p95\": %.4f, \"mean\": %.4f, \"min\": %.4f}", percentile(v, 50), percentile(v, 95),
                std::accumulate(v.begin(), v.end(), 0.0) / v.size(), *std::min_element(v.begin(), v.end()));
  return b;
}

static std::string cpu_name() {
#ifdef __APPLE__
  char buf[128]; size_t n = sizeof buf;
  if (sysctlbyname("machdep.cpu.brand_string", buf, &n, nullptr, 0) == 0) return buf;
#endif
  return "unknown";
}

static uint64_t pixel_sum(const cv::Mat& m) {                      // cheap fingerprint of the decoded pixels
  uint64_t s = 0;
  for (int y = 0; y < m.rows; ++y) {
    const uint8_t* p = m.ptr<uint8_t>(y);
    for (int x = 0; x < m.cols * m.channels(); ++x) s += p[x];
  }
  return s;
}

// Path of the libonnxruntime that is actually loaded (Homebrew's or an override), via dladdr on its C entry point.
static std::string ort_library_path() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&OrtGetApiBase), &info) && info.dli_fname) {
    std::error_code ec;
    auto p = fs::canonical(info.dli_fname, ec);
    return ec ? std::string(info.dli_fname) : p.string();
  }
  return "unknown";
}

static std::string ort_config_str(const Args& a) {
  std::string s;
  for (auto& kv : a.ort_config) s += (s.empty() ? "" : ",") + kv.first + "=" + kv.second;
  return s;
}

static double g_load_before = -1;                                  // 1-minute load average when the timed loop started

static std::string meta_json(const Args& a, const Pipeline& p, const std::string& mode) {
  std::ostringstream o;
  o << "{\"lang\": \"cpp\", \"mode\": \"" << mode << "\", \"variant\": \"" << a.variant << "\""
    << ", \"person_model\": \"" << (p.person ? p.person->path() : "") << "\""
    << ", \"ppe_model\": \"" << (p.ppe ? p.ppe->path() : "") << "\""
    << ", \"conf\": " << a.conf << ", \"iou\": " << a.iou << ", \"threads\": " << a.threads << ", \"ort_config\": \"" << ort_config_str(a) << "\"" 
    << ", \"onnxruntime\": \"" << Ort::GetVersionString() << "\", \"ort_library\": \"" << ort_library_path() << "\", \"opencv\": \"" << CV_VERSION_STR << "\""
#ifdef NDEBUG
    << ", \"build\": \"Release\""
#else
    << ", \"build\": \"Debug\""
#endif
    << ", \"compiler\": \"" << __VERSION__ << "\", \"cpu\": \"" << cpu_name() << "\"";
  if (g_load_before >= 0) { double l[1]; getloadavg(l, 1); o << ", \"loadavg_before_after\": [" << g_load_before << ", " << l[0] << "]"; }
  o << "}";
  return o.str();
}

static const cv::Scalar kColors[] = {{0, 200, 0}, {0, 0, 255}, {255, 128, 0}, {0, 200, 255}, {255, 0, 200},
                                     {200, 200, 0}, {128, 0, 255}, {0, 128, 255}};

static void draw_box(cv::Mat& img, const Det& d, const std::string& label, const cv::Scalar& c) {
  cv::rectangle(img, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), c, 2);
  int base = 0;
  cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
  const int ly = std::max(ts.height + 4, static_cast<int>(d.y1));
  cv::rectangle(img, cv::Point(d.x1, ly - ts.height - 4), cv::Point(d.x1 + ts.width + 4, ly), c, cv::FILLED);
  cv::putText(img, label, cv::Point(d.x1 + 2, ly - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

static cv::Mat render(const cv::Mat& frame, const Pipeline& p, const FrameResult& r, double fps) {
  cv::Mat out = frame.clone();
  char b[96];
  if (p.cascade || p.person)
    for (const Det& d : r.persons) {
      std::snprintf(b, sizeof b, "person %.2f", d.score);
      draw_box(out, d, b, kColors[0]);
    }
  if (p.cascade || !p.person)
    for (const Det& d : r.items) {
      std::snprintf(b, sizeof b, "%s %.2f", ppe_names()[d.cls].c_str(), d.score);
      draw_box(out, d, b, kColors[1 + d.cls % 7]);
    }
  StageMs t = r.person_t; t += r.ppe_t;
  std::snprintf(b, sizeof b, "FPS %.1f | pre %.1f inf %.1f post %.1f ms", fps, t.pre, t.inf, t.post);
  cv::rectangle(out, cv::Point(0, 0), cv::Point(380, 24), cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(out, b, cv::Point(6, 17), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
  return out;
}

// ---------------------------------------------------------------- modes
static int run_bench(const Args& a, Pipeline& p, const std::string& mode) {
  std::vector<fs::path> files = list_images(a.images);
  if (files.empty()) { std::cerr << "no images in " << a.images << "\n"; return 1; }
  std::vector<cv::Mat> imgs;                                        // decode once, outside the timed region
  for (auto& f : files) imgs.push_back(cv::imread(f.string()));
  for (int w = 0; w < a.warmup; ++w) for (auto& im : imgs) p.process(im);

  std::vector<double> pre, inf, post, stage_sum, e2e, person_stage, ppe_stage, crops, dets;
  { double l[1]; if (getloadavg(l, 1) == 1) g_load_before = l[0]; }
  for (int pass = 0; pass < a.passes; ++pass)
    for (auto& im : imgs) {
      FrameResult r = p.process(im);
      StageMs t = r.person_t; t += r.ppe_t;
      pre.push_back(t.pre); inf.push_back(t.inf); post.push_back(t.post);
      stage_sum.push_back(t.pre + t.inf + t.post);
      e2e.push_back(r.total_ms);
      person_stage.push_back(r.person_t.pre + r.person_t.inf + r.person_t.post);
      ppe_stage.push_back(r.ppe_t.pre + r.ppe_t.inf + r.ppe_t.post);
      crops.push_back(r.n_crops); dets.push_back(r.items.size());
    }
  std::ofstream f(a.bench_out);
  if (!f) { std::cerr << "cannot write " << a.bench_out << "\n"; return 1; }
  f << "{\n \"meta\": " << meta_json(a, p, mode) << ",\n"
    << " \"n_images\": " << imgs.size() << ", \"warmup_passes\": " << a.warmup << ", \"timed_passes\": " << a.passes
    << ", \"n_timings\": " << e2e.size() << ",\n"
    << " \"note\": \"Per-image times in ms. In cascade mode each stage is summed over the person run and all PPE crops of that image. Image decoding is excluded.\",\n"
    << " \"preprocess_ms\": " << stat_json(pre) << ",\n \"inference_ms\": " << stat_json(inf) << ",\n"
    << " \"postprocess_ms\": " << stat_json(post) << ",\n \"stages_total_ms\": " << stat_json(stage_sum) << ",\n"
    << " \"e2e_ms\": " << stat_json(e2e) << ",\n \"person_stage_ms\": " << stat_json(person_stage) << ",\n"
    << " \"ppe_stage_ms\": " << stat_json(ppe_stage) << ",\n"
    << " \"crops_per_image_mean\": " << std::accumulate(crops.begin(), crops.end(), 0.0) / crops.size() << ",\n"
    << " \"dets_per_image_mean\": " << std::accumulate(dets.begin(), dets.end(), 0.0) / dets.size() << "\n}\n";
  std::printf("bench %s %s: e2e median %.2f ms, p95 %.2f ms (%zu timings) -> %s\n", mode.c_str(), a.variant.c_str(),
              percentile(e2e, 50), percentile(e2e, 95), e2e.size(), a.bench_out.c_str());
  return 0;
}

static void write_dets(std::ostream& f, const std::vector<Det>& v) {
  f << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    char b[160];
    std::snprintf(b, sizeof b, "%s[%.4f, %.4f, %.4f, %.4f, %.6f, %d]", i ? ", " : "", v[i].x1, v[i].y1, v[i].x2, v[i].y2,
                  v[i].score, v[i].cls);
    f << b;
  }
  f << "]";
}

static int run_dump(const Args& a, Pipeline& p, const std::string& mode) {
  std::ofstream f(a.dump_out);
  if (!f) { std::cerr << "cannot write " << a.dump_out << "\n"; return 1; }
  f << "{\n \"meta\": " << meta_json(a, p, mode) << ",\n \"images\": {\n";
  bool first = true;
  for (auto& path : list_images(a.images)) {
    cv::Mat im = cv::imread(path.string());
    FrameResult r = p.process(im, path.stem().string());
    f << (first ? "" : ",\n") << "  \"" << path.filename().string() << "\": {\"pixel_sum\": " << pixel_sum(im) << ", \"persons\": ";
    write_dets(f, r.persons);
    f << ", \"items\": ";
    write_dets(f, r.items);
    f << "}";
    first = false;
  }
  f << "\n }\n}\n";
  std::printf("dumped detections (%s, %s) -> %s\n", mode.c_str(), a.variant.c_str(), a.dump_out.c_str());
  return 0;
}

static int run_ui(const Args& a, Pipeline& p) {
  // Source: image dir / image file / video file / webcam index. Frames are produced lazily.
  std::vector<fs::path> files;
  cv::VideoCapture cap;
  std::string src = a.source;
  bool is_cam = !src.empty() && std::all_of(src.begin(), src.end(), ::isdigit);
  if (is_cam) { cap.open(std::stoi(src)); }
  else if (fs::is_directory(src)) files = list_images(src);
  else {
    std::string ext = fs::path(src).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files.push_back(src); else cap.open(src);
  }
  if (files.empty() && !cap.isOpened()) { std::cerr << "FAILED: cannot open source '" << src << "'\n"; return 1; }

  const std::string win = "ppe_runner (q / esc to quit)";
  if (!a.no_display) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
  double fps = 0;
  int n = 0;
  for (size_t idx = 0;; ++idx) {
    const auto t0 = std::chrono::steady_clock::now();
    cv::Mat frame;
    if (!files.empty()) { if (idx >= files.size()) break; frame = cv::imread(files[idx].string()); }
    else { if (!cap.read(frame)) break; }
    if (frame.empty()) continue;
    FrameResult r = p.process(frame);
    // FPS = 1 / (read + preprocess + inference + postprocess), smoothed over frames; drawing and window time excluded.
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fps = (n == 0) ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
    cv::Mat shown = render(frame, p, r, fps);
    if (!a.save_frame.empty() && n == a.save_index) {
      fs::create_directories(fs::path(a.save_frame).parent_path().empty() ? "." : fs::path(a.save_frame).parent_path());
      cv::imwrite(a.save_frame, shown);
      std::printf("saved frame %d (FPS overlay %.1f) -> %s\n", n, fps, a.save_frame.c_str());
    }
    ++n;
    if (!a.no_display) {
      cv::imshow(win, shown);
      const int k = cv::waitKey(1);
      if (k == 'q' || k == 27) break;
    }
    if (a.max_frames && n >= a.max_frames) break;
    if (!a.save_frame.empty() && a.no_display && !a.max_frames && n > a.save_index) break;
  }
  std::printf("processed %d frames, smoothed FPS %.1f\n", n, fps);
  return 0;
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  try {
    if (!a.variant.empty()) {
      if (a.person_path.empty()) a.person_path = "models/person_" + a.variant + ".onnx";
      if (a.ppe_path.empty()) a.ppe_path = "models/ppe_" + a.variant + ".onnx";
    }
    if (a.person_path.empty() && a.ppe_path.empty()) { usage(); return 2; }
    if (a.variant.empty()) a.variant = "custom";
    Pipeline p;
    p.cascade = a.cascade;
    p.conf = a.conf; p.iou = a.iou;
    if (!a.save_lb.empty()) { fs::create_directories(a.save_lb); p.save_lb_dir = a.save_lb; }
    const bool need_person = a.cascade || !a.person_path.empty();
    if (need_person) p.person = std::make_unique<Detector>(a.person_path, person_names(), a.threads, 640, a.ort_config);
    if (a.cascade || a.person_path.empty()) p.ppe = std::make_unique<Detector>(a.ppe_path, ppe_names(), a.threads, 640, a.ort_config);
    const std::string mode = a.cascade ? "cascade" : (p.person ? "person" : "ppe");
    for (Detector* d : {p.person.get(), p.ppe.get()})
      if (d) {
        auto s = d->output_shape();
        std::printf("loaded %s  output shape (%lld, %lld, %lld)\n", d->path().c_str(), (long long)s[0], (long long)s[1], (long long)s[2]);
      }
    if (a.bench) { if (a.bench_out.empty()) { std::cerr << "--bench needs --out FILE\n"; return 2; } return run_bench(a, p, mode); }
    if (!a.dump_out.empty()) return run_dump(a, p, mode);
    if (a.source.empty()) a.source = a.images;
    return run_ui(a, p);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
