#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sndfile.h>

#include <webrtc/modules/audio_processing/include/audio_processing.h>

namespace {

constexpr int kTargetSampleRateHz = 48000;
constexpr int kTargetChannels = 1;
constexpr int kFrameMs = 10;
constexpr std::size_t kFrameSamples = static_cast<std::size_t>(
    (kTargetSampleRateHz * kFrameMs) / 1000);

struct Options {
  std::string input_path;
  std::string output_path;
  std::string preset = "light";
};

struct AudioData {
  std::vector<float> samples;
  int sample_rate_hz = 0;
  int channels = 0;
};

struct PresetSettings {
  const char* name = "light";
  webrtc::NoiseSuppression::Level ns_level = webrtc::NoiseSuppression::kLow;
};

std::string JsonEscape(const std::string& value) {
  std::ostringstream escaped;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << ch;
        break;
    }
  }
  return escaped.str();
}

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--input" && index + 1 < argc) {
      options.input_path = argv[++index];
    } else if (arg == "--output" && index + 1 < argc) {
      options.output_path = argv[++index];
    } else if (arg == "--preset" && index + 1 < argc) {
      options.preset = argv[++index];
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "usage: webrtc_apm_wav --input <input.wav> --output <output.wav> "
             "[--preset light|heavy]\n";
      std::exit(0);
    } else {
      Fail("Unknown or incomplete argument: " + arg);
    }
  }

  if (options.input_path.empty()) {
    Fail("Missing required --input");
  }
  if (options.output_path.empty()) {
    Fail("Missing required --output");
  }
  return options;
}

PresetSettings ResolvePreset(const std::string& preset_name) {
  if (preset_name == "light") {
    return {"light", webrtc::NoiseSuppression::kLow};
  }
  if (preset_name == "heavy") {
    return {"heavy", webrtc::NoiseSuppression::kVeryHigh};
  }
  Fail("Unsupported preset: " + preset_name);
}

AudioData ReadWavMono(const std::string& input_path) {
  SF_INFO info {};
  SNDFILE* raw_file = sf_open(input_path.c_str(), SFM_READ, &info);
  if (raw_file == nullptr) {
    Fail("Failed to open input WAV: " + input_path);
  }
  std::unique_ptr<SNDFILE, decltype(&sf_close)> file(raw_file, &sf_close);

  if (info.channels <= 0 || info.samplerate <= 0) {
    Fail("Input WAV has invalid format metadata: " + input_path);
  }

  std::vector<float> interleaved(static_cast<std::size_t>(info.frames) *
                                 static_cast<std::size_t>(info.channels));
  const sf_count_t frames_read = sf_readf_float(file.get(), interleaved.data(), info.frames);
  if (frames_read != info.frames) {
    Fail("Failed to read complete input WAV: " + input_path);
  }

  AudioData audio;
  audio.sample_rate_hz = info.samplerate;
  audio.channels = info.channels;
  audio.samples.resize(static_cast<std::size_t>(info.frames));

  for (sf_count_t frame = 0; frame < info.frames; ++frame) {
    float mono_sample = 0.0f;
    const std::size_t base = static_cast<std::size_t>(frame) *
                             static_cast<std::size_t>(info.channels);
    for (int channel = 0; channel < info.channels; ++channel) {
      mono_sample += interleaved[base + static_cast<std::size_t>(channel)];
    }
    audio.samples[static_cast<std::size_t>(frame)] =
        mono_sample / static_cast<float>(info.channels);
  }

  return audio;
}

std::vector<float> ResampleLinear(const std::vector<float>& input,
                                  int source_rate_hz,
                                  int target_rate_hz) {
  if (source_rate_hz == target_rate_hz || input.empty()) {
    return input;
  }

  const double ratio = static_cast<double>(source_rate_hz) /
                       static_cast<double>(target_rate_hz);
  const std::size_t output_frames = static_cast<std::size_t>(std::llround(
      static_cast<double>(input.size()) * static_cast<double>(target_rate_hz) /
      static_cast<double>(source_rate_hz)));

  std::vector<float> output(output_frames, 0.0f);
  for (std::size_t output_index = 0; output_index < output_frames; ++output_index) {
    const double source_position = static_cast<double>(output_index) * ratio;
    const std::size_t left_index = static_cast<std::size_t>(source_position);
    const std::size_t clamped_left = std::min(left_index, input.size() - 1);
    const std::size_t right_index = std::min(clamped_left + 1, input.size() - 1);
    const double fraction = source_position - static_cast<double>(clamped_left);
    const double left_value = static_cast<double>(input[clamped_left]);
    const double right_value = static_cast<double>(input[right_index]);
    output[output_index] = static_cast<float>(
        left_value + ((right_value - left_value) * fraction));
  }

  return output;
}

std::vector<float> ProcessWithWebRtcApm(const std::vector<float>& mono_samples,
                                        const PresetSettings& preset) {
  std::unique_ptr<webrtc::AudioProcessing> apm(webrtc::AudioProcessing::Create());
  if (!apm) {
    Fail("Failed to create WebRTC AudioProcessing instance");
  }

  if (apm->noise_suppression()->set_level(preset.ns_level) != webrtc::AudioProcessing::kNoError) {
    Fail("Failed to configure WebRTC noise suppression level");
  }
  if (apm->noise_suppression()->Enable(true) != webrtc::AudioProcessing::kNoError) {
    Fail("Failed to enable WebRTC noise suppression");
  }

  const webrtc::StreamConfig stream_config(kTargetSampleRateHz, kTargetChannels, false);

  const std::size_t padded_frames =
      ((mono_samples.size() + kFrameSamples - 1) / kFrameSamples) * kFrameSamples;
  std::vector<float> output(padded_frames, 0.0f);
  std::vector<float> input_frame(kFrameSamples, 0.0f);
  std::vector<float> output_frame(kFrameSamples, 0.0f);
  const float* input_channels[] = {input_frame.data()};
  float* output_channels[] = {output_frame.data()};

  for (std::size_t frame_offset = 0; frame_offset < padded_frames;
       frame_offset += kFrameSamples) {
    std::fill(input_frame.begin(), input_frame.end(), 0.0f);
    std::fill(output_frame.begin(), output_frame.end(), 0.0f);

    const std::size_t available = std::min(kFrameSamples, mono_samples.size() - std::min(frame_offset, mono_samples.size()));
    if (available > 0 && frame_offset < mono_samples.size()) {
      std::copy_n(mono_samples.data() + frame_offset, available, input_frame.data());
    }

    const int rc = apm->ProcessStream(
        input_channels,
        stream_config,
        stream_config,
        output_channels);
    if (rc != webrtc::AudioProcessing::kNoError) {
      Fail("WebRTC ProcessStream failed with code " + std::to_string(rc));
    }

    std::copy(output_frame.begin(), output_frame.end(), output.begin() + frame_offset);
  }

  output.resize(mono_samples.size());
  return output;
}

void WriteWavMono16(const std::string& output_path, const std::vector<float>& mono_samples) {
  SF_INFO info {};
  info.channels = kTargetChannels;
  info.samplerate = kTargetSampleRateHz;
  info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

  SNDFILE* raw_file = sf_open(output_path.c_str(), SFM_WRITE, &info);
  if (raw_file == nullptr) {
    Fail("Failed to open output WAV: " + output_path);
  }
  std::unique_ptr<SNDFILE, decltype(&sf_close)> file(raw_file, &sf_close);

  std::vector<float> clamped(mono_samples.size(), 0.0f);
  std::transform(mono_samples.begin(),
                 mono_samples.end(),
                 clamped.begin(),
                 [](float sample) {
                   return std::clamp(sample, -1.0f, 0.9999695f);
                 });

  const sf_count_t frames_written =
      sf_writef_float(file.get(), clamped.data(), static_cast<sf_count_t>(clamped.size()));
  if (frames_written != static_cast<sf_count_t>(clamped.size())) {
    Fail("Failed to write output WAV: " + output_path);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const PresetSettings preset = ResolvePreset(options.preset);

    const AudioData input_audio = ReadWavMono(options.input_path);
    const std::vector<float> resampled_audio =
        ResampleLinear(input_audio.samples, input_audio.sample_rate_hz, kTargetSampleRateHz);
    const std::vector<float> denoised_audio =
        ProcessWithWebRtcApm(resampled_audio, preset);
    WriteWavMono16(options.output_path, denoised_audio);

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{"
         << "\"processor\":\"webrtc_apm_wav\","
         << "\"preset\":\"" << JsonEscape(preset.name) << "\","
         << "\"ns_level\":\""
         << (preset.ns_level == webrtc::NoiseSuppression::kLow ? "kLow" : "kVeryHigh")
         << "\","
         << "\"input_path\":\"" << JsonEscape(options.input_path) << "\","
         << "\"output_path\":\"" << JsonEscape(options.output_path) << "\","
         << "\"input_sample_rate_hz\":" << input_audio.sample_rate_hz << ","
         << "\"input_channels\":" << input_audio.channels << ","
         << "\"output_sample_rate_hz\":" << kTargetSampleRateHz << ","
         << "\"output_channels\":" << kTargetChannels << ","
         << "\"frames\":" << denoised_audio.size() << ","
         << "\"seconds\":"
         << (static_cast<double>(denoised_audio.size()) /
             static_cast<double>(kTargetSampleRateHz))
         << "}";
    std::cout << json.str() << std::endl;
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << std::endl;
    return 1;
  }
}
