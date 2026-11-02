#pragma once
// A parametric capture: a NAM WaveNet whose conditioning input is the audio
// plus N knob values (0..1) held constant over each block. The knobs' effect
// is learned at training time (PANAMA-style: one recording per knob setting),
// so the same model plays every setting of the amp it was captured from.
//
// File format: a standard .nam JSON with architecture "ParametricWaveNet"
// (files written before the format went public say "CoroParametricWaveNet"; both load).
// `config.knobs` lists the knobs (name, default); every layer's condition_size
// and the first layer's input_size are 1 + knobs. Rows 1..N of the condition
// are the knobs. Other NAM hosts refuse the architecture instead of mis-wiring
// it. parametric_nam/convert_panama.py writes this format; SPEC.md describes it.
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <NAM/dsp.h>
#include <NAM/wavenet/model.h>

namespace nam_parametric
{
class ParametricWaveNet : public nam::wavenet::WaveNet
{
public:
    static constexpr const char* kArchitecture = "ParametricWaveNet";
    static constexpr const char* kLegacyArchitecture = "CoroParametricWaveNet";
    static constexpr int kMaxKnobs = 8;

    struct Knob
    {
        std::string name;
        float def = 0.5f;
    };

    static bool isParametric(const nlohmann::json& model);
    // Builds the model or throws std::runtime_error saying what is wrong.
    static std::unique_ptr<ParametricWaveNet> fromJson(const nlohmann::json& model);

    int knobCount() const { return (int) knobs_.size(); }
    const Knob& knob(int i) const { return knobs_[(size_t) i]; }
    // Any thread. Clamped to 0..1; the audio thread picks it up on its next block.
    void setKnob(int i, float v01);
    float knobValue(int i) const { return values_[(size_t) i].load(std::memory_order_relaxed); }

protected:
    int _get_condition_dim() const override { return NumInputChannels() + knobCount(); }
    void _set_condition_array(NAM_SAMPLE** input, const int num_frames) override;

private:
    ParametricWaveNet(nam::wavenet::WaveNetConfig& cfg, std::vector<float> weights,
                      std::vector<Knob> knobs, double sampleRate);
    std::vector<Knob> knobs_;
    std::array<std::atomic<float>, kMaxKnobs> values_ {};
};
} // namespace nam_parametric
