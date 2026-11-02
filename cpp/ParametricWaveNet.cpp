#include "ParametricWaveNet.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nam_parametric
{
namespace
{
double sampleRateOf(const nlohmann::json& model)
{
    if (model.contains("sample_rate") && model["sample_rate"].is_number())
        return model["sample_rate"].get<double>();
    return -1.0;
}
} // namespace

bool ParametricWaveNet::isParametric(const nlohmann::json& model)
{
    if (!model.is_object() || !model.contains("architecture") || !model["architecture"].is_string())
        return false;
    const auto a = model["architecture"].get<std::string>();
    return a == kArchitecture || a == kLegacyArchitecture;
}

std::unique_ptr<ParametricWaveNet> ParametricWaveNet::fromJson(const nlohmann::json& model)
{
    if (!isParametric(model))
        throw std::runtime_error(std::string("not a ") + kArchitecture + " model");
    for (const char* k : { "config", "weights" })
        if (!model.contains(k))
            throw std::runtime_error(std::string("model JSON missing key ") + k);
    const auto& config = model["config"];
    if (!config.is_object() || !config.contains("knobs") || !config["knobs"].is_array())
        throw std::runtime_error("parametric model needs config.knobs");

    std::vector<Knob> knobs;
    for (const auto& k : config["knobs"])
    {
        Knob knob;
        if (k.is_object() && k.contains("name") && k["name"].is_string())
            knob.name = k["name"].get<std::string>();
        else
            throw std::runtime_error("each knob needs a name");
        if (k.contains("default") && k["default"].is_number())
            knob.def = std::clamp(k["default"].get<float>(), 0.0f, 1.0f);
        knobs.push_back(std::move(knob));
    }
    if (knobs.empty() || (int) knobs.size() > kMaxKnobs)
        throw std::runtime_error("parametric model needs 1.." + std::to_string(kMaxKnobs) + " knobs");

    const double sr = sampleRateOf(model);
    auto cfg = nam::wavenet::parse_config_json(config, sr);
    if (cfg.condition_dsp != nullptr)
        throw std::runtime_error("parametric model cannot also have a condition_dsp");
    const int want = 1 + (int) knobs.size();
    if (cfg.layer_array_params.empty() || cfg.layer_array_params.front().input_size != want)
        throw std::runtime_error("first layer input_size must be 1 + knobs");
    for (const auto& la : cfg.layer_array_params)
        if (la.condition_size != want)
            throw std::runtime_error("every layer's condition_size must be 1 + knobs");

    std::vector<float> weights = model["weights"].get<std::vector<float>>();
    std::unique_ptr<ParametricWaveNet> net(new ParametricWaveNet(cfg, std::move(weights), std::move(knobs), sr));

    if (model.contains("metadata") && model["metadata"].is_object())
    {
        const auto& md = model["metadata"];
        if (md.contains("loudness") && md["loudness"].is_number())
            net->SetLoudness(md["loudness"].get<double>());
        if (md.contains("input_level_dbu") && md["input_level_dbu"].is_number())
            net->SetInputLevel(md["input_level_dbu"].get<double>());
    }
    return net;
}

ParametricWaveNet::ParametricWaveNet(nam::wavenet::WaveNetConfig& cfg, std::vector<float> weights,
                                     std::vector<Knob> knobs, double sampleRate)
    : nam::wavenet::WaveNet(1, cfg.layer_array_params, cfg.head_scale, cfg.with_head, cfg.head_params,
                            std::move(weights), nullptr, sampleRate),
      knobs_(std::move(knobs))
{
    for (size_t i = 0; i < knobs_.size(); ++i)
        values_[i].store(knobs_[i].def, std::memory_order_relaxed);
}

void ParametricWaveNet::setKnob(int i, float v01)
{
    if (i < 0 || i >= knobCount())
        return;
    values_[(size_t) i].store(std::clamp(v01, 0.0f, 1.0f), std::memory_order_relaxed);
}

void ParametricWaveNet::_set_condition_array(NAM_SAMPLE** input, const int num_frames)
{
    // row 0: the audio (as the base class); rows 1..N: the knobs, constant over the block
    for (int j = 0; j < num_frames; ++j)
        _condition_input(0, j) = input[0][j];
    for (int k = 0; k < knobCount(); ++k)
    {
        const float v = values_[(size_t) k].load(std::memory_order_relaxed);
        for (int j = 0; j < num_frames; ++j)
            _condition_input(1 + k, j) = v;
    }
}
} // namespace nam_parametric
