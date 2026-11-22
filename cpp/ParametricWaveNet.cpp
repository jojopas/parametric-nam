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
        if (k.contains("labels"))
        {
            const auto& l = k["labels"];
            if (!l.is_array() || l.size() < 2)
                throw std::runtime_error("knob " + knob.name + ": labels must be 2+ strings");
            for (const auto& x : l)
            {
                if (!x.is_string())
                    throw std::runtime_error("knob " + knob.name + ": labels must be 2+ strings");
                knob.labels.push_back(x.get<std::string>());
            }
        }
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
    std::vector<float> base = weights;
    std::unique_ptr<ParametricWaveNet> net(new ParametricWaveNet(cfg, std::move(weights), std::move(knobs), sr));
    if (model.contains("knob_scale"))
    {
        net->base_ = std::move(base);
        net->readKnobScale(model, config);
    }

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

void ParametricWaveNet::readKnobScale(const nlohmann::json& model, const nlohmann::json& config)
{
    // Walk the weights the way SPEC.md lays them out, noting each layer's conv / bias / mixin.
    const auto& ks = model["knob_scale"];
    const auto& layers = config["layers"];
    const int N = knobCount();
    if (!ks.is_array() || ks.size() != layers.size())
        throw std::runtime_error("knob_scale: one entry per layer array");
    size_t p = 0;
    for (size_t a = 0; a < layers.size(); ++a)
    {
        const auto& L = layers[a];
        if (L.value("gated", false))
            throw std::runtime_error("knob_scale: gated layers are not supported");
        for (const char* k : { "layer1x1", "head1x1", "bottleneck", "groups_input", "groups_input_mixin" })
            if (L.contains(k))
                throw std::runtime_error(std::string("knob_scale: layer key ") + k + " is not supported");
        const int ch = L.at("channels").get<int>(), K = L.at("kernel_size").get<int>();
        const int head = L.at("head_size").get<int>(), inW = L.at("input_size").get<int>();
        const auto& dil = L.at("dilations");
        if (!ks[a].is_array() || ks[a].size() != dil.size())
            throw std::runtime_error("knob_scale: one entry per layer");
        p += (size_t) (ch * inW);                                   // rechannel
        for (size_t l = 0; l < dil.size(); ++l)
        {
            const auto& row = ks[a][l];
            if (!row.is_array() || row.size() != (size_t) (ch * N))
                throw std::runtime_error("knob_scale: each layer needs channels x knobs values");
            ScaledLayer s;
            s.channels = ch;
            s.kernel = K;
            s.conv = p;
            p += (size_t) (ch * ch * K);
            s.bias = p;
            p += (size_t) ch;
            s.mixin = p;
            p += (size_t) (ch * (1 + N));
            p += (size_t) (ch * ch + ch);                           // 1x1
            s.scale = scale_.size();
            for (const auto& v : row)
                scale_.push_back(v.get<float>());
            scaled_.push_back(s);
        }
        p += (size_t) (head * ch + (L.value("head_bias", true) ? head : 0));
    }
    if (p + 1 != base_.size())
        throw std::runtime_error("knob_scale: the weights do not match a plain layout");
    work_ = base_;
    foldScale();
}

void ParametricWaveNet::foldScale()
{
    // conv weights, conv bias and the knob columns of input_mixin, times (1 + S . g) per channel.
    // The audio column of input_mixin is not scaled (SPEC.md).
    const int N = knobCount();
    for (int k = 0; k < N; ++k)
        folded_[(size_t) k] = values_[(size_t) k].load(std::memory_order_relaxed);
    for (const auto& s : scaled_)
    {
        const int ch = s.channels, K = s.kernel;
        for (int i = 0; i < ch; ++i)
        {
            float si = 1.0f;
            for (int k = 0; k < N; ++k)
                si += scale_[s.scale + (size_t) (i * N + k)] * folded_[(size_t) k];
            for (int j = 0; j < ch * K; ++j)
                work_[s.conv + (size_t) (i * ch * K + j)] = base_[s.conv + (size_t) (i * ch * K + j)] * si;
            work_[s.bias + (size_t) i] = base_[s.bias + (size_t) i] * si;
            for (int k = 0; k < N; ++k)
            {
                const size_t m = s.mixin + (size_t) (i * (1 + N) + 1 + k);
                work_[m] = base_[m] * si;
            }
        }
    }
    set_weights_(work_);
    foldedOnce_ = true;
}

void ParametricWaveNet::setKnob(int i, float v01)
{
    if (i < 0 || i >= knobCount())
        return;
    values_[(size_t) i].store(std::clamp(v01, 0.0f, 1.0f), std::memory_order_relaxed);
}

void ParametricWaveNet::_set_condition_array(NAM_SAMPLE** input, const int num_frames)
{
    if (!scaled_.empty())
    {
        bool moved = !foldedOnce_;
        for (int k = 0; k < knobCount() && !moved; ++k)
            moved = values_[(size_t) k].load(std::memory_order_relaxed) != folded_[(size_t) k];
        if (moved)
            foldScale();   // before this block's layers run
    }
    // row 0: the audio (as the base class); rows 1..N: the knobs, constant over the block.
    // With knob_scale the rows hold the values folded into the weights, so the two always agree.
    for (int j = 0; j < num_frames; ++j)
        _condition_input(0, j) = input[0][j];
    for (int k = 0; k < knobCount(); ++k)
    {
        const float v = scaled_.empty() ? values_[(size_t) k].load(std::memory_order_relaxed) : folded_[(size_t) k];
        for (int j = 0; j < num_frames; ++j)
            _condition_input(1 + k, j) = v;
    }
}
} // namespace nam_parametric
