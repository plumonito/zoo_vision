// This file is part of zoo_vision.
//
// zoo_vision is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// zoo_vision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// zoo_vision. If not, see <https://www.gnu.org/licenses/>.

#include <chrono>
#include <deque>
#include <numeric>

namespace zoo {
class RateSampler {
public:
  using clock_t = std::chrono::high_resolution_clock;

  RateSampler(size_t maxSamples = 50) : maxSamples_{maxSamples} {}

  void tick() {
    auto now = clock_t::now();
    if (lastTick_) {
      auto period = now - *lastTick_;
      auto periodNs = std::chrono::duration_cast<std::chrono::nanoseconds>(period).count();

      if (samplesNanoseconds_.size() >= maxSamples_) {
        samplesNanoseconds_.pop_back();
      }
      samplesNanoseconds_.push_front(periodNs);
    }
    lastTick_ = now;
  }

  float sampleRateHz() const {
    float averageNs = std::reduce(samplesNanoseconds_.begin(), samplesNanoseconds_.end()) /
                      static_cast<float>(samplesNanoseconds_.size());
    constexpr float NS_TO_S = 1e9f;
    return 1.0f / (averageNs * NS_TO_S);
  }

private:
  size_t maxSamples_;
  std::deque<uint64_t> samplesNanoseconds_;
  std::optional<clock_t::time_point> lastTick_;
};

class TimedSection {
public:
  using clock_t = std::chrono::high_resolution_clock;

  TimedSection() : startTime_{clock_t::now()} {}
  std::chrono::nanoseconds time() const { return clock_t::now() - startTime_; }

  clock_t::time_point startTime_;
};
} // namespace zoo