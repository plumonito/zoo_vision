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

#include "zoo_vision/track_matcher.hpp"

#include <ranges>

namespace {
float iou(const Eigen::AlignedBox2f a, const Eigen::AlignedBox2f b) {
  const auto anb = a.intersection(b);
  const auto aub = a.merged(b);
  const auto area = [](const Eigen::AlignedBox2f x) {
    const auto sizes = x.sizes();
    return sizes[0] * sizes[1];
  };
  return area(anb) / area(aub);
}

std::pair<std::pair<Eigen::Index, Eigen::Index>, float> eigen_argmax(Eigen::MatrixXf &m) {
  float maxValue = m(0, 0);
  std::pair<Eigen::Index, Eigen::Index> maxIndex{0, 0};

  for (const Eigen::Index r : std::views::iota(Eigen::Index{0}, m.rows())) {
    for (const Eigen::Index c : std::views::iota(Eigen::Index{0}, m.cols())) {
      const float value = m(r, c);
      if (value > maxValue) {
        maxValue = value;
        maxIndex = {r, c};
      }
    }
  }
  return {maxIndex, maxValue};
}

} // namespace

namespace zoo {

TrackMatcher::TrackMatcher() = default;

void TrackMatcher::update(std::span<const Eigen::AlignedBox2f> boxes, std::span<TrackId> outputTrackIds) {
  const size_t inputBoxCount = boxes.size();

  Eigen::MatrixXf score;
  score.resize(validTrackCount_, inputBoxCount);

  for (const int r : std::views::iota(0uz, validTrackCount_)) {
    for (const int c : std::views::iota(0uz, inputBoxCount)) {
      score(r, c) = iou(tracks_[r].second, boxes[c]);
    }
  }

  // Init output to invalids
  std::array<bool, MAX_TRACK_COUNT> inputUsed{false};
  std::array<bool, MAX_TRACK_COUNT> trackUsed{false};

  // Greedy matching
  if (validTrackCount_ > 0 && inputBoxCount > 0) {
    auto argmax = eigen_argmax(score);
    while (argmax.second > 0) {
      const auto [r, c] = argmax.first;
      outputTrackIds[c] = tracks_[r].first;
      tracks_[r].second = boxes[c];
      inputUsed[c] = true;
      trackUsed[r] = true;
      score.row(r).setConstant(0.0f);
      score.col(c).setConstant(0.0f);

      argmax = eigen_argmax(score);
    }
  }

  // Drop missed tracks
  for (const int r : std::views::iota(0uz, validTrackCount_)) {
    if (!trackUsed[r]) {
      tracks_[r].first = INVALID_TRACK_ID;
    }
  }

  // Compact
  {
    size_t dst = 0;
    for (size_t src = 0; src < validTrackCount_; src++) {
      const bool isValid = tracks_[src].first != INVALID_TRACK_ID;
      if (isValid) {
        if (src != dst) {
          tracks_[dst] = tracks_[src];
        }
        dst += 1;
      }
    }
    validTrackCount_ = dst;
  }

  // Create new tracks
  for (const int c : std::views::iota(0uz, inputBoxCount)) {
    if (inputUsed[c]) {
      continue;
    }
    const auto newTrackId = nextTrackId_;
    nextTrackId_ += 1;

    outputTrackIds[c] = newTrackId;
    tracks_[validTrackCount_] = {newTrackId, boxes[c]};
    validTrackCount_ += 1;
  }
}
} // namespace zoo
