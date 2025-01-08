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

#include <filesystem>

namespace zoo {

std::filesystem::path getDataPath() {
  const int MAX_DEPTH = 5;

  std::filesystem::path root = std::filesystem::path(".");
  int depth = 0;
  while (depth < MAX_DEPTH) {
    std::filesystem::path dataPath = root / "data";
    if (std::filesystem::is_directory(dataPath)) {
      return dataPath;
    }

    depth++;
    root = root / "..";
  }
  throw std::runtime_error("Could not find data path");
}

} // namespace zoo