# Project Structure
my_hockey_project/
├── WORKSPACE
├── BUILD.bazel
├── .bazelrc
├── cpp/
│   ├── BUILD.bazel
│   ├── hockey_engine.h
│   └── hockey_engine.cpp
├── python/
│   ├── BUILD.bazel
│   ├── my_hockey/
│   │   ├── __init__.py
│   │   └── py.typed
│   ├── pybind_module.cpp
│   ├── setup.py
│   └── pyproject.toml
└── tests/
    ├── BUILD.bazel
    ├── test_local_install.py
    └── test_hockey.py

# File Contents:

## WORKSPACE
workspace(name = "my_hockey_project")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Python rules
http_archive(
    name = "rules_python",
    sha256 = "c68bdc4fbec25de5b5493b8819cfc877c4ea299c0dcb15c244c5a00208cde311",
    strip_prefix = "rules_python-0.31.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python3_11",
    python_version = "3.11",
)

load("@python3_11//:defs.bzl", "interpreter")

# pybind11
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-2.11.1",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/v2.11.1.tar.gz"],
    sha256 = "22bc0e7b3a0abc45a8d4f5a3b8d0d42ccba24b8b5c93c9c5ea9c2e5a3b6bfa3c",
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.11.1",
    urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
    sha256 = "d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

## .bazelrc
# Python configuration
build --action_env=PYTHONPATH
build --python_top=@python3_11//:python_runtimes
test --test_env=PYTHONPATH

# C++ configuration
build --cxxopt=-std=c++17
build --host_cxxopt=-std=c++17

## BUILD.bazel (root)
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    requirements_lock = "//python:requirements.txt",
)

load("@pip_deps//:requirements.bzl", "install_deps")
install_deps()

## cpp/BUILD.bazel
cc_library(
    name = "hockey_engine",
    srcs = ["hockey_engine.cpp"],
    hdrs = ["hockey_engine.h"],
    visibility = ["//visibility:public"],
)

## cpp/hockey_engine.h
#pragma once
#include <string>
#include <vector>

namespace hockey {

class Player {
public:
    Player(const std::string& name, int number);
    
    std::string getName() const { return name_; }
    int getNumber() const { return number_; }
    int getGoals() const { return goals_; }
    
    void scoreGoal();
    std::string getStats() const;

private:
    std::string name_;
    int number_;
    int goals_ = 0;
};

class Game {
public:
    Game(const std::string& home_team, const std::string& away_team);
    
    void addPlayer(const Player& player);
    std::vector<Player> getPlayers() const { return players_; }
    std::string getGameInfo() const;
    
    void simulateGoal(int player_number);

private:
    std::string home_team_;
    std::string away_team_;
    std::vector<Player> players_;
};

std::string greet(const std::string& name);

} // namespace hockey

## cpp/hockey_engine.cpp
#include "hockey_engine.h"
#include <sstream>
#include <algorithm>

namespace hockey {

Player::Player(const std::string& name, int number) 
    : name_(name), number_(number) {}

void Player::scoreGoal() {
    goals_++;
}

std::string Player::getStats() const {
    std::ostringstream oss;
    oss << name_ << " (#" << number_ << "): " << goals_ << " goals";
    return oss.str();
}

Game::Game(const std::string& home_team, const std::string& away_team)
    : home_team_(home_team), away_team_(away_team) {}

void Game::addPlayer(const Player& player) {
    players_.push_back(player);
}

std::string Game::getGameInfo() const {
    std::ostringstream oss;
    oss << home_team_ << " vs " << away_team_ << "\n";
    oss << "Players: " << players_.size() << "\n";
    for (const auto& player : players_) {
        oss << "  " << player.getStats() << "\n";
    }
    return oss.str();
}

void Game::simulateGoal(int player_number) {
    auto it = std::find_if(players_.begin(), players_.end(),
        [player_number](Player& p) { return p.getNumber() == player_number; });
    
    if (it != players_.end()) {
        it->scoreGoal();
    }
}

std::string greet(const std::string& name) {
    return "Hello from C++ Hockey Engine, " + name + "!";
}

} // namespace hockey

## python/BUILD.bazel
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

pybind_extension(
    name = "_hockey_engine",
    srcs = ["pybind_module.cpp"],
    deps = ["//cpp:hockey_engine"],
)

py_library(
    name = "my_hockey",
    srcs = glob(["my_hockey/*.py"]),
    data = [":_hockey_engine"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

# Traditional wheel build
genrule(
    name = "build_wheel",
    srcs = [
        ":_hockey_engine",
        "setup.py",
        "pyproject.toml",
        "//python/my_hockey:__init__.py",
        "//python/my_hockey:py.typed",
    ],
    outs = ["dist/my_hockey-0.1.0-py3-none-any.whl"],
    cmd = """
        export PYTHONPATH=$$PWD
        cd python
        cp $(location :_hockey_engine) my_hockey/
        python setup.py bdist_wheel
        cp dist/*.whl $(location dist/my_hockey-0.1.0-py3-none-any.whl)
    """,
    tools = ["@python3_11//:python_runtimes"],
)

# Local installation target
genrule(
    name = "local_install",
    srcs = [":build_wheel"],
    outs = ["build/my_hockey/.installed"],
    cmd = """
        mkdir -p $$(dirname $(location build/my_hockey/.installed))/my_hockey
        cd $$(dirname $(location build/my_hockey/.installed))
        python -m pip install --target . --force-reinstall $(location :build_wheel)
        touch .installed
    """,
    tools = ["@python3_11//:python_runtimes"],
)

# Develop mode installation (editable install simulation)
genrule(
    name = "develop_install",
    srcs = [
        ":_hockey_engine",
        "//python/my_hockey:__init__.py",
        "//python/my_hockey:py.typed",
    ],
    outs = ["build/develop/.installed"],
    cmd = """
        mkdir -p $$(dirname $(location build/develop/.installed))/my_hockey
        cp $(location //python/my_hockey:__init__.py) $$(dirname $(location build/develop/.installed))/my_hockey/
        cp $(location //python/my_hockey:py.typed) $$(dirname $(location build/develop/.installed))/my_hockey/
        cp $(location :_hockey_engine) $$(dirname $(location build/develop/.installed))/my_hockey/
        touch $(location build/develop/.installed)
    """,
)

## python/requirements.txt
# No external Python dependencies needed for this example

## python/pybind_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp/hockey_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(_hockey_engine, m) {
    m.doc() = "Hockey Engine Python Bindings";
    
    m.def("greet", &hockey::greet, "A function that greets");
    
    py::class_<hockey::Player>(m, "Player")
        .def(py::init<const std::string&, int>())
        .def("get_name", &hockey::Player::getName)
        .def("get_number", &hockey::Player::getNumber)
        .def("get_goals", &hockey::Player::getGoals)
        .def("score_goal", &hockey::Player::scoreGoal)
        .def("get_stats", &hockey::Player::getStats);
    
    py::class_<hockey::Game>(m, "Game")
        .def(py::init<const std::string&, const std::string&>())
        .def("add_player", &hockey::Game::addPlayer)
        .def("get_players", &hockey::Game::getPlayers)
        .def("get_game_info", &hockey::Game::getGameInfo)
        .def("simulate_goal", &hockey::Game::simulateGoal);
}

## python/my_hockey/__init__.py
"""My Hockey Package - A C++ backed hockey simulation library."""

from ._hockey_engine import greet, Player, Game

__version__ = "0.1.0"
__all__ = ["greet", "Player", "Game"]

def hello_world():
    """Simple hello world function."""
    return greet("World")

## python/my_hockey/py.typed
# Marker file for type checking

## python/setup.py
from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "my_hockey._hockey_engine",
        [
            "pybind_module.cpp",
            "../cpp/hockey_engine.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir() + "/../../../include",
            "../cpp",
        ],
        cxx_std=17,
    ),
]

setup(
    name="my_hockey",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

## python/pyproject.toml
[build-system]
requires = ["setuptools", "wheel", "pybind11"]
build-backend = "setuptools.build_meta"

[project]
name = "my_hockey"
version = "0.1.0"
description = "A hockey simulation library with C++ backend"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.8"

## tests/BUILD.bazel
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_hockey",
    srcs = ["test_hockey.py"],
    deps = ["//python:my_hockey"],
)

py_test(
    name = "test_local_install",
    srcs = ["test_local_install.py"],
    data = ["//python:local_install"],
    env = {"PYTHONPATH": "python/build/my_hockey"},
)

## tests/test_hockey.py
import unittest
import my_hockey

class TestHockey(unittest.TestCase):
    
    def test_greet(self):
        result = my_hockey.greet("Test")
        self.assertEqual(result, "Hello from C++ Hockey Engine, Test!")
    
    def test_hello_world(self):
        result = my_hockey.hello_world()
        self.assertEqual(result, "Hello from C++ Hockey Engine, World!")
    
    def test_player(self):
        player = my_hockey.Player("Wayne Gretzky", 99)
        self.assertEqual(player.get_name(), "Wayne Gretzky")
        self.assertEqual(player.get_number(), 99)
        self.assertEqual(player.get_goals(), 0)
        
        player.score_goal()
        self.assertEqual(player.get_goals(), 1)
    
    def test_game(self):
        game = my_hockey.Game("Edmonton Oilers", "Calgary Flames")
        player = my_hockey.Player("Connor McDavid", 97)
        
        game.add_player(player)
        players = game.get_players()
        self.assertEqual(len(players), 1)
        self.assertEqual(players[0].get_name(), "Connor McDavid")
        
        game.simulate_goal(97)
        # Note: simulate_goal modifies the player in the game's vector
        info = game.get_game_info()
        self.assertIn("Edmonton Oilers vs Calgary Flames", info)

if __name__ == "__main__":
    unittest.main()

## tests/test_local_install.py
import unittest
import sys
import os

class TestLocalInstall(unittest.TestCase):
    
    def test_local_install_works(self):
        """Test that the locally installed package can be imported and used."""
        
        # Add the local install path to Python path
        local_install_path = os.path.join(os.getcwd(), "python", "build", "my_hockey")
        if local_install_path not in sys.path:
            sys.path.insert(0, local_install_path)
        
        try:
            import my_hockey
            
            # Test basic functionality
            result = my_hockey.hello_world()
            self.assertEqual(result, "Hello from C++ Hockey Engine, World!")
            
            # Test C++ bindings work
            player = my_hockey.Player("Local Test Player", 1)
            self.assertEqual(player.get_name(), "Local Test Player")
            player.score_goal()
            self.assertEqual(player.get_goals(), 1)
            
            print("✓ Local installation test passed!")
            
        except ImportError as e:
            self.fail(f"Failed to import locally installed my_hockey package: {e}")

if __name__ == "__main__":
    unittest.main()

# Build and Usage Commands:

# Build everything:
# bazel build //...

# Build wheel:
# bazel build //python:build_wheel

# Install locally:
# bazel build //python:local_install

# Develop mode:
# bazel build //python:develop_install

# Run tests:
# bazel test //tests:test_hockey
# bazel test //tests:test_local_install

# Clean:
# bazel clean
