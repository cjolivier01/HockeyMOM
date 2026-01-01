
PRE_RUN="source .bazel_setup.sh"

all: print_targets

.PHONY: print_targets perf debug develop wheel test

perf:
	bazel/bazel.sh build --config=release //...

debug:
	bazel/bazel.sh build --config=debug //...

test:
	bazel/bazel.sh test --config=release //...

wheel:
	bazel/bazel.sh run --config=release //hockeymom:bdist_wheel //hmlib:bdist_wheel

develop:
#	bazel/bazel.sh build --config=debug //hockeymom:develop
	bazel/bazel.sh run --config=release //hockeymom:link_ext
	bazel/bazel.sh build --config=release //hmlib:develop

print_targets:
	@echo "Available targets:"
	@echo "------------------"
	@grep -E '^[a-zA-Z_-]+:' Makefile | grep -v print_targets | grep -v '^[.#]' | sed 's/://' | sort | uniq
# --- IGNORE ---
