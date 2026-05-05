
PRE_RUN="source .bazel_setup.sh"

TOPDIR=$(shell pwd)

all: print_targets

.PHONY: print_targets perf debug develop wheel docs test clean distclean expunge

perf:
	bazel/bazel.sh build --config=release //...

debug:
	bazel/bazel.sh build --config=debug //...

test:
	bazel/bazel.sh test --config=release //...

wheel:
	bazel/bazel.sh run --config=release //hockeymom:bdist_wheel 
	bazel/bazel.sh run --config=release //hmlib:bdist_wheel

docs:
	bazel/bazel.sh build //:all_doxygen_docs

clean:
	bazel/bazel.sh clean

distclean expunge:
	bazel/bazel.sh clean --expunge

develop:
#	bazel/bazel.sh build --config=debug //hockeymom:develop
	@if command -v nvcc >/dev/null 2>&1 || [ -f /usr/local/cuda/include/cuda_runtime.h ]; then \
		bazel/bazel.sh run --config=release //hockeymom:link_ext; \
	else \
		printf '%s\n' 'Skipping //hockeymom:link_ext: no local CUDA toolkit detected (nvcc and /usr/local/cuda/include/cuda_runtime.h are both missing).' >&2; \
	fi
	bazel/bazel.sh build --config=release //hmlib:develop

deps:
	cd external/hugin && $(TOPDIR)/bazel/bazel.sh run --config=release //:install_tree -- --prefix=$(CONDA_PREFIX)
	cd -
	touch .hugin_built
print_targets:
	@printf '%s\n' \
		"Available make targets (run 'make <target>'):" \
		'' \
		'Build Outputs' \
		'-------------' \
		'perf         Build every Bazel target with --config=release; use for optimized binaries before packaging or deploying.' \
		'debug        Build every Bazel target with --config=debug; use while iterating locally when you need symbols and asserts.' \
		'' \
		'Documentation' \
		'--------------' \
		'docs         Builds both hockeymom and hmlib Doxygen archives via //:all_doxygen_docs; run when you need refreshed API docs.' \
		'' \
		'Developer Workflow' \
		'------------------' \
		'develop      Refreshes hockeymom extension symlinks when a local CUDA toolkit is available, then builds the hmlib develop wheel.' \
		'test         Runs the release-configured Bazel test suite; use to verify regressions before submitting or tagging builds.' \
		'wheel        Builds release wheels for hockeymom and hmlib; run when you need distributable Python packages.' \
		'' \
		'Maintenance & Cleanup' \
		'---------------------' \
		'clean        bazel clean to drop cached outputs when builds behave strangely or you switch branches.' \
		'distclean    bazel clean --expunge (also aliased as expunge); run for a fully fresh Bazel state if clean is insufficient.' \
		'expunge      Same as distclean; provided for convenience.' \
		'' \
		'Dependencies' \
		'------------' \
		'deps         Installs the external hugin tree into your active conda prefix and marks it as built; run after installing a new environment or when hugin headers go missing.' \
		'' \
		'Meta' \
		'----' \
		'print_targets  Shows this help text.'
