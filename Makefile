
PRE_RUN="source .bazel_setup.sh"

TOPDIR=$(shell pwd)

all: print_targets

.PHONY: print_targets perf debug develop wheel test clean distclean expunge

perf:
	bazel/bazel.sh build --config=release //...

debug:
	bazel/bazel.sh build --config=debug //...

test:
	bazel/bazel.sh test --config=release //...

wheel:
	bazel/bazel.sh run --config=release //hockeymom:bdist_wheel 
	bazel/bazel.sh run --config=release //hmlib:bdist_wheel

clean:
	bazel/bazel.sh clean

distclean expunge:
	bazel/bazel.sh clean --expunge

develop:
#	bazel/bazel.sh build --config=debug //hockeymom:develop
	bazel/bazel.sh run --config=release //hockeymom:link_ext
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
		'Developer Workflow' \
		'------------------' \
		'develop      Refreshes symlinks for external hockeymom assets then builds the hmlib develop wheel; run after editing shared code to sync local installs.' \
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
