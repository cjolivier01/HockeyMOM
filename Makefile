
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
	@echo "Available targets:"
	@echo "------------------"
	@grep -E '^[a-zA-Z_-]+:' Makefile | grep -v print_targets | grep -v '^[.#]' | sed 's/://' | sort | uniq
	@echo "------------------"
