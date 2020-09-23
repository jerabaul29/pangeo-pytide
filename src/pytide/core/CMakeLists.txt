# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
file(GLOB_RECURSE WRAPPED_SOURCES "*.cpp")
pybind11_add_module(core ${WRAPPED_SOURCES})

# BLAS
find_package(BLAS)
