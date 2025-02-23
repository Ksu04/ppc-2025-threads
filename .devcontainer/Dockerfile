FROM mcr.microsoft.com/devcontainers/base:noble

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
   && apt-get -y install \
      gcc-14 g++-14 clang-19 \
      cmake \
      mpich openmpi-bin libopenmpi-dev libomp-dev \
      valgrind \
      python3-pip \
      clangd-19 clang-tidy-19 \
      x11-xserver-utils xterm \
   && apt-get autoremove -y && apt-get clean -y

RUN pip install --no-cache-dir xlsxwriter==3.2.0 --break-system-packages