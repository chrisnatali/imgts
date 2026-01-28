
FROM condaforge/miniforge3:latest

WORKDIR /app

# Copy package
COPY . /app

# Avoid installing a GPU-specific torch; image already contains it
RUN mamba env update -n base -f environment.yml && mamba clean -afy


