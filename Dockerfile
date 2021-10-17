FROM ubuntu:18.04

# RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update
RUN apt-get install -y sudo

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu

SHELL ["/bin/bash", "-c"]

RUN sudo apt-get -qq install curl vim git zip libglib2.0-0 libsndfile1 libsm6 libxext6 libxrender-dev

WORKDIR /home/ubuntu/

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda config --set auto_activate_base false

RUN echo $'name: specvqgan\n\
channels:\n\
  - pytorch\n\
  - conda-forge\n\
  - defaults\n\
dependencies:\n\
  - _libgcc_mutex=0.1=conda_forge\n\
  - _openmp_mutex=4.5=1_llvm\n\
  - abseil-cpp=20210324.0=h9c3ff4c_0\n\
  - absl-py=0.12.0=pyhd8ed1ab_0\n\
  - aiohttp=3.7.4=py38h497a2fe_0\n\
  - altair=4.1.0=py_1\n\
  - appdirs=1.4.4=pyh9f0ad1d_0\n\
  - argh=0.26.2=pyh9f0ad1d_1002\n\
  - argon2-cffi=20.1.0=py38h497a2fe_2\n\
  - arrow-cpp=4.0.0=py38hd6878d3_0_cpu\n\
  - astor=0.8.1=pyh9f0ad1d_0\n\
  - async-timeout=3.0.1=py_1000\n\
  - async_generator=1.10=py_0\n\
  - attrs=20.3.0=pyhd3deb0d_0\n\
  - audioread=2.1.9=py38h578d9bd_0\n\
  - autopep8=1.5.6=pyhd3eb1b0_0\n\
  - aws-c-cal=0.4.5=h76129ab_8\n\
  - aws-c-common=0.5.2=h7f98852_0\n\
  - aws-c-event-stream=0.2.7=h6bac3ce_1\n\
  - aws-c-io=0.9.1=ha5b09cb_1\n\
  - aws-checksums=0.1.11=h99e32c3_3\n\
  - aws-sdk-cpp=1.8.151=hceb1b1e_1\n\
  - backcall=0.2.0=pyh9f0ad1d_0\n\
  - backports=1.0=py_2\n\
  - backports.functools_lru_cache=1.6.4=pyhd8ed1ab_0\n\
  - base58=2.1.0=pyhd8ed1ab_0\n\
  - blas=1.0=mkl\n\
  - bleach=3.3.0=pyh44b312d_0\n\
  - blinker=1.4=py_1\n\
  - boto3=1.17.59=pyhd8ed1ab_0\n\
  - botocore=1.20.59=pyhd8ed1ab_1\n\
  - brotli=1.0.9=h9c3ff4c_4\n\
  - brotlipy=0.7.0=py38h497a2fe_1001\n\
  - bzip2=1.0.8=h7f98852_4\n\
  - c-ares=1.17.1=h7f98852_1\n\
  - ca-certificates=2021.5.30=ha878542_0\n\
  - cachetools=4.2.2=pyhd8ed1ab_0\n\
  - certifi=2021.5.30=py38h578d9bd_0\n\
  - cffi=1.14.5=py38ha65f79e_0\n\
  - chardet=4.0.0=py38h578d9bd_1\n\
  - click=7.1.2=pyh9f0ad1d_0\n\
  - cryptography=3.4.7=py38ha5dfef3_0\n\
  - cudatoolkit=11.1.1=h6406543_8\n\
  - cycler=0.10.0=py_2\n\
  - defusedxml=0.7.1=pyhd8ed1ab_0\n\
  - entrypoints=0.3=pyhd8ed1ab_1003\n\
  - ffmpeg=4.3.1=hca11adc_2\n\
  - flake8=3.9.0=pyhd3eb1b0_0\n\
  - freetype=2.10.4=h0708190_1\n\
  - fsspec=2021.4.0=pyhd8ed1ab_0\n\
  - future=0.18.2=py38h578d9bd_3\n\
  - gettext=0.19.8.1=h0b5b191_1005\n\
  - gflags=2.2.2=he1b5a44_1004\n\
  - gitdb=4.0.7=pyhd8ed1ab_0\n\
  - gitpython=3.1.15=pyhd8ed1ab_0\n\
  - glog=0.4.0=h49b9bf7_3\n\
  - gmp=6.2.1=h58526e2_0\n\
  - gnutls=3.6.13=h85f3911_1\n\
  - google-auth=1.28.0=pyh44b312d_0\n\
  - google-auth-oauthlib=0.4.1=py_2\n\
  - grpc-cpp=1.37.0=h36de60a_1\n\
  - grpcio=1.37.0=py38hdd6454d_0\n\
  - idna=2.10=pyh9f0ad1d_0\n\
  - imageio=2.9.0=py_0\n\
  - imageio-ffmpeg=0.4.3=pyhd8ed1ab_0\n\
  - importlib-metadata=4.0.1=py38h578d9bd_0\n\
  - ipykernel=5.5.5=py38hd0cf306_0\n\
  - ipython=7.22.0=py38hd0cf306_0\n\
  - ipython_genutils=0.2.0=py_1\n\
  - ipywidgets=7.6.3=pyhd3deb0d_0\n\
  - jedi=0.18.0=py38h578d9bd_2\n\
  - jinja2=2.11.3=pyh44b312d_0\n\
  - jmespath=0.10.0=pyh9f0ad1d_0\n\
  - joblib=1.0.1=pyhd8ed1ab_0\n\
  - jpeg=9b=h024ee3a_2\n\
  - jsonschema=3.2.0=pyhd8ed1ab_3\n\
  - jupyter_client=6.1.12=pyhd8ed1ab_0\n\
  - jupyter_core=4.7.1=py38h578d9bd_0\n\
  - jupyterlab_pygments=0.1.2=pyh9f0ad1d_0\n\
  - jupyterlab_widgets=1.0.0=pyhd8ed1ab_1\n\
  - kiwisolver=1.3.1=py38h1fd1430_1\n\
  - krb5=1.17.2=h926e7f8_0\n\
  - lame=3.100=h7f98852_1001\n\
  - lcms2=2.12=h3be6417_0\n\
  - ld_impl_linux-64=2.35.1=hea4e1c9_2\n\
  - libcurl=7.76.1=hc4aaa36_1\n\
  - libedit=3.1.20191231=he28a2e2_2\n\
  - libev=4.33=h516909a_1\n\
  - libevent=2.1.10=hcdb4288_3\n\
  - libffi=3.3=h58526e2_2\n\
  - libflac=1.3.3=h9c3ff4c_1\n\
  - libgcc-ng=9.3.0=h2828fa1_19\n\
  - libgfortran-ng=7.5.0=h14aa051_19\n\
  - libgfortran4=7.5.0=h14aa051_19\n\
  - libiconv=1.16=h516909a_0\n\
  - libllvm10=10.0.1=he513fc3_3\n\
  - libnghttp2=1.43.0=h812cca2_0\n\
  - libogg=1.3.4=h7f98852_1\n\
  - libopus=1.3.1=h7f98852_1\n\
  - libpng=1.6.37=h21135ba_2\n\
  - libprotobuf=3.15.8=h780b84a_0\n\
  - librosa=0.8.0=pyh9f0ad1d_0\n\
  - libsndfile=1.0.31=h9c3ff4c_1\n\
  - libsodium=1.0.18=h36c2ea0_1\n\
  - libssh2=1.9.0=ha56f1ee_6\n\
  - libstdcxx-ng=9.3.0=h6de172a_19\n\
  - libthrift=0.14.1=he6d91bd_1\n\
  - libtiff=4.1.0=h2733197_1\n\
  - libutf8proc=2.6.1=h7f98852_0\n\
  - libuv=1.41.0=h7f98852_0\n\
  - libvorbis=1.3.7=h9c3ff4c_0\n\
  - llvm-openmp=11.1.0=h4bd325d_1\n\
  - llvmlite=0.36.0=py38h4630a5e_0\n\
  - lz4-c=1.9.3=h9c3ff4c_0\n\
  - markdown=3.3.4=pyhd8ed1ab_0\n\
  - markupsafe=1.1.1=py38h497a2fe_3\n\
  - matplotlib-base=3.4.1=py38hcc49a3a_0\n\
  - mccabe=0.6.1=py38_1\n\
  - mistune=0.8.4=py38h497a2fe_1003\n\
  - mkl=2020.4=h726a3e6_304\n\
  - mkl-service=2.3.0=py38h1e0a361_2\n\
  - mkl_fft=1.3.0=py38h5c078b8_1\n\
  - mkl_random=1.2.0=py38hc5bc63f_1\n\
  - multidict=5.1.0=py38h497a2fe_1\n\
  - nbclient=0.5.3=pyhd8ed1ab_0\n\
  - nbconvert=6.0.7=py38h578d9bd_3\n\
  - nbformat=5.1.3=pyhd8ed1ab_0\n\
  - ncurses=6.2=h58526e2_4\n\
  - nest-asyncio=1.5.1=pyhd8ed1ab_0\n\
  - nettle=3.6=he412f7d_0\n\
  - ninja=1.10.2=h4bd325d_0\n\
  - notebook=6.3.0=pyha770c72_1\n\
  - numba=0.53.1=py38h0e12cce_0\n\
  - numpy=1.19.2=py38h54aff64_0\n\
  - numpy-base=1.19.2=py38hfa32c7d_0\n\
  - oauthlib=3.0.1=py_0\n\
  - olefile=0.46=pyh9f0ad1d_1\n\
  - omegaconf=2.0.6=py38h578d9bd_0\n\
  - openh264=2.1.1=h780b84a_0\n\
  - openssl=1.1.1k=h7f98852_0\n\
  - orc=1.6.7=heec2584_1\n\
  - packaging=20.9=pyh44b312d_0\n\
  - pandas=1.2.4=py38h1abd341_0\n\
  - pandoc=2.12=h7f98852_0\n\
  - pandocfilters=1.4.2=py_1\n\
  - parquet-cpp=1.5.1=2\n\
  - parso=0.8.2=pyhd8ed1ab_0\n\
  - pexpect=4.8.0=pyh9f0ad1d_2\n\
  - pickleshare=0.7.5=py_1003\n\
  - pillow=8.2.0=py38he98fc37_0\n\
  - pip=21.1=pyhd8ed1ab_0\n\
  - pooch=1.3.0=pyhd8ed1ab_0\n\
  - prometheus_client=0.10.1=pyhd8ed1ab_0\n\
  - prompt-toolkit=3.0.18=pyha770c72_0\n\
  - protobuf=3.15.8=py38h709712a_0\n\
  - ptyprocess=0.7.0=pyhd3deb0d_0\n\
  - pyarrow=4.0.0=py38hc9229eb_0_cpu\n\
  - pyasn1=0.4.8=py_0\n\
  - pyasn1-modules=0.2.7=py_0\n\
  - pycodestyle=2.6.0=pyhd3eb1b0_0\n\
  - pycparser=2.20=pyh9f0ad1d_2\n\
  - pydeck=0.5.0=pyh9f0ad1d_0\n\
  - pyflakes=2.2.0=pyhd3eb1b0_0\n\
  - pygments=2.8.1=pyhd8ed1ab_0\n\
  - pyjwt=2.0.1=pyhd8ed1ab_1\n\
  - pyopenssl=20.0.1=pyhd8ed1ab_0\n\
  - pyparsing=2.4.7=pyh9f0ad1d_0\n\
  - pyrsistent=0.17.3=py38h497a2fe_2\n\
  - pysocks=1.7.1=py38h578d9bd_3\n\
  - pysoundfile=0.10.3.post1=pyhd3deb0d_0\n\
  - python=3.8.8=hffdb5ce_0_cpython\n\
  - python-dateutil=2.8.1=py_0\n\
  - python_abi=3.8=1_cp38\n\
  - pytorch=1.8.1=py3.8_cuda11.1_cudnn8.0.5_0\n\
  - pytorch-lightning=1.2.10=pyhd8ed1ab_0\n\
  - pytz=2021.1=pyhd8ed1ab_0\n\
  - pyyaml=5.4.1=py38h497a2fe_0\n\
  - pyzmq=22.0.3=py38h2035c66_1\n\
  - re2=2021.04.01=h9c3ff4c_0\n\
  - readline=8.1=h46c0cb4_0\n\
  - requests=2.25.1=pyhd3deb0d_0\n\
  - requests-oauthlib=1.3.0=pyh9f0ad1d_0\n\
  - resampy=0.2.2=py_0\n\
  - rsa=4.7.2=pyh44b312d_0\n\
  - s2n=1.0.0=h9b69904_0\n\
  - s3transfer=0.4.2=pyhd8ed1ab_0\n\
  - scikit-learn=0.24.1=py38ha9443f7_0\n\
  - scipy=1.6.2=py38h91f5cce_0\n\
  - send2trash=1.5.0=py_0\n\
  - setuptools=49.6.0=py38h578d9bd_3\n\
  - six=1.15.0=pyh9f0ad1d_0\n\
  - smmap=3.0.5=pyh44b312d_0\n\
  - snappy=1.1.8=he1b5a44_3\n\
  - sqlite=3.35.5=h74cdb3f_0\n\
  - streamlit=0.80.0=pyhd8ed1ab_0\n\
  - tensorboard=2.4.1=pyhd8ed1ab_0\n\
  - tensorboard-plugin-wit=1.8.0=pyh44b312d_0\n\
  - terminado=0.9.4=py38h578d9bd_0\n\
  - testpath=0.4.4=py_0\n\
  - threadpoolctl=2.1.0=pyh5ca1d4c_0\n\
  - tk=8.6.10=h21135ba_1\n\
  - toml=0.10.2=pyhd8ed1ab_0\n\
  - toolz=0.11.1=py_0\n\
  - torchaudio=0.8.1=py38\n\
  - torchmetrics=0.3.1=pyhd8ed1ab_0\n\
  - torchvision=0.9.1=py38_cu111\n\
  - tornado=6.1=py38h497a2fe_1\n\
  - tqdm=4.60.0=pyhd8ed1ab_0\n\
  - traitlets=5.0.5=py_0\n\
  - typing-extensions=3.7.4.3=0\n\
  - typing_extensions=3.7.4.3=py_0\n\
  - tzlocal=2.1=pyh9f0ad1d_0\n\
  - urllib3=1.26.4=pyhd8ed1ab_0\n\
  - validators=0.18.2=pyhd3deb0d_0\n\
  - watchdog=2.0.3=py38h578d9bd_0\n\
  - wcwidth=0.2.5=pyh9f0ad1d_2\n\
  - webencodings=0.5.1=py_1\n\
  - werkzeug=1.0.1=pyh9f0ad1d_0\n\
  - wheel=0.36.2=pyhd3deb0d_0\n\
  - widgetsnbextension=3.5.1=py38h578d9bd_4\n\
  - x264=1!161.3030=h7f98852_1\n\
  - xz=5.2.5=h516909a_1\n\
  - yaml=0.2.5=h516909a_0\n\
  - yarl=1.6.3=py38h497a2fe_1\n\
  - zeromq=4.3.4=h9c3ff4c_0\n\
  - zipp=3.4.1=pyhd8ed1ab_0\n\
  - zlib=1.2.11=h516909a_1010\n\
  - zstd=1.4.9=ha95c52a_0\n\
  - pip:\n\
    - albumentations==0.5.2\n\
    - decorator==4.4.2\n\
    - imgaug==0.4.0\n\
    - networkx==2.5.1\n\
    - opencv-python==4.1.2.30\n\
    - opencv-python-headless==4.5.1.48\n\
    - pywavelets==1.1.1\n\
    - scikit-image==0.18.1\n\
    - shapely==1.7.1\n\
    - test-tube==0.7.5\n\
    - tifffile==2021.4.8\n\

' >> conda_env.yml

RUN conda env create -f conda_env.yml
RUN conda clean -afy
RUN rm ./Miniconda3-latest-Linux-x86_64.sh

SHELL ["conda", "run", "-n", "specvqgan", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "specvqgan"]
