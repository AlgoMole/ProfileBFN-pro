ARG BASE_IMAGE=vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:1.13.1-cu11.6.0-devel-py3.9-ubuntu20.04
ARG PUBLIC_IMAGE=${PUBLIC_IMAGE:-null}
FROM ${BASE_IMAGE} as public

COPY ./asset/ubuntu20.4_sources.list /etc/apt/sources.list
COPY ./asset /tmp/build
RUN apt update &&\
    apt install -y software-properties-common &&\
    add-apt-repository -y ppa:ubuntu-toolchain-r/test &&\
    xargs apt install -y < /tmp/build/apt_packages.txt &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN sh -c "$(curl -fsSL https://install.ohmyz.sh/)" \
    && chsh -s `which zsh` \
    && sed -i.bak '/ZSH_THEME/s/\".*\"/\"candy\"/' ${HOME}/.zshrc

RUN mkdir -p ${ZSH:-${HOME}/.oh-my-zsh}/custom/plugins/ &&\
    cp -r /tmp/build/misc/zsh-autosuggestions ${ZSH:-${HOME}/.oh-my-zsh}/custom/plugins/zsh-autosuggestions &&\
    sed -i.bak '/plugin/s/(.*)/(git zsh-autosuggestions)/' ${HOME}/.zshrc

RUN cp -r /tmp/build/misc/tmux ${HOME}/.tmux \
    && mv ${HOME}/.tmux/.tmux.conf ${HOME}/.tmux.conf


RUN conda init zsh \
    && conda init bash 
RUN conda clean -i 
# RUN conda install -y -c conda-forge openbabel
# RUN conda install -c rdkit rdkit
# RUN pip install rdkit
RUN conda clean -ya

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install -r /tmp/build/requirements.txt
RUN rm -rf /tmp/build

ENV PYTHONPATH .:${PYTHONPATH}

##############################################
FROM ${PUBLIC_IMAGE} as private
ARG TORCH_HOME
ARG _USER
ARG _UID
ARG _GID
ARG PW
ARG _HOME

COPY ./asset /tmp/build
RUN xargs apt install -y < /tmp/build/apt_packages.txt

RUN pip install -r /tmp/build/requirements.txt

# Option1: Using unencrypted password/ specifying password
RUN usermod --password $(echo ${PW} | openssl passwd -1 -stdin) root
RUN useradd -m ${_USER} -d ${_HOME} --uid=${_UID} -s /usr/bin/zsh && echo "${_USER}:${PW}" | chpasswd

# make the color of zsh-autosuggestions right
ENV TERM xterm-256color

RUN cp -r /root/.oh-my-zsh ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.oh-my-zsh &&\
    cp /root/.zshrc ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.zshrc &&\
    cp /root/.tmux.conf ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.tmux.conf && \
    cp -r /root/.tmux ${_HOME} && chown ${_USER}:${_USER} -R ${_HOME}/.tmux


RUN chown ${_USER}:${_USER} -R /tmp/build
USER ${_UID}:${_GID}
WORKDIR ${_HOME}

ENV TORCH_HOME ${TORCH_HOME}

COPY --chown=${_USER}:${_USER} ./asset/ssh .ssh
RUN chmod 0700 .ssh && chmod 600 .ssh/id_rsa && chmod 644 .ssh/id_rsa.pub \
    &&ssh-keyscan github.com >> .ssh/known_hosts \
    && cp /tmp/build/gitconfig ${_HOME}/.gitconfig \
    && cp /tmp/build/download-vs-code-server.sh ${_HOME}/

RUN rm -rf /tmp/build


CMD "zsh"


