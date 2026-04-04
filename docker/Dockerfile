FROM firedrakeproject/firedrake-vanilla:2024-04
LABEL authors="hosseinkhan"



#====================
# Firedrake change uid approach
USER root
ARG USER_ID
ARG GROUP_ID
# TODO: Maybe set this at the top of the file is enough, and faster?
RUN usermod -u $USER_ID firedrake
RUN groupmod -g $GROUP_ID firedrake

# Install git-lfs to handle large files in HydroGym
RUN apt-get update && apt-get install -y git-lfs && git lfs install --skip-repo && apt-get clean


# Rename firedrake to user_firedrake as it is more descriptive as root
#USER root
#RUN usermod --login user_firedrake firedrake

# Switch back to user_firedrake
#USER user_firedrake

#RUN bash -c "echo 'home: $(ls -la /home)'"
#RUN echo $(ls -la /home)

#====================
USER firedrake
# Source the firedrake environment
ENV PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT=/home/firedrake/firedrake/bin/activate

# Create a directory for mounting
RUN mkdir -p /home/firedrake/mount_dir/project_root/

# Set the owner of the mounted directory to firedrake
# RUN chown -R firedrake:firedrake /home/firedrake/mount_dir

# Switch back to the firedrake user
USER firedrake
RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
    pip install --upgrade pip==24.0 && \
    pip install beartype \
                control \
                git+https://github.com/ReHoss/iCEM.git@control_dde_dynamics_estimator \
                gymnasium==0.29.1 \
                jaxtyping==0.2.24 \
                matplotlib \
                mlflow==1.27.0 \
                numpy \
                omegaconf \
                pandas \
                pytest \
                pyyaml \
                ruff \
                ruff-lsp \
                scipy \
                git+https://github.com/ReHoss/singularity-hydrogym.git@feature/working_environments \
                tensorboard \
                torch==2.1 \
                torchdde \
                torchdiffeq \
                typing_extensions==4.5.0 \
    "
#                git+https://github.com/thibmonsel/torchdde.git@master \

# The gymnasium==0.29.1 is needed to avoid the wrapped env attribute access error
# See release notes on that topic.

# Copy the package source code to the container
#COPY src /home/firedrake/mount_dir/project_root/src
#COPY pyproject.toml /home/firedrake/mount_dir/project_root/
# Readme is needed by the build process
#COPY README.md /home/firedrake/mount_dir/project_root/
#WORKDIR /home/firedrake/mount_dir/project_root

WORKDIR /home/firedrake/mount_dir/project_root


# Set the PYTHONPATH
ENV PYTHONPATH="/home/firedrake/mount_dir/project_root:$PYTHONPATH"

# Install the package  # TODO: Set back?
#RUN /bin/bash -c "source $PATH_FIREDRAKE_VENV_ACTIVATION_SCRIPT && \
#    pip install -e ."


# PRINT THE CURRENT ENVIRONMENT TO THE TERMINAL
RUN /bin/bash -c "echo 'Current python: $(which python)' && \
    echo 'Current pip: $(which pip)' && \
    echo 'Current python version: $(python --version)' && \
    echo 'Current pip version: $(pip --version)' && \
    echo 'Current pip list: $(pip list)'"


# --- Add firedrake user to the group of the host user ---
# Arguments passed from the host to the build command using --build-arg
#ARG GROUP_ID
# Switch to root user
#USER root
# Create a group with the same id as the host user
#RUN groupadd -g $GROUP_ID host_group_replica

# Add firedrake user to the group
#RUN usermod -aG host_group_replica firedrake

# Switch back to firedrake user
#USER firedrake
# ---



#RUN addgroup --gid $GROUP_ID host_user_replica
# Add a new user with the same id and group id as the host user
# --disabled-password: Do not assign a password
# --gecos: Set the user information
# --gid: Set the group id
# --uid: Set the user id

#====================
# Switch to root user to create a new user
#USER root
#ARG USER_ID
#ARG GROUP_ID
#RUN addgroup --gid $GROUP_ID host_group_replica
#
## Add firedrake user to the group
#RUN usermod -aG host_group_replica firedrake
#
#RUN adduser \
#    --disabled-password \
#    --gecos '' \
#    --gid $GROUP_ID \
#    --uid $USER_ID \
#    host_user_replica
#
## Set appropriate permissions on the directories
#RUN chown -R host_user_replica:host_group_replica /home/firedrake/
#
#USER host_user_replica
#====================


# USER firedrake



ENTRYPOINT ["/bin/bash", "-l", "-c", "source /home/firedrake/firedrake/bin/activate && /bin/bash"]
