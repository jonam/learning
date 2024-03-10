# Use the official Ubuntu image as the base
FROM ubuntu:latest

# Install the necessary packages including Java, Python, Git, Python venv, and utilities for downloading and unpacking
RUN apt-get update && \
    apt-get install -y gcc-12 g++-12 make vim gdb clang-tidy openjdk-21-jdk python3-pip python3-dev python3.10-venv git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the default compiler to gcc-12 and g++-12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Create symlinks for Python and Pip if they don't already exist
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Download and install the latest Hadoop (adjust the version as necessary)
RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz && \
    tar -xzf hadoop-3.3.6.tar.gz && \
    mv hadoop-3.3.6 /usr/local/hadoop && \
    rm hadoop-3.3.6.tar.gz

# Install PySpark
RUN pip install pyspark

# Set environment variables for Java, Hadoop, and Spark
ENV JAVA_HOME /usr/lib/jvm/java-21-openjdk-amd64/
ENV HADOOP_HOME /usr/local/hadoop
ENV SPARK_HOME /usr/local/spark
ENV PATH $PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

# Set the working directory inside the container
WORKDIR /workspace

# By default, run a shell when the container starts
CMD ["/bin/bash"]
