FROM centos:7

RUN yum update -y \
 && yum install -y make gcc perl-core pcre-devel wget zlib-devel git automake

# openssl
RUN wget https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz \
 && tar xf openssl*.gz \
 && cd openssl* \
 && ./config --prefix=/usr --openssldir=/etc/ssl zlib-dynamic \
 && make -j$(nproc) \
 && make install

# openssh
RUN git clone https://github.com/openssh/openssh-portable \
 && pushd openssh-portable \
 && autoreconf \
 && ./configure --prefix=/usr --sysconfdir=/etc \
 && make -j$(nproc) \
 && make install