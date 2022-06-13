FROM ziqipeter/symsim:latest

# sudo docker build -f Dockerfile -t test_multimap .
RUN apt-get -y update
RUN apt-get -y install git python3-pip
RUN cd ~ \
&& git clone https://github.com/Teichlab/MultiMAP.git\
&& cd MultiMAP\
&& pip3 install .\
&& pip3 install anndata scanpy leidenalg\
&& cd ..\
&& rm -rf MultiMAP\
&& mkdir -p /test_multimap/outputs

# COPY run_multimap.py /test_multimap/run_multimap.py  
COPY * /test_multimap/
CMD cd /test_multimap\
&& python3 run_multimap.py