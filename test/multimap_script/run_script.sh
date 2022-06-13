cp -r ../../data/real/diag/healthy_hema/multimap/BMMC/* .
sudo docker build -f Dockerfile -t run_multimap .
sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
sudo rm -rf ../bmmc_healthyhema_1000/multimap
sudo mv multimap ../bmmc_healthyhema_1000/multimap
rm -rf *.h5ad

cp -r ../../data/real/diag/mouse_brain_cortex/multimap/*.h5ad .
sudo docker build -f Dockerfile -t run_multimap .
sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
sudo rm -rf ../mbc/multimap
sudo mv multimap ../mbc/multimap
rm -rf *.h5ad

cp -r ../../data/real/diag/Xichen/multimap/*.h5ad .
sudo docker build -f Dockerfile -t run_multimap .
sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
sudo rm -rf ../spleen/multimap
sudo mv multimap ../spleen/multimap
rm -rf *.h5ad

cp -r ../../data/real/MOp/multimap/*.h5ad .
sudo docker build -f Dockerfile -t run_multimap .
sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
sudo mv multimap ../MOp/multimap
rm -rf *.h5ad

cp -r ../../data/real/ASAP-PBMC/multimap/*.h5ad .
sudo docker build -f Dockerfile -t run_multimap .
sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
sudo mv multimap ../pbmc/multimap
rm -rf *.h5ad

# simulated
for num in 10 # 1 2 3 4 5 6 7 8 9 10
do
    cp -r ../../data/simulated/6b16c_test_${num}_large/unequal2/multimap/*.h5ad .
    sudo docker build -f Dockerfile -t run_multimap .
    sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap 
    mkdir ../simulated/6b16c_${num}_large2_2
    sudo chown -R 1059113:372345 multimap
    sudo mv multimap ../simulated/6b16c_${num}_large2_2
    rm -rf *.h5ad
done