# cp -r ../../data/real/diag/healthy_hema/multimap/BMMC/* .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo rm -rf ../bmmc_healthyhema_1000/multimap
# sudo mv multimap ../bmmc_healthyhema_1000/multimap
# rm -rf *.h5ad

# cp -r ../../data/real/diag/mouse_brain_cortex/multimap/*.h5ad .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo rm -rf ../mbc/multimap
# sudo mv multimap ../mbc/multimap
# rm -rf *.h5ad

# cp -r ../../data/real/diag/Xichen/multimap/*.h5ad .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo rm -rf ../spleen/multimap
# sudo mv multimap ../spleen/multimap
# rm -rf *.h5ad

# cp -r ../../data/real/MOp/multimap/*.h5ad .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo mv multimap ../MOp/multimap
# rm -rf *.h5ad

# cp -r ../../data/real/ASAP-PBMC/multimap/*.h5ad .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/CFRM/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo mv multimap ../pbmc/multimap
# rm -rf *.h5ad

# cp -r ../../data/real/hori/Pancreas/multimap/*.h5ad .
# sudo docker build -f Dockerfile -t run_multimap .
# sudo docker run -v /localscratch/ziqi/scMoMaT/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap
# sudo chmod -R o+rw /localscratch/ziqi/scMoMaT/test/multimap_script/multimap
# sudo mv multimap ../pancreas/multimap
# rm -rf *.h5ad

# simulated
# for num in 7 # 1 2 3 4 5 6 7 8 9 10
# do
#     cp -r ../../data/simulated/6b16c_test_${num}/imbalanced/multimap/*.h5ad .
#     sudo docker build -f Dockerfile -t run_multimap .
#     sudo docker run -v /localscratch/ziqi/scMoMaT/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap 
#     mkdir -p ../simulated/6b16c_test_${num}/imbalanced
#     sudo chown -R 1059113:372345 multimap
#     sudo mv multimap ../simulated/6b16c_test_${num}/imbalanced/
#     rm -rf *.h5ad
# done

# simulated protein
for num in 1 2 3 4 5 6 7 8 9 10
do
    cp -r ../../data/simulated/6b16c_test_${num}/unequal/multimap/*.h5ad .
    sudo docker build -f Dockerfile -t run_multimap .
    sudo docker run -v /localscratch/ziqi/scMoMaT/test/multimap_script/multimap:/test_multimap/outputs --rm run_multimap 
    mkdir -p ../simulated/6b16c_test_${num}/protein_scenario2
    sudo chown -R 1059113:372345 multimap
    sudo rm -rf ../simulated/6b16c_test_${num}/protein_scenario2/multimap
    sudo mv multimap ../simulated/6b16c_test_${num}/protein_scenario2/
    rm -rf *.h5ad
done