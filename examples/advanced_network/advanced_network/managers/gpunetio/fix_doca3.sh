#!/usr/bin/env bash
# fix_doca3.sh  –  run from the directory that contains your .cu/.cpp files
for f in *.cu *.cpp; do
    echo "Processing $f ..."
    sed -i \
        -e 's/doca_gpu_dev_semaphore_get_custom_info_addr/doca_gpu_semaphore_get_custom_info_addr/g' \
        -e 's/doca_gpu_dev_eth_rxq_receive_block/doca_gpu_eth_rxq_receive_block/g' \
        -e 's/doca_gpu_dev_eth_rxq_get_buf/doca_gpu_eth_rxq_get_buf/g' \
        -e 's/doca_gpu_dev_buf_get_addr/doca_gpu_buf_get_addr/g' \
        -e 's/doca_gpu_dev_eth_rxq_get_buf_bytes/doca_gpu_eth_rxq_get_buf_bytes/g' \
        -e 's/doca_gpu_dev_semaphore_set_status/doca_gpu_semaphore_set_status/g' \
        -e 's/doca_gpu_dev_eth_txq_get_info/doca_gpu_eth_txq_get_info/g' \
        -e 's/doca_gpu_dev_buf_get_buf/doca_gpu_buf_arr_get_buf/g' \
        -e 's/doca_gpu_dev_eth_txq_send_enqueue_weak/doca_gpu_eth_txq_send_enqueue_weak/g' \
        -e 's/doca_gpu_dev_eth_txq_commit_weak/doca_gpu_eth_txq_commit_weak/g' \
        -e 's/doca_gpu_dev_eth_txq_push/doca_gpu_eth_txq_push/g' \
        "$f"
done
echo "All done."
