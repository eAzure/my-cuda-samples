# GEMM implementation
## implementation

- 0-naive.cu : naive implement (done)
- 1-others.cu : add Matrix View (done)
- 1-matrix_view.cu (done)
- 2-wmma.cu : (doing)
- cublas.cu

## wmma nvcc
nvcc -arch=sm_xx 2-wmma.cu -o wmma

## nsight compute

```
# get the bank conflict information
sudo ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./xxx
```