############## Full Evaluate ###################
# Train in 35 datasets using in Multi-task (Scaffold Split)
# MTL-GNN
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-gnn --model GNN --device cuda --epoch 100 --emb_dim 1024 --lr 1e-2 --eta_min 1e-8 --seed 0 --train_times 10 --dataset cls --dataset cls --drop_ratio 0.3 --use_valid --full_evaluate --num_layer 4 --full_eval_step 1 --batch_size 512  --JK last --graph_pooling sum --gnn_type gin

# GNN-MoE
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex-moe --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10

# GNN-MoE + ES_Projection + ES_Loss  (GNN-MoCE)
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex-moce-dotSAG-ESLoss --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10 --sag_pool --open_dy --sag_att_type dot

# ablation part
# GNN-MoE + ES_Projection
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex-moce-dotSAG --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10 --sag_pool --sag_att_type dot
# GNN-MoE + ES_Loss
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex-moce-ESLoss --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10 --open_dy