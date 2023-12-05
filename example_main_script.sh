############## Full Evaluate ###################
# Train in 35 datasets using MoCE with GNN setting (Scaffold Split)
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10 --sag_pool --open_dy --iattvec_loss --no_unsupervised

# Train in 35 datasets using MoCE with GNN setting (Random Split)
CUDA_VISIBLE_DEVICES=0 python main.py --task allseed-35ds-scaf-60ex --model TRMoCE --device cuda --epoch 100 --num_experts 60 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --use_valid --train_times 10 --sag_pool --open_dy --iattvec_loss --no_unsupervised


############## Ablation Study ###################
# PS: SAG+ATTLoss=ESProjection
# MoCE
CUDA_VISIBLE_DEVICES=0 python main.py --task ablation-scaf-moce-SAG-ATTLoss-ESLoss --model TRMoCE --device cuda --sag_pool --iattvec_loss --open_dy --epoch 100 --num_experts 30 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --dataset_comb "SkinReaction,CYP2C9_Veith,CYP3A4_Veith,AMES,CYP2C19_Veith,CYP1A2_Veith,CYP3A4_Substrate_CarbonMangels,Pgp_Broccatelli,hERG_Karim,DILI" --use_valid --train_times 10  --no_unsupervised
# MoCE w/o ESLoss
CUDA_VISIBLE_DEVICES=0 python main.py --task ablation-scaf-moce-SAG-ATTLoss --model TRMoCE --device cuda --sag_pool --iattvec_loss --epoch 100 --num_experts 30 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --dataset_comb "SkinReaction,CYP2C9_Veith,CYP3A4_Veith,AMES,CYP2C19_Veith,CYP1A2_Veith,CYP3A4_Substrate_CarbonMangels,Pgp_Broccatelli,hERG_Karim,DILI" --use_valid --train_times 10  --no_unsupervised
# MoCE w/o (ESLoss & ATTLoss)
CUDA_VISIBLE_DEVICES=0 python main.py --task ablation-scaf-moce-SAG --model TRMoCE --device cuda --sag_pool --epoch 100 --num_experts 30 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --dataset_comb "SkinReaction,CYP2C9_Veith,CYP3A4_Veith,AMES,CYP2C19_Veith,CYP1A2_Veith,CYP3A4_Substrate_CarbonMangels,Pgp_Broccatelli,hERG_Karim,DILI" --use_valid --train_times 10  --no_unsupervised
# MoCE w/o (ESLoss & ESProjection)
CUDA_VISIBLE_DEVICES=0 python main.py --task ablation-scaf-moe --model TRMoCE --device cuda --epoch 100 --num_experts 30 --emb_dim 300 --lr 1e-2 --eta_min 1e-8 --seed 0 --k 4 --emb_desc --dataset cls --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --num_g_experts 6 --dataset_comb "SkinReaction,CYP2C9_Veith,CYP3A4_Veith,AMES,CYP2C19_Veith,CYP1A2_Veith,CYP3A4_Substrate_CarbonMangels,Pgp_Broccatelli,hERG_Karim,DILI" --use_valid --train_times 10  --no_unsupervised
# BaseGNN
CUDA_VISIBLE_DEVICES=0 python main.py --task ablation-baseGNN --model GNN --device cuda --epoch 100 --emb_dim 1024 --lr 1e-2 --eta_min 1e-8 --seed 0 --train_times 10 --dataset cls --dataset_comb "SkinReaction,CYP2C9_Veith,CYP3A4_Veith,AMES,CYP2C19_Veith,CYP1A2_Veith,CYP3A4_Substrate_CarbonMangels,Pgp_Broccatelli,hERG_Karim,DILI" --use_valid --drop_ratio 0.3 --num_layer 4 --full_evaluate --full_eval_step 1 --batch_size 512 --JK last --graph_pooling sum --gnn_type gin --no_unsupervised