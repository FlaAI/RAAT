CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
	--fname cifar10_preres18 \
	--model PreActResNet18 \
	--chkpt-iters 10 \
	--lr-factor 1.5 \
	--beta 1.0 \
	--num-classes 10

#CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py --eval \
#  --fname cifar10_preres18 \
#  --model PreActResNet18 \
#  --resume 200 \
#  --num-classes 10
