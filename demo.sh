for attack in  PGD
do
	for epsilon in 0.00784 0.03137 0.06275
	do
		python train_tf_wideRes.py --test_mode 9 --attack_method $attack --epsilon $epsilon | tee ./logs/$attack\_$epsilon.txt
	done
done
:<<eof
#for attack in  PGD  Momentum
#do
#	for epsilon in 0.00784 0.03137 0.06275
#	do
#		python train_tf_wideRes.py --test_mode 9 --attack_method $attack --epsilon $epsilon
#	done
#done
#
#for attack in STA FGSM
#do
#	python train_tf_wideRes.py --test_mode 9 --attack_method $attack --epsilon 0.00784
#done
eof
