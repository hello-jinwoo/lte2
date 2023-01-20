bicubic_pytorch github 알아서 설치하고 path도 알아서 맞추시고 (난 root에 깔았음)


bicubic_pytorch github으로 학습 (liif, lte 용)
--> python train1.py --config configs/v1.yaml --gpu 0

기존 코드 그대로 트레이닝 셋팅으로 학습
--> python train2.py --config configs/v2.yaml --gpu 0

bicubic_pytorch 학습 (rgb 이미지로 바로 아웃풋) 
--> python train2.py --config configs/v3.yaml --gpu 0

weight 저장은 config 입력시 [name].yaml 에서 [name]으로 save 폴더에 저장함

데이터셋 path는 datasets 안에 다 넣으면 됨 

코드 버그나 수정할 거 있으면 자유롭게 의견부탁스

python train3.py --config configs/v3.yaml --gpu 0 이거하면됨 (edsr-baseline-bicubic 학습)

