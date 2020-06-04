import argparse


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='lrw')
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')

	parser.add_argument('--lrs3_root', type=str, default='./dataset/lrs3')

	parser.add_argument('--dataset', type=str, default='lrs3')

	parser.add_argument('--encode', type=str, default='233')
	parser.add_argument('--middle', type=str, default='tc')
	parser.add_argument('--decode', type=str, default='pass')
	parser.add_argument('--load', type=str, default='./save/lrw/check/model_12_4200.pth.tar')
	parser.add_argument('--min_frame', type=int, default=29)
	parser.add_argument('--topw', type=int, default=300)

	parser.add_argument('--epoches', type=int, default=60)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--loss_smooth', type=float, default=0.1)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--in_channel', type=int, default=1)
	parser.add_argument('--out_channel', type=int, default=500)

	opt = parser.parse_args()

	return opt