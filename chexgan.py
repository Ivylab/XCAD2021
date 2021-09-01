import random
import warnings
import torch.backends.cudnn as cudnn
from generation import *
from model_Unet import *
from model_Backbone import *
from ops import *
from dataloader import *
from glob import glob

warnings.filterwarnings("ignore")


class CheXGAN(object):
    def __init__(self, args):
        # Define Variables
        self.data_dir = args.data_dir
        self.v_seed = 42

    def build_model(self):
        # Build Backbone Network (DenseNet121)
        self.backbone_net = DenseNet121(14, 0.2).to("cuda:0")
        self.backbone_net.apply(weights_init).cuda()
        self.backbone_net = torch.nn.DataParallel(self.backbone_net).cuda()

    def build_gan_model(self):
        # Build GAN (U Net)
        self.NetG = U_net().to("cuda:0")
        self.NetD = Discriminator().to("cuda:0")
        self.NetG = torch.nn.DataParallel(self.NetG).cuda()
        self.NetD = torch.nn.DataParallel(self.NetD).cuda()

    @property
    def model_dir(self):
        return "{}_{}".format("CheXNet", "ChestX-ray14")

    @property
    def gan_dir(self):
        return "{}_{}".format("GAN", "ChestX-ray14")

    def load(self, dir, epoch, counter):
        params = torch.load(
            os.path.join(
                dir, "ChestX-ray14" + "_params_%07d_%07d.pt" % (epoch, counter)
            )
        )

        self.backbone_net.module.load_state_dict(params["backbone_net"])

    def gan_load(self, dir, epoch, counter):
        params = torch.load(
            os.path.join(
                dir, "ChestX-ray14" + "_params_%07d_%07d.pt" % (epoch, counter)
            )
        )

        self.NetG.module.load_state_dict(params["NetG"])
        self.NetD.module.load_state_dict(params["NetD"])

    def generation(self):
        # Set up Random Seed
        random.seed(self.v_seed)

        # Images to be Saved at
        chexgan_dir = "outputs"
        check_dir(chexgan_dir)

        # Load Backbone Model Parameters
        model = self.backbone_net.to("cuda:0")
        model_list = glob(os.path.join("parameters", self.model_dir, "*.pt"))
        if not len(model_list) == 0:
            model_list.sort()
            start_epoch = int(model_list[-1].split("_")[-2])
            counter = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load(os.path.join("parameters", self.model_dir), start_epoch, counter)
            print("\n [*] Load SUCCESS")
            print(
                " [*] Params from epoch %d / counter %d LOADED !\n"
                % (start_epoch, counter)
            )
        else:
            print("\n [*] Load FAIL")
            print(" [*] Train Model FIRST !\n")
        model.eval()

        # Load GAN Parameters
        gmodel = self.NetG
        gmodel_list = glob(os.path.join("parameters", self.gan_dir, "*.pt"))
        if not len(gmodel_list) == 0:
            gmodel_list.sort()
            start_epoch = int(gmodel_list[-1].split("_")[-2])
            counter = int(gmodel_list[-1].split("_")[-1].split(".")[0])
            self.gan_load(
                os.path.join("parameters", self.gan_dir), start_epoch, counter
            )
            print("\n [*] Load SUCCESS")
            print(
                " [*] Params from epoch %d / counter %d LOADED !\n"
                % (start_epoch, counter)
            )
        else:
            print("\n [*] Load FAIL")
            print(" [*] Train Model FIRST !\n")
        gmodel.eval()
        cudnn.benchmark = True

        # Declare Dataloader
        test_dataloader = get_xai_data_loader(
            self.data_dir + "image",
            self.data_dir + "list/demo_bbox.txt",
            1500,
            gray=True,
            gan=True,
        )
        in_img, gt, img_root, bbox, boxlabel = test_dataloader.__iter__().next()

        for i_idx in range(in_img.size(0)):
            generateImg(
                in_img,
                boxlabel,
                img_root,
                bbox,
                i_idx,
                model,
                gmodel,
                "cuda:0",
                chexgan_dir,
                visual=True,
            )
