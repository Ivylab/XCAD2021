from ops import *
from skimage.transform import resize


def generateImg(
    imgBatch,
    boxLabelBatch,
    rootBatch,
    bboxCoordinateBatch,
    index,
    model,
    gmodel,
    device,
    chexgan_dir,
    visual=False,
):
    # Disable Gradient Tracing
    for param in model.parameters():
        param.requires_grad_(False)

    # Variables for Image Transform
    imgTransformMean = (
        torch.tensor([0.485, 0.456, 0.406]).float().to(device)[None, :, None, None]
    )
    imgTransformSTD = (
        torch.tensor([0.229, 0.224, 0.225]).float().to(device)[None, :, None, None]
    )

    # Declare Function
    sigmoid = torch.nn.Sigmoid()

    # Select Samples from Batch
    inputImg = imgBatch[index].unsqueeze(0).to(device)
    boxLabel = (boxLabelBatch[index] != 0).nonzero()
    root = rootBatch[index]
    bboxCoordinate = bboxCoordinateBatch[index]

    # Set Backbone Network and GAN to CUDA
    model = model.to(device)
    gmodel = gmodel.to(device)

    # Make Transform for the Input
    ori_input = (
        ((inputImg.repeat(1, 3, 1, 1) + 1.0) / 2.0).to(device) - imgTransformMean
    ) / imgTransformSTD

    # Make Inference
    logits = model(Variable(ori_input).to(device))
    originalPrediction = sigmoid(logits)
    oneHotPrediction = (originalPrediction >= 0.5) * 1.0

    # Create Conditioned Z Vector
    vectorZ = conditionZ(oneHotPrediction.clone(), originalPrediction.clone())

    oneHotPredictionTF = (oneHotPrediction >= 0.5).nonzero()

    # Check if the Prediction Mathches the GT Label
    try:
        isAbnormal = (
            len((torch.eq(boxLabel, oneHotPredictionTF.cpu())[:, 1] == True).nonzero())
            != 0
        )
    except:
        isAbnormal = False

    if isAbnormal == True:

        # Generated Image from GAN
        generatedImg = gmodel(
            Variable(inputImg).to(device), vectorZ.to(device).detach()
        )

        ori_img = np.transpose(
            ((inputImg.cpu() + 1.0) / 2.0).repeat(1, 3, 1, 1).detach().numpy(),
            [0, 2, 3, 1],
        )
        gen_img = np.transpose(
            ((generatedImg.cpu() + 1.0) / 2.0).repeat(1, 3, 1, 1).detach().numpy(),
            [0, 2, 3, 1],
        )

        # Post-process Saliency Map
        saliency = tensor_centercrop(torch.abs(generatedImg - inputImg), keepdim=True)
        resizedSaliency = resize(
            saliency.cpu().detach().numpy(), (1024, 1024), anti_aliasing=True
        )

        if visual == True:
            # Output Visualization Results
            _ = visualization(
                resizedSaliency,
                root,
                bboxCoordinate,
                boxLabel,
                index,
                chexgan_dir,
            )
            save_ori = Image.fromarray((ori_img[0] * 255).astype(np.uint8))
            save_gen = Image.fromarray((gen_img[0] * 255).astype(np.uint8))

            # Draw Bounding Box for the Original and Generated Image
            bbox_draw1 = ImageDraw.Draw(save_ori)
            bbox_draw1.rectangle(
                [
                    (
                        round(bboxCoordinate[0].item() / 1024 * 224),
                        round(bboxCoordinate[1].item() / 1024 * 224),
                    ),
                    (
                        round(
                            (bboxCoordinate[0] + bboxCoordinate[2]).item() / 1024 * 224
                        ),
                        round(
                            (bboxCoordinate[1] + bboxCoordinate[3]).item() / 1024 * 224
                        ),
                    ),
                ],
                outline="red",
                width=3,
            )
            bbox_draw2 = ImageDraw.Draw(save_gen)
            bbox_draw2.rectangle(
                [
                    (
                        round(bboxCoordinate[0].item() / 1024 * 224),
                        round(bboxCoordinate[1].item() / 1024 * 224),
                    ),
                    (
                        round(
                            (bboxCoordinate[0] + bboxCoordinate[2]).item() / 1024 * 224
                        ),
                        round(
                            (bboxCoordinate[1] + bboxCoordinate[3]).item() / 1024 * 224
                        ),
                    ),
                ],
                outline="red",
                width=3,
            )

            # Output Image
            save_ori.save(chexgan_dir + "/" + "Original_Image_%3d.png" % (index))
            save_gen.save(chexgan_dir + "/" + "Generated_Image_%3d.png" % (index))

            print(("[*] file: CheXGAN%3d.png saved\n" % (index)))
