import regex as re
from MedCLIP_Datasets import *
import CLIP_Embedding
import torch
import torch.nn as nn
import Vision_Model
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn import metrics
from torch.autograd import Variable


#Organizing training experiments
#Retrieves from exp path, or starts a new one
def getExperiment(args, mp):
    if args.debug:
        return "debug"

    if args.resume and args.resume > 0:
        fp =  os.path.join(mp, 'exp'+str(args.resume))
        if os.path.exists(fp):
            return fp
        else:
            raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

    if not os.listdir(os.path.join(mp)): #empty list - no experiments exist
        if args.resume:
            raise Exception("No experiment exist, cannot resume last one.")
        print("No models exist, creating directory")
        fp = os.path.join(mp, 'exp1')
    else:
        all_files = os.listdir(os.path.join(mp))
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        if not args.resume:
            highest = highest + 1
        fp = os.path.join(mp, 'exp'+str(highest))
    return fp

#Document the args used in experiment
def writeArgs(fp, args):
    '''
    Document args used to train
    '''
    writestr = str(args)
    with open(fp + '/args.txt', 'w') as f:
        f.write(writestr)

#Initializes from experiment
def startExperiment(args, fp, cnn=False, pretrained=True):
    '''
    Initialize variables for experiment:
    start (epoch), je_model, params, optimizer, best_val_loss
    '''
    je_model = CLIP_Embedding.MedCLIP(eval=False).to(device) if not cnn else Vision_Model.getCNN(pretrained=pretrained, classifier=True).to(device)
    params = list(je_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
    if fp == "debug":
        return 0, je_model, params, optimizer, 100000

    if args.resume:
        if os.listdir(os.path.join(fp)):
            all_files = os.listdir(os.path.join(fp))
            je_files = [file for file in all_files if 'je_model' in file]
            num = [int(re.search('\d+', file).group(0)) for file in je_files]
            highest = np.argmax(np.array(num))
            loadpath = os.path.join(fp, np.array(je_files)[highest])
            print("Loading " + loadpath)
            checkpoint = torch.load(loadpath)
            je_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss'] if 'best_val_loss' in checkpoint.keys() else checkpoint['val_loss']
        else:
            raise Exception("Experiment doesn't exist: " + fp)
    else:
        print("Starting from scratch")
        start = 0
        best_val_loss = 1000000000
        if not args.debug:
            os.makedirs(fp)
            writeArgs(fp, args)
    return start, je_model, params, optimizer, best_val_loss

# GLoRIA cosine similarity
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

# GLoRIA Local Attention Loss
def alt_attn_penalty(cross_weights, context, word_embs, soft = nn.Softmax(dim=2)):
    
    N, T, E = word_embs.shape #word emb shape - NTE
    attn_loss = 0
    eps = 1e-7
    losses = []

    for c in cross_weights: # c - N x T x P - 1 iteration only

        ############# Calculate image to text posterior probability #############

        #Get attention
        # --> N x T x P (softmax along image patches dimension)
        attn = soft(c) + eps #attn = soft(c * temp1)
        # --> N x P x T
        attnT = torch.transpose(attn, 1, 2).contiguous() # N x P x T

        #print(context)

        # (N x E x P)(N x P x T)
        # --> N x E x T
        weiContext = torch.bmm(context, attnT) #context - patch embeddings
        weiContext = weiContext.transpose(1, 2).contiguous() #NTE - rows are c1 ... cT for each batch

        # --> N x T x E
        word = word_embs.view(N*T, -1) #N*T, E
        weiContext = weiContext.view(N*T, -1) #N*T, E

        #print(word)
        #print(weiContext)

        row_sim = cosine_similarity(word, weiContext) #[N*T, 1]
        row_sim = row_sim.view(N, T) # cosine similarity for jth textual feature, split into i batches

        #print(row_sim)

        # similarity of one image to each token
        row_sim = row_sim.exp_() #row_sim = row_sim.mul_(temp2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) #[N, 1]
        row_sim = torch.log(row_sim) #[N, 1]
        # row_sim = row_sim ^ temp3

        ############# Calculate reverse posterior probability #############
 
        # --> N x E x T
        weiContextT = torch.bmm(context, attnT) #N x E x T
        wordT = word_embs.transpose(1, 2).contiguous() #N x E x T

        wordT = wordT.view(N*E, -1) #N*E, T
        weiContextT = weiContextT.view(N*E, -1) #N*E, T

        col_sim = cosine_similarity(wordT, weiContextT)
        col_sim = col_sim.view(N, E)

        # similarity of one text feature to each image
        col_sim = col_sim.exp_()
        col_sim = col_sim.sum(dim=1, keepdim=True) #[N, 1]
        col_sim = torch.log(col_sim)

        labels = torch.FloatTensor(np.arange(N))
        
        #print(row_sim.squeeze(1))
        #print(labels)

        ############# Calculate cross entropy losses #############

        loss0 = nn.CrossEntropyLoss()(row_sim.squeeze(1), labels.to(device)) #~300
        loss1 = nn.CrossEntropyLoss()(col_sim.squeeze(1), labels.to(device)) #~300

        loss = (loss0 * 0.5 + loss1 * 0.5) / N #temperature functions in place of lambda function
        #print(loss)
        losses.append(loss.cpu().detach())
        attn_loss += loss

    return attn_loss, losses

# TIER Penalty
def attn_penalty(cross_weights, soft = nn.Softmax(dim=2), lam = (0.0, 0.0)):
    attn_loss = 0 #lam patch * entropy of each word similarity (across patches)
    losses = []   #lam words * entropy of each patch similarity (across words)
    eps = 1e-7

    for c in cross_weights:
        entropy = soft(c) + eps #NTP; softmax dim=2
        entropy = -entropy * torch.log(entropy) #calculations still in dim 2
        entropy_text = torch.sum(entropy, dim=2) #N, T
        entropy_text = torch.mean(entropy_text, dim=(1, 0)) #1

        entropy_im = soft(c.permute(0, 2, 1)) + eps
        entropy_im = -entropy_im * torch.log(entropy_im)
        entropy_im = torch.sum(entropy_im, dim=2) #N, P
        entropy_im = torch.mean(entropy_im, dim=(1, 0)) # 1

        loss = (entropy_text * float(lam[0])) + (entropy_im * float(lam[1]))
        
        losses.append(loss.cpu().detach())
        attn_loss += loss
    return attn_loss, losses #1

# Standard CLIP loss, for a list of images and texts.
# When len is 1, just normal CLIP loss. With multiple image augmentations, also do image-image contrasting
def clip_loss(im_logits, aug_logits = None, loss_weight = 1, criterion = nn.CrossEntropyLoss()):
    text_logits = [im.t() for im in im_logits]
    clip_loss = 0
    losses = []
    for i in np.arange(len(im_logits)): #for each image augmentation-text matrix (usually just 1)
        #calculate symmetric loss function, exactly from CLIP pseudocode
        samp = torch.tensor(np.arange(im_logits[i].shape[0]))
        loss_a = criterion(im_logits[i], samp.to(device))
        loss_b = criterion(text_logits[i], samp.to(device))
        closs = (loss_a + loss_b) / 2
        losses.append(closs.cpu().detach())
        clip_loss += closs * loss_weight

    if aug_logits is not None: #for each image augmentation-image augmentation matrix (usually 0)
        for i in np.arange(len(aug_logits)):
            samp = torch.tensor(np.arange(im_logits[i].shape[0]))
            imloss = criterion(im_logits[i], samp.to(device))
            losses.append(imloss.cpu().detach())
            clip_loss += imloss
    
    #Check that number of losses = number of CLIP losses from each image-text pair + number of combinations between each image pair
    assert len(losses) == int((len(im_logits) + (len(im_logits) * (len(im_logits) -1)/2.0)))
    return clip_loss, losses

#Compute total loss
def compute_loss(je_model, samples, args, attn_lam_words = 0.0, attn_lam_patches = 0.0):
    ims = samples['images']
    texts = samples['texts']
    im_logits, crosses, aug_logits, stack_patches, word_embs = je_model(ims, texts)
    cl, cl_losses = clip_loss(im_logits, aug_logits)
    #attn_pen, attn_losses = alt_attn_penalty(crosses, stack_patches, word_embs)
    attn_pen, attn_losses = attn_penalty(crosses, lam=(attn_lam_words, attn_lam_patches))
    cl_count = len(cl_losses)
    attn_count = len(attn_losses)
    loss = cl / cl_count + attn_pen / attn_count
    all_losses = cl_losses + attn_losses
    return loss, torch.tensor(all_losses)

#Train TIER CLIP
def train(train_data_loader, je_model, args, epoch, optimizer, total_step_mimic=-1, lam_words = -1, lam_patch = -1):
    mean_loss, mean_losses, ct = 0.0, 0.0, 0
    if lam_words < 0:
        lam_words = args.lam_words #lambda words
    if lam_patch < 0:
        lam_patch = args.lam_patches #lambda patches

    je_model.train(True)
    for i, samples in enumerate(train_data_loader):
        je_model.zero_grad(set_to_none=True)
        loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=lam_words, attn_lam_patches = lam_patch)
        # Forward, backward and optimize
        loss.backward()
        optimizer.step()
        if total_step_mimic > 0:
            if i % args.log_step == 0:
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step_mimic, loss.item()))
                print(all_losses)
        l = loss.cpu().detach().numpy()
        mean_loss += l
        mean_losses += all_losses
        ct += 1
    if ct > 0:
        mean_loss = mean_loss/ct
        mean_losses = mean_losses/ct

    return mean_loss, mean_losses

#Validate TIER CLIP
def validate(val_data_loader, je_model, args):
    val_losses = []
    avg_loss, ct = 0.0, 0
    je_model.train(False)
    with torch.no_grad():
        for j, samples in enumerate(val_data_loader):
            loss, all_losses = compute_loss(je_model, samples, args, attn_lam_words=args.lam_words, attn_lam_patches = args.lam_patches)
            val_losses.append(all_losses.view(-1,1))
            avg_loss += loss
            ct += 1
    avg_loss = avg_loss/ct

    val_losses = torch.cat(val_losses, dim=1) #num batches x num losses
    avg_losses = torch.mean(val_losses, dim=1)

    if avg_losses.shape[0] == 5:
        names = ['im1-t', 'im2-t', 'im1-im2', 'im1-cross', 'im2-cross']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)
    elif avg_losses.shape[0] == 2:
        names = ['im-t', 'attn im-t']
        lossstr = ""
        for i in range(len(names)):
            lossstr += (", " + names[i] + ": " + str(avg_losses[i].item()))
        print("Val losses" + lossstr)

    return avg_loss.item(), avg_losses

# Loss for supervised baseline
def b_loss(cnn_model, samples, args, heads, criterion=torch.nn.BCEWithLogitsLoss(reduction='mean')):
    im = samples['images'][0].to(device)
    impreds = cnn_model(im).class_logits.to(device)
    impreds = impreds.squeeze(dim=2)
    labels = samples['labels']
    losses = torch.zeros(len(heads))
    for i, h in enumerate(heads):
        label = labels[h]
        label[label == -1.0] = float('nan')
        label[label == 0.0] = 0
        label[label == 1.0] = 1
        label = label.float().to(device)
        mypreds = impreds[torch.logical_not(torch.isnan(label)), i]
        mylabels = label[torch.logical_not(torch.isnan(label))]
        losses[i] = criterion(mypreds, mylabels)
    losses = losses[torch.logical_not(torch.isnan(losses))]
    loss = torch.mean(losses)
    if torch.isnan(loss):
        loss = 0
    return loss

# Train supervised
def train_vision(train_data_loader, cnn_model, args, epoch, optimizer, totstep= None, heads=None, list_mods = False, je_inds = [], all_pos_embeds=None, all_neg_embeds=None):
    if list_mods:
        full_loss = [0 for i in cnn_model]
    else:
        full_loss = [0]
        cnn_model = [cnn_model]
        optimizer = [optimizer]

    for i, samples in enumerate(train_data_loader):
        for j, cnn_mod in enumerate(cnn_model):
            cnn_mod.train(True)
            cnn_mod.zero_grad(set_to_none=True)
            if j not in je_inds:
                loss = b_loss(cnn_mod, samples, args, heads)
            else:
                cnn_mod.cnn.train(True)
                loss = pseudo_b_loss(cnn_mod, samples, args, heads, all_pos_embeds[j], all_neg_embeds[j])
            loss.backward()
            # Forward, backward and optimize

            optimizer[j].step()
            if (totstep) and (i % args.log_step == 0):
                print('MIMIC Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, totstep,
                                                                               loss.item()))
            full_loss[j] += loss.detach().cpu().item()
    avg_loss = [f/(i + 1) for f in full_loss]
    return avg_loss[0] if not list_mods else avg_loss

# Validate supervised
def validate_vision(val_data_loader, cnn_model, args=None, heads=None, list_mods = False):
    if list_mods:
        avg_loss = [0.0 for mod in cnn_model]
    else:
        avg_loss = [0.0]
        cnn_model = [cnn_model]

    ct = 0
    with torch.no_grad():
        for i, samples in enumerate(val_data_loader):
            for j, mod in enumerate(cnn_model):
                mod.train(False)
                loss = b_loss(mod, samples, args, heads)
                avg_loss[j] += loss
            ct += 1
    avg_loss = [a/ ct for a in avg_loss]
    print("Val loss: " + str(avg_loss))
    return avg_loss[0].item() if not list_mods else [a.item() for a in avg_loss]

# Get labels from datafrae
def getLabels(df, heads, replace_nan = False):
    labels = None
    for i, h in enumerate(heads):
        label = df[h].float()
        label[label==-1.0] = float('nan')
        if replace_nan:
            label = torch.nan_to_num(label)
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c

# Get labels from chexperrt radiologist
def getRadLabels(df, heads, suffix):
    labels = None
    for i, h in enumerate(heads):
        label = df[h + '_' + suffix].float()
        label[label==-1.0] = float('nan')
        if labels is None:
            labels = label
            labels = labels[:, None]
        else:
            labels = torch.cat((labels, label[:, None]), axis=1)

    return labels #N x c

# Compute zeroshot accuracy from a given checkpoint
def getZeroShotAcc(checkpoint, DL, heads = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], usecheckpoint=True):
    aucs, tprs, fprs, thresholds = {}, {}, {}, {}
    if usecheckpoint:
        clip_model = CLIP_Embedding.getCLIPModel(checkpoint=checkpoint, eval=True)
    else:
        clip_model = checkpoint
    preds, labels = get_all_preds(DL, clip_model, similarity=True, heads = heads, normalization=True)
    preds = preds[0].cpu().detach().numpy()
    targs = labels.cpu().int().detach().numpy()
    for i, h in enumerate(heads):
        myt = targs[:, i]
        myp = preds[:, i]
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(myt, myp)
        aucs[h] = metrics.auc(fprs[h], tprs[h])
    return np.mean(np.array([aucs[h] for h in heads]))

def get_all_preds(DL, mod=None,similarity=False, im_embeds=False, only_labels=False, getlabels=True,
                  heads = [], convirt=True,  normalization=False,
                  im_classifier=False, autochoose='', also_studies=False, getrad=False):

    if autochoose != '':
        if 'Zeroshot' in autochoose or 'CNN' not in autochoose:
            similarity, normalization =True, True
        elif 'Finetuned' in autochoose or 'CNN' in autochoose:
            im_classifier=True
    if getrad:
        rad1, rad2, rad3 = [], [], []

    if only_labels:
        tstudies = []
        tt = []
        for i, samples in enumerate(DL):
            if also_studies:
                tstudies+= samples['study']
            labels = getLabels(samples['labels'], heads)
            tt.append(labels)
        tt = torch.cat(tt, axis=0)
        if also_studies:
            return tt, tstudies
        return tt

    with torch.no_grad():
        #Get query embeddings
        if similarity:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=convirt)
            embed_list = [label_embeds[h][None, :] for h in heads]
            label_embeds = torch.cat(embed_list, dim=0)
            label_embeds = label_embeds/label_embeds.norm(dim=1, keepdim=True)
            if normalization:
                neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=convirt, getneg=True)
                neg_embed_list = [neg_label_embeds[h][None, :] for h in heads]
                neg_label_embeds = torch.cat(neg_embed_list, dim=0)
                neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)

        # Get all image embeddings/labels
        for i, samples in enumerate(DL):
            images = samples['images']
            if i == 0:
                tt = []
                tps = [[] for im in images]

            if im_classifier:
                mod.train(False)
                list_preds = [mod(im.to(device)).class_logits.squeeze(dim=2) for im in images]
                if list_preds[0].shape[1] > len(heads):
                    list_preds = [l[:, :len(heads)] for l in list_preds]

            elif similarity:
                list_im_embeds = mod.get_im_embeddings(images, only_ims = True)
                list_im_embeds = [im_embeds/im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]
                # N P E x c E = N c
                list_preds = [im_embeds @ label_embeds.t() for im_embeds in list_im_embeds]  # N c

                if normalization:
                    list_neg_preds = [im_embeds @ neg_label_embeds.t() for im_embeds in list_im_embeds] #N c
                    for i, sim_preds in enumerate(list_preds):
                        list_preds[i] = list_preds[i]- list_neg_preds[i] # if just subtraction

            elif im_embeds:
                list_im_embeds = mod.get_im_embeddings(images, only_ims=True)
                list_preds = [im_embeds / im_embeds.norm(dim=1, keepdim=True) for im_embeds in list_im_embeds]

            labels = getLabels(samples['labels'], heads) if getlabels else None
            tt.append(labels) if getlabels else None
            if getrad:
                rad1.append(getRadLabels(samples['labels'], heads, 'rad1'))
                rad2.append(getRadLabels(samples['labels'], heads, 'rad2'))
                rad3.append(getRadLabels(samples['labels'], heads, 'rad3'))

            for j, pred in enumerate(list_preds):
                tps[j].append(pred.cpu())

        #collect all labels and predictions
        tplist = [torch.cat(tp, axis=0) for tp in tps]
        tt = torch.cat(tt, axis=0) if getlabels else None
        if getrad:
            rad1 = torch.cat(rad1, axis=0)
            rad2 = torch.cat(rad2, axis=0)
            rad3 = torch.cat(rad3, axis=0)

        if getrad:
            return tplist, tt, rad1, rad2, rad3
        else:
            return tplist, tt

def getPadPredictions(DL, models = None, ensemble=True, soft_norm=True, only_labels = False):
    tt = []
    for z, sample in enumerate(DL):
        if z == 0:
            name_list = list(sample['labels'].keys())
            label_embeds_all = []
            neg_label_embeds_all = []
            if not only_labels:
                tps_models = [[] for model in models]
                with torch.no_grad():
                    for j, m in enumerate(models):
                        label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list)
                        embed_list = [label_embeds[h][None, :] for h in name_list]
                        label_embeds = torch.cat(embed_list, dim=0)
                        label_embeds = label_embeds / label_embeds.norm(dim=1, keepdim=True)
                        label_embeds_all.append(label_embeds)
                        print("labelembed shape",label_embeds_all[0].shape)

                        neg_label_embeds = CLIP_Embedding.getLabelEmbeddings(m, name_list, customdescs=name_list, getneg=True)
                        neg_embed_list = [neg_label_embeds[h][None, :] for h in name_list]
                        neg_label_embeds = torch.cat(neg_embed_list, dim=0)
                        neg_label_embeds = neg_label_embeds / neg_label_embeds.norm(dim=1, keepdim=True)
                        neg_label_embeds_all.append(neg_label_embeds)
                    l = label_embeds[2, :]
                    n = neg_label_embeds[2, :]
                    l1 = label_embeds[1, :]
                    print(torch.sum(l * n), torch.sum(l * l1))

        labels = getLabels(sample['labels'], name_list)
        tt.append(labels)
        if only_labels:
            continue

        images = sample['images']
        for j, m in enumerate(models):
            with torch.no_grad():
                im_embeds = m.get_im_embeddings(images, only_ims=True)[0]
                im_embeds = im_embeds / im_embeds.norm(dim=1, keepdim=True)
                # N P E x c E = N c
                preds = im_embeds @ label_embeds_all[j].t()
                if soft_norm:
                    neg_preds = im_embeds @ neg_label_embeds_all[j].t() # N c
                    preds= torch.stack([preds[:, :, None], neg_preds[:, :, None]],
                                                    dim=2)  # N C 2
                    preds = torch.nn.Softmax(dim=2)(preds)[:, :, 0].squeeze(dim=2)

                tps_models[j].append(preds.cpu())

    if not only_labels:
        tps_models = [torch.cat(tps, dim=0) for tps in tps_models]#list of models, list of ims,tensor preds
    tt = torch.cat(tt, dim=0)
    if only_labels:
        return tt

    if ensemble:
        if len(models)>1:
            tps_models = [modelpred[None, :, :] for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = torch.cat(tps_models, dim = 0).mean(dim = 0, keepdim=False) #stacked prediction ten
        else:
            tps_models = [modelpred for modelpred in tps_models]  # list of models prediction tensors
            tps_avg = tps_models[0]
        print(tps_avg.shape)
        print(tt.shape)
        assert tt.shape == tps_avg.shape
        tps_models = tps_avg

    return tps_models, tt, name_list


def normalize(image, getOne = True):
    img = torch.clone(image)
    img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
    img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
    img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
    if getOne:
        img = img.permute(0, 2, 3, 1)[0, :, :, :].squeeze()
    else:
        img = img.permute(0, 2, 3, 1)
    return img

def getLabelSimilarities(mod, heads, label_embeds=None, compare_mimic = False):
    with torch.no_grad():
        if compare_mimic:
            label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            label_embeds_mimic = CLIP_Embedding.getLabelEmbeddings(mod, heads, convirt=False)
            for i, h in enumerate(heads):
                print(h, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                       label_embeds_mimic[h] / label_embeds_mimic[h].norm(dim=-1, keepdim=True)).cpu())
        else:
            if not label_embeds:
                label_embeds = CLIP_Embedding.getLabelEmbeddings(mod, heads)
            for i, h in enumerate(heads):
                for j, h2 in enumerate(heads):
                    if i < j:
                        print(h, h2, torch.dot(label_embeds[h] / label_embeds[h].norm(dim=-1, keepdim=True),
                                               label_embeds[h2] / label_embeds[h2].norm(dim=-1, keepdim=True)).cpu())



