
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch_dct as dct


def base_train(model, trainloader, optimizer, scheduler, epoch, args, criterion_Cross):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        images, train_label = [_ for _ in batch]
        train_label = train_label.cuda()


        images = images.cuda()
        logits1, logits2, x1, part = model(images)

        logits1 = logits1[:, :args.base_class]
        logits2 = logits2[:, :args.base_class]

        loss=0
        logits =0


        mutual_loss = criterion_Cross(logits1, logits2) + criterion_Cross(logits2, logits1)
        loss1 = F.cross_entropy(logits1, train_label)
        loss2 = F.cross_entropy(logits2, train_label)

        loss = loss + (1 - args.part_weight) * loss1 + args.part_weight * loss2 + args.complex_weight * mutual_loss
        logits = logits + (1 - args.part_weight) * logits1 + args.part_weight * logits2



        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta



def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    embedding_list1 = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            size = data.shape[1:]

            model.module.mode = 'encoder'

            embedding, embedding1 = model(data)

            embedding_list.append(embedding.cpu())
            embedding_list1.append(embedding1.cpu())

            label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        embedding_list1 = torch.cat(embedding_list1, dim=0)

        label_list = torch.cat(label_list, dim=0)

        proto_list = []
        proto_list1 = []


        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this1 = embedding_list1[data_index.squeeze(-1)]

            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
            embedding_this1 = embedding_this1.mean(0)
            proto_list1.append(embedding_this1)

        proto_list = torch.stack(proto_list, dim=0)
        proto_list1 = torch.stack(proto_list1, dim=0)

        model.module.fc.weight.data[:args.base_class] = proto_list
        model.module.part_fc.weight.data[:args.base_class] = proto_list1

        return model


def test(model, testloader, epoch,args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va1 = Averager()

    label_list = []
    logit1 = []

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            size = data.shape[1:]
            logits1, logits2, _ , _ = model(data)

            logits1 = logits1[:, :test_class]
            logits2 = logits2[:, :test_class]
            agg_preds1 = logits1
            agg_preds2 = logits2

            loss = (1-args.part_weight)*F.cross_entropy(agg_preds1, test_label) + args.part_weight*F.cross_entropy(agg_preds2, test_label)

            acc1 = count_acc(agg_preds1, test_label)
            label_list.append(test_label.cpu())
            logit1.append(agg_preds1.cpu())

            vl.add(loss.item())
            va1.add(acc1)

        vl = vl.item()
        va1 = va1.item()


        if session == 0:
            print('Session {}, epo {}, test, loss={:.4f} acc1={:.4f}'.format(session, epoch, vl, va1))
        else:
            print('Session {}, test acc1={:.4f}'.format(session, va1))

    return vl, va1