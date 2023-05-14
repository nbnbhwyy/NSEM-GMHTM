from audioop import bias
import os
from pickle import FALSE, TRUE
from re import X
import time
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch import nn, optim
from customized_linear import CustomizedLinear
from utils import *
from learning_utils import *
from scipy import sparse
import numpy as np
from pyparsing import Word
import torch.optim as optim
import yaml
from numpy.random import normal
inv_flag = True
# import km
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import init
from tqdm import tqdm
import utils
from reader import TextReader

Tensor = torch.cuda.FloatTensor
np.random.seed(0)
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(inv_flag)

def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class LossFunctions:
    eps = 1e-8
    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type="mse"):
        if rec_type == "mse":
            if dropout_mask is None:
                loss = -torch.sum(torch.log(predicted) * real)
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none").mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda(2)).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_loss(
        self, z, z_mu, z_var, z_mu_prior, z_var_prior
    ):  
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.sum(targets * log_q, dim=-1))

class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda(2)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu, logvar

# Encoder
class InferenceNet(nn.Module):
    def __init__(self,topic_num_1,topic_num_2,topic_num_3, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(topic_num_1,topic_num_2), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.encoder_2 = nn.Sequential(nn.Linear(topic_num_2,topic_num_3), nn.BatchNorm1d(topic_num_3), nonLinear)
        self.inference_qyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3, 300),  # 64 1
                nn.BatchNorm1d(300),
                nonLinear,
                GumbelSoftmax(300, y_dim),  # 1 256
            ]
        )
        self.inference_qzyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3 + y_dim, 300),
                nn.BatchNorm1d(300),
                nonLinear,
                Gaussian(300, topic_num_3),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def qyx3(self, x,temperature,hard):
        num_layers = len(self.inference_qyx3)
        for i, layer in enumerate(self.inference_qyx3):
            if i == num_layers - 1:
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x
    def qzxy3(self, x, y):
        concat = torch.cat((x.squeeze(2), y), dim=1)
        for layer in self.inference_qzyx3:
            concat = layer(concat)
        return concat


    def forward(self, x, adj, adj_2, adj_3, temperature, hard = 0):
        if inv_flag ==True:
            x_1 = torch.matmul(adj.to(torch.float32),x.squeeze(2).T).T
            x_2 = self.encoder(x_1)
            x_2 = torch.matmul(adj_2.to(torch.float32),x_2.T).T
            x_3 = self.encoder_2(x_2)
            x_3 = torch.matmul(adj_3.to(torch.float32),x_3.T).T
        else:
            x_1 = x.squeeze(2)
            x_2 = self.encoder(x_1)
            x_3 = self.encoder_2(x_2)                     

        logits_3, prob_3, y_3  = self.qyx3(x_3,temperature, hard = 0)
        mu_3, logvar_3 = self.qzxy3(x_3.view(x_3.size(0), -1, 1), y_3)
        var_3 = torch.exp(logvar_3)
        z_3 = self.reparameterize(mu_3, var_3)
        output_3 = {"mean": mu_3, "var": var_3, "gaussian": z_3, "categorical": y_3,'logits': logits_3, 'prob_cat': prob_3}
        return output_3   

# Decoder
class GenerativeNet(nn.Module):
    def __init__(self, topic_num_1,topic_num_2,topic_num_3, x_dim=1, z_dim=1, y_dim=256, n_gene=None, nonLinear=None):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.y_var_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.decoder = nn.Sequential(CustomizedLinear(torch.ones(topic_num_3,topic_num_2),bias=False), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.decoder_2 = nn.Sequential(CustomizedLinear(torch.ones(topic_num_2,topic_num_1),bias=False), nn.BatchNorm1d(topic_num_1), nonLinear)

        if True:
            print('Constraining decoder to positive weights', flush=True)

            self.decoder[0].reset_params_pos()
            self.decoder[0].weight.data *= self.decoder[0].mask        
            self.decoder_2[0].reset_params_pos()    
            self.decoder_2[0].weight.data *= self.decoder_2[0].mask 

        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_3),
                nonLinear,
            ]
        )
        self.generative_pxz_1 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_2),
                nonLinear,
            ]
        )
        self.generative_pxz_2 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_1),
                nonLinear,
            ]
        )

    def pzy1(self, y):
        y_mu = self.y_mu_1(y)
        y_logvar = self.y_var_1(y)
        return y_mu, y_logvar
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z
    def pxz_1(self, z):
        for layer in self.generative_pxz_1:
            z = layer(z)
        return z
    def pxz_2(self, z):
        for layer in self.generative_pxz_2:
            z = layer(z)
        return z

    def forward(
        self,
        z,
        y_3,
        adj_A_t_inv_2,
        adj_A_t_inv_1,
        adj_A_t_3,
    ):
        y_mu_3, y_logvar_3 = self.pzy1(y_3)
        y_var_3 = torch.exp(y_logvar_3)

        if inv_flag ==True:
            z = torch.matmul(adj_A_t_3.to(torch.float32), z.T).T
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            z_2 = torch.matmul(adj_A_t_inv_2.to(torch.float32), z_2.T).T
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            z_3 =  torch.matmul(adj_A_t_inv_1.to(torch.float32), z_3.T).T
            out_3 = self.pxz_2(z_3)
        else:
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            out_3 = self.pxz_2(z_3)      

        output_1 = { "x_rec": out_1}
        output_2 = { "x_rec": out_2}
        output_3 = {"y_mean": y_mu_3, "y_var": y_var_3, "x_rec": out_3}
        return output_1, output_2, output_3


class net(nn.Module):
    def __init__(
        self,
        max_topic_num=64,
        batch_size=None,
        adj_A=None,
        adj_A_2=None,
        adj_A_3=None,
        mask=None,
        emb_mat=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        vocab_num=None,
        hidden_num=None,
        prior_beta=None,
        **kwargs,
    ):
        super(net, self).__init__()
        print("net topic_num_1={}".format(topic_num_1))

        self.dropout = nn.Dropout(0.1)
        self.max_topic_num = max_topic_num
        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        if emb_mat == None:
            self.word_embed = nn.Parameter(torch.rand(hidden_num, vocab_num))
        else:
            print("Using pre-train word embedding")
            self.word_embed = nn.Parameter(emb_mat)

        self.topic_embed = nn.Parameter(xavier_init.sample((topic_num_1, hidden_num)))
        self.topic_embed_1 = nn.Parameter(xavier_init.sample((topic_num_2, hidden_num)))
        self.topic_embed_2 = nn.Parameter(xavier_init.sample((topic_num_3, hidden_num)))
        self.eta = nn.Linear(vocab_num, 3)#n
        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name="adj_A")
        )
        self.adj_A_2 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_2).double(), requires_grad=True, name="adj_A_2"
            )
        )
        self.adj_A_3 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_3).double(), requires_grad=True, name="adj_A_3"
            )
        )

        self.encoder = nn.Sequential(nn.Linear(vocab_num, max_topic_num), nn.BatchNorm1d(max_topic_num), nn.Tanh())
        x_dim, y_dim, z_dim = 64, 10, 10  # x:  y:   z:
        self.n_gene = n_gene = len(adj_A)  # topic_num_1

        self.inference = InferenceNet( topic_num_1,topic_num_2,topic_num_3,x_dim, y_dim,z_dim, n_gene, nn.Tanh())
        self.generative = GenerativeNet(topic_num_1,topic_num_2,topic_num_3,x_dim, y_dim,z_dim, n_gene, nn.Tanh())

        self.losses = LossFunctions()
        for m in self.modules():
            if (
                type(m) == nn.Linear
                or type(m) == nn.Conv2d
                or type(m) == nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def to_np(self,x):
        return x.cpu().detach().numpy()

    def build_tree(self, dependency_matrix_0_1, dependency_matrix_1_2):
        [level0, level1] = dependency_matrix_0_1.shape  # 32*32
        level2 = dependency_matrix_1_2.shape[1]  # 64*32
        parents_1 = np.argmax(self.to_np(dependency_matrix_0_1), axis=0) 
        parents_2 = np.argmax(self.to_np(dependency_matrix_1_2), axis=0)

    def get_topic_dist(self, level=2):
        if level == 2:
            return torch.softmax(self.topic_embed_2 @ self.word_embed, dim=1)
        elif level == 1:
            return torch.softmax(self.topic_embed_1 @ self.word_embed, dim=1)
        elif level == 0:
            return torch.softmax(self.topic_embed @ self.word_embed, dim=1)

    def encode(self, x):
        p1 = self.encoder(x)
        return p1

    def decode(self, x_ori, out_1, out_2, out_3):
        out_3 = torch.softmax(out_3, dim=1)
        out_1 = torch.softmax(out_1, dim=1)
        out_2 = torch.softmax(out_2, dim=1)
        beta = torch.softmax(self.topic_embed @ self.word_embed, dim=1)
        beta_2 = torch.softmax(self.topic_embed_1 @ self.word_embed, dim=1)
        beta_3 = torch.softmax(self.topic_embed_2 @ self.word_embed, dim=1)
        p1 = out_3 @ beta  # #
        p2 = out_2 @ beta_2 
        p3 = out_1 @ beta_3
        p_fin = (p1.T+p2.T+p3.T)/3.0
        return p_fin.T
    def normalize_adj(self, adj: sp.csr_matrix) -> sp.coo_matrix:
        adj = sp.coo_matrix(adj)
        adj_ = adj
        rowsum = np.array(adj_.sum(0))
        rowsum_power = []
        for i in rowsum:
            for j in i:
                if j !=0 :
                    j_power = np.power(j, -0.5)
                    rowsum_power.append(j_power)
                else:
                    j_power = 0
                    rowsum_power.append(j_power)
        rowsum_power = np.array(rowsum_power)
        degree_mat_inv_sqrt = sp.diags(rowsum_power)
        adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_norm

    def _one_minus_A_t(self, adj):

        adj_normalized = abs(adj) 
       # adj_normalized = adj_normalized + (adj_normalized.T - adj_normalized)*(adj_normalized.T > adj_normalized)
        adj_normalized = Tensor(np.eye(adj_normalized.shape[0])).cuda(2) - (adj_normalized.transpose(0, 1)).cuda(2)   
        return adj_normalized

    def forward(self, x, dropout_mask=None, temperature=1.0, hard=0):
        x_ori = x
        x = self.encode(x)
        x = x.view(x.size(0), -1, 1)

        mask = Variable(
            torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(),
            requires_grad=False,
        ).cuda(2) 

        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)

        mask_1 = Variable(
            torch.from_numpy(np.ones(32) - np.eye(32)).float(), requires_grad=False
        ).cuda(2)
        adj_A_t_2 = self._one_minus_A_t(self.adj_A_2 * mask_1)
        adj_A_t_inv_2 = torch.inverse(adj_A_t_2)

        mask_2 = Variable(
            torch.from_numpy(np.ones(128) - np.eye(128)).float(), requires_grad=False
        ).cuda(2)
        adj_A_t_3 = self._one_minus_A_t(self.adj_A_3 * mask_2)
        adj_A_t_inv_3 = torch.inverse(adj_A_t_3)

        out_inf_1 = self.inference(  #
            x, adj_A_t, adj_A_t_2, adj_A_t_3, temperature, x_ori.view(x.size(0), -1, 1)
        )
        z_3, y_3 = out_inf_1["gaussian"], out_inf_1["categorical"]
        output_1, output_2, output_3 = self.generative(  # here
            z_3,
            y_3,
            adj_A_t_inv_2,
            adj_A_t_inv,
            adj_A_t_inv_3,
        )  # here

        dec_1 = output_1["x_rec"]
        dec_2 = output_2["x_rec"]
        dec_3 = output_3["x_rec"]
        dec_res = self.decode(x_ori,dec_1,dec_2,dec_3)

        loss_rec_1 = self.losses.reconstruction_loss(
            x_ori, dec_res, dropout_mask, "mse"
        )
        loss_gauss_3 = (
            self.losses.gaussian_loss(
                z_3,
                out_inf_1["mean"],
                out_inf_1["var"],
                output_3["y_mean"],
                output_3["y_var"],
            )
            * 1)
        loss_cat_3 = (-self.losses.entropy(out_inf_1['logits'], out_inf_1['prob_cat']) - np.log(0.1)) 

        loss = (
            loss_rec_1 
            + loss_gauss_3
            + loss_cat_3
        )
        return loss

class AMM_no_dag(object):
    def __init__(
        self,
        reader=None,
        max_topic_num=64,
        model_path=None,
        emb_mat=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        rho_max=None,
        rho=None,
        phi=None,
        epsilon=None,
        lam=None,
        threshold_1=None,
        threshold_2=None,
        **kwargs,
    ):
        # prepare dataset
        if reader == None:
            raise Exception(" [!] Expected data reader")

        self.reader = reader
        self.model_path = model_path
        self.n_classes = self.reader.get_n_classes()  # document class
        self.topic_num_1 = topic_num_1
        self.topic_num_2 = topic_num_2
        self.topic_num_3 = topic_num_3

        self.adj = self.initalize_A(topic_num_1)
        self.adj_2 = self.initalize_A(topic_num_2)  # topic_num_2
        self.adj_3 = self.initalize_A(topic_num_3)  # topic_num_3
        print("AMM_no_dag init model.")

        if emb_mat is None:
            self.Net = net(
                max_topic_num,
                batch_size,
                adj_A=self.adj,
                adj_A_2=self.adj_2,
                adj_A_3=self.adj_3,
                topic_num_1=self.topic_num_1,
                topic_num_2=self.topic_num_2,
                topic_num_3=self.topic_num_3,
                **kwargs,
            ).to(device)
        else:
            emb_mat = torch.from_numpy(emb_mat.astype(np.float32)).to(device)
            self.Net = net(
                max_topic_num,
                batch_size,
                adj_A=self.adj,
                adj_A_2=self.adj_2,
                adj_A_3=self.adj_3,
                topic_num_1=self.topic_num_1,
                topic_num_2=self.topic_num_2,
                topic_num_3=self.topic_num_3,
                emb_mat=emb_mat.T,
                **kwargs,
            ).to(device)

        print(self.Net)

        self.topic_num = max_topic_num
        self.pi_ave = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        # optimizer uses ADAM

    def initalize_A(self, topic_nums=16):
        A = np.ones([topic_nums, topic_nums]) / (topic_nums - 1) + (
            np.random.rand(topic_nums * topic_nums) * 0.0002
        ).reshape([topic_nums, topic_nums])
        for i in range(topic_nums):
            A[i, i] = 0
        return A

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.Net.state_dict(), f"{self.model_path}/model.pkl")
        with open(f"{self.model_path}/topic_num.txt", "w") as f:
            f.write(str(self.topic_num))
        np.save(f"{self.model_path}/pi_ave.npy", self.pi_ave)
        print(f"Models save to  {self.model_path}/model.pkl")

    def load_model(self, model_filename="model.pkl"):
        model_path = os.path.join(self.model_path, model_filename)

        self.Net.load_state_dict(torch.load(model_path))
        # self.Net = torch.load(model_path)
        with open(f"{self.model_path}/topic_num.txt", "r") as f:
            self.topic_num = int(f.read())
        self.pi_ave = np.load(f"{self.model_path}/pi_ave.npy")
        print("AMM_no_dag model loaded from {}.".format(model_path))


    def get_word_topic(self, data):
        word_topic = self.Net.infer(torch.from_numpy(data).to(device))
        word_topic = self.to_np(word_topic)
        return word_topic

    def get_topic_dist(self, level=2):
        # topic_dist = self.Net.get_topic_dist()[self.topics]
        topic_dist = self.Net.get_topic_dist(level)
        return topic_dist

    def get_topic_word(self, level=2, top_k=15, vocab=None):
        topic_dist = self.get_topic_dist(level)
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [
            [self.reader.vocab[idx] for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]
        return topic_words

    def get_topic_parents(self, mat):
        return 0

    def evaluate(self):
        # 重定向回文件
        _, _, texts = self.reader.get_sequence("all")

        for level in range(3):
            topic_word = self.get_topic_word(
                top_k=10, level=level, vocab=self.reader.vocab
            )
            # 打印top N的主题词
            for k, top_word_k in enumerate(topic_word):
                print(f"Topic {k}:{top_word_k}")

    # NPMI
    def sampling(self, flag):
        # 计算coherence，祖传方法
        test_data, test_label, _ = self.reader.get_matrix("test", mode="count")

        # for level in range(3):
        topic_dist_2 = self.to_np(self.get_topic_dist(level=2))  # 最高层主题的 coherence
        topic_dist_1 = self.to_np(self.get_topic_dist(level=1))
        topic_dist_0 = self.to_np(self.get_topic_dist(level=0))
    #    topic_dist_res = self.to_np(self.get_topic_dist(level=3))
        topic_dist = np.concatenate(
            (np.concatenate((topic_dist_2, topic_dist_1), axis=0), topic_dist_0), axis=0
        )
      # train_coherence_res = utils.evaluate_NPMI(test_data, topic_dist_res)
        train_coherence_2 = utils.evaluate_NPMI(test_data, topic_dist_2)
        train_coherence_1 = utils.evaluate_NPMI(test_data, topic_dist_1)
        train_coherence_0 = utils.evaluate_NPMI(test_data, topic_dist_0)
        train_coherence = utils.evaluate_NPMI(test_data, topic_dist)
        if flag == 1:

            TU2 = utils.evaluate_TU(topic_dist_2)
            TU1 = utils.evaluate_TU(topic_dist_1)
            TU0 = utils.evaluate_TU(topic_dist_0)
            TU = utils.evaluate_TU(topic_dist)

            print("TU level 2: " + str(TU2))
            print("TU level 1: " + str(TU1))
            print("TU level 0: " + str(TU0))
            print("TU: " + str(TU))
            print("Topic coherence  level 2: ", train_coherence_2)
            print("Topic coherence  level 1: ", train_coherence_1)
            print("Topic coherence  level 0: ", train_coherence_0)
    #    print("Topic coherence  res: ", train_coherence_res)
        print("Topic coherence:", train_coherence)
        if (train_coherence_2 + train_coherence_1 + train_coherence_0)/3 > self.best_coherence:
            self.best_coherence = (train_coherence_2 + train_coherence_1 + train_coherence_0)/3
            print("New best coherence found!!")
            self.save_model()

        pass

    def get_batches(self, batch_size=300, rand=True):
        n, d = self.train_data.shape

        batchs = n // batch_size
        while True:
            idxs = np.arange(self.train_data.shape[0])

            if rand:
                np.random.shuffle(idxs)

            for count in range(batchs):
                wordcount = []
                beg = count * batch_size
                end = (count + 1) * batch_size

                idx = idxs[beg:end]
                data = self.train_data[idx].toarray()
                data = torch.from_numpy(data).to(device)
                yield data 

    def train(self, epochs=320, batch_size=256, data_type="train+valid"):
        self.t_begin = time.time()
        batch_size = self.batch_size
        (
            self.train_data,
            self.train_label,
            self.train_text,
        ) = self.reader.get_sparse_matrix(data_type, mode="count")
        
        self.train_generator = self.get_batches(batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // batch_size
        print(batch_size)
        self.best_coherence = -1
        optimizer = optim.RMSprop(self.Net.parameters(), lr=self.lr)
        optimizer2 = optim.RMSprop(
            [self.Net.adj_A, self.Net.adj_A_2, self.Net.adj_A_3], lr=self.lr * 0.2
        )
        clipper = WeightClipper(frequency=1)
        for epoch in tqdm(range(self.epochs)):

            self.Net.train()
            epoch_word_all = 0
            doc_count = 0

            if epoch % (3) < 1:  #
                self.Net.adj_A.requires_grad = False
                self.Net.adj_A_2.requires_grad = False
                self.Net.adj_A_3.requires_grad = False

            else:
                self.Net.adj_A.requires_grad = True
                self.Net.adj_A_2.requires_grad = True
                self.Net.adj_A_3.requires_grad = True

            for i in range(n_batchs):
                optimizer.zero_grad()
                optimizer2.zero_grad()
                temperature = max(0.95 ** epoch, 0.5)
                ori_docs = next(self.train_generator)

                doc_count += ori_docs.shape[0]
                count_batch = []
                for idx in range(ori_docs.shape[0]):
                    count_batch.append(np.sum(self.to_np(ori_docs[idx])))

                epoch_word_all += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)

                loss = self.Net(
                    ori_docs, temperature = temperature
                )
                sparse_loss = (
                    1 * torch.mean(torch.abs(self.Net.adj_A))
                    + 1 * torch.mean(torch.abs(self.Net.adj_A_2))
                    + 1 * torch.mean(torch.abs(self.Net.adj_A_3))
                )
                if inv_flag:
                    loss = loss + sparse_loss
                else:
                    loss = loss

                loss.backward()

                if epoch % (3) < 1:
                    optimizer.step()
                else:
                    optimizer2.step()
                if True:
                    self.Net.generative.decoder[0].apply(clipper)
                    self.Net.generative.decoder_2[0].apply(clipper)

            self.Net.eval()
            if epoch == 1999:
                 self.save_model()
            if (epoch + 1) % 10 == 0:
                if (epoch + 1) % 50 == 0:
                    self.sampling(flag = 1)
                else:
                    self.sampling(flag = 0)

        self.t_end = time.time()
        print("Time of training-{}".format((self.t_end - self.t_begin)))

    def detach_np(self, x):
        return x.cpu().detach().numpy()  


    def test(self):
        self.load_model()
        self.Net.eval()
        self.best_coherence = 999
        self.evaluate()
        self.sampling(flag = 1)
def main(mode='Train',
         dataset="20news",
         max_topic_num=300,
         emb_type="glove",
         **kwargs):

 #   base_path = os.path.expanduser('~') + '/Methods/HNTM and nHNTM/'
    data_path = f"./data/{dataset}"
    reader = TextReader(data_path)
    print(emb_type)
    if emb_type == "bert":
        bert_emb_path = f"./emb/bert.npy"
        embedding_mat = utils.build_bert_embedding(bert_emb_path, reader.vocab,
                                                   data_path)
    elif emb_type == "glove":
        emb_path = f"./emb/glove.6B.300d.txt"
        embedding_mat = utils.build_embedding(emb_path, reader.vocab,
                                              data_path)[:-1]
    else:
        embedding_mat = None

    model_path = f'./model/{dataset}_{max_topic_num}_{reader.vocab_size}'
    model = AMM_no_dag(reader, max_topic_num, model_path, embedding_mat, **kwargs)

    if mode == 'Train':
        model.train()
    elif mode == 'Test':
        model.test()
    else:
        print(f'Unknowned mode {mode}!') 

if __name__ == '__main__':
    config = yaml.load(open('config.yaml'), yaml.FullLoader)
    main(mode="Train", **config["para"])
    main(mode="Test", **config["para"])