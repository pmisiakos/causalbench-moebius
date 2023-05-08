from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network,
    remove_lowly_expressed_genes,
)

class moebius(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.gene_expression_threshold = 0.25
        self.soft_adjacency_matrix_threshold = 0.5

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        
        # We remove genes that have a non-zero expression in less than 25% of samples.
        # You could also preprocess the expression matrix, for example to impute 0.0 expression values.
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix,
            gene_names,
            expression_threshold=self.gene_expression_threshold,
        )

        gene_index = {gene_names[idx] : idx for idx in range(len(gene_names))} # inverse map from gene to index

        # filtering samples that correspond to observations or known-target interventions 
        sample_index = [i for i in range(len(interventions)) if (interventions[i] == "non-targeting" or interventions[i] in gene_index.keys())]

        # initializing intervention mask with all ones
        mask = np.ones(expression_matrix.shape)

        for i in sample_index:
            protein_name = interventions[i]
            if protein_name != "non-targeting":
                if (protein_name not in gene_index.keys()):
                    print("Protein {} not found".format(protein_name))
                else:
                    mask[i, gene_index[protein_name]] = 0 # masking the data point which was intervened.
        
        # excluding all unknown-gene interventions
        mask = mask[sample_index, :]
        X = expression_matrix[sample_index,:]

        print("Original samples where {}. After excluding counfounding interventions {}".format(expression_matrix.shape[0], X.shape[0]))

        W_est = moebius_solver(X, lambda1=0, lambda2=1, intervention_mask=mask)
        parents, children = np.nonzero(W_est > 0)
        edges = set()
        for i in range(len(parents)):
            edges.add((gene_names[parents[i]],gene_names[children[i]]))
        return list(edges)


class Moebius(nn.Module):
    def __init__(self, X, lambda1, lambda2, constraint='notears', intervention_mask=None):
        super().__init__()
        self.X = torch.tensor(X)
        self.d = self.X.shape[1]
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.constraint = constraint
        self.fc = torch.nn.Linear(self.d, self.d, bias=False) # input x : output (A, ) A^Tx + b
        self.intervention_mask = intervention_mask 

    def postprocess_A(self):
        A = self.fc.weight.T
        A_est = torch.where(torch.abs(A) > 0.035, A, 0)
        return A_est.detach().cpu().numpy()
    
    def l1_reg(self):
        A = self.fc.weight
        return torch.sum(torch.abs(A)) # 
    
    def acyclicity(self):
        A = self.fc.weight
        if self.constraint == 'notears':
            return torch.trace(torch.matrix_exp(A * A)) - self.d
        elif self.constraint == 'dag-gnn':
            M = torch.eye(self.d) + A * A / self.d  # (Yu et al. 2019)
            return  torch.trace(torch.linalg.matrix_power(M, self.d)) - self.d
        elif self.constraint == 'frobenius':
            return torch.sum((A * A.T) ** 2)
        
    def forward(self, X, i, j):
        if j > X.shape[0]:
            j = X.shape[0]
        if self.intervention_mask is not None:
            return self.fc(X[i:j,:]) * self.intervention_mask[i:j,:] # output is XA * mask
        
        return self.fc(X[i:j,:]) # output is XA
        

def moebius_solver(X, lambda1, lambda2, epochs=1500, constraint="notears", intervention_mask=None):
    '''
        MOEBIUS solver
        params:
        X: data (np.array) of size n x d
        lambda1: coefficient (double) for l1 regularization λ||Α||_1
        lambda2: coefficient (double) for the graph constraint 
        epochs: upper bound for the number of iterations of the optimization solver.
    '''
    X = torch.Tensor(X, device=device)
    
    if intervention_mask is not None:
        intervention_mask = torch.Tensor(intervention_mask, device=device)

    N = X.shape[0]

    model = Moebius(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint, intervention_mask=intervention_mask)
    print("Initializing Moebius model for {} samples and {} nodes".format(N, model.d))
    print("Data info: shape of X {}, {}, min val {:.3f}, max val {:.3f}, mean = {:.3f}".format(X.shape[0], X.shape[1], X.min(), X.max(), X.mean()))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    early_stop = 40
    best_loss = 10000

    batch_size = N // 10

    for i in range(epochs):

        a = 0
        total_loss = 0

        while(a < N):
            # zero gradients and compute output = XA
            optimizer.zero_grad()
            output = model(X, a, a + batch_size) # passes the indices to know which rows of the intervention mask to use.

            # compute total optimization loss and back-propagate
            b = min(N, a + batch_size)
            input = X[a:b,:]
            loss = ( 1 / (2 * batch_size) ) * torch.norm((input - output), p=1)   # (1/2n) * |X-XA|_1 
            loss = loss + lambda1 * model.l1_reg() + lambda2 * model.acyclicity() # (1/2n) * |X-XA|_1  + λ1 * |A| + λ2 *  h(A)
            loss.backward()

            optimizer.step()

            # move on to next batch
            a += batch_size
            #calculating total loss 
            total_loss += loss.item()

        # overview of performance
        if i % 10 == 0:
            print("Epoch: {}. Loss = {:.3f}".format(i, total_loss))
        
        # early stopping 
        if total_loss >= best_loss:
            early_stop -= 1
        else:
            early_stop = 40
            best_loss = total_loss
            torch.save(model.state_dict(), 'best_model.pl')

        if early_stop == 0:
            break

    model = Moebius(X, lambda1=lambda1, lambda2=lambda2, constraint=constraint)
    model.load_state_dict(torch.load('best_model.pl'))
    print(model.fc.weight.max(), model.fc.weight.min())
    A = model.postprocess_A()
    print("Number of proposed edges is = {}".format(np.count_nonzero(A)))
    
    return A


