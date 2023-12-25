import torch
import torch.nn as nn
import numpy as np
import pickle

class BoundaryInfo:
    def __init__(self, W=None, b=None):
        self.Wb = None
        if (W is not None) & (b is not None):
            self.initWb(W, b)
        
    def initWb(self, W, b):
        self.Wb = torch.cat((W, b.reshape(1,-1)), axis=0)
        
    def extendWb(self, bdinfo):
        if self.Wb is None:
            self.Wb = bdinfo.Wb
        else:
            self.Wb = torch.cat((self.Wb, bdinfo.Wb), axis=1)
            
            
            ''' getBBOfLastlayer의 return값의 type과 매칭되게'''
            ''' 아직 argument의 type을 모름'''
    
        
    def updateDeltaG(self, idx1, idx2):
        self.Wb = self.Wb[:,idx1:idx1+1] - self.Wb[:, idx2:idx2+1]
    
    def computeVal(self, features, isSelectedBD=None):
        if isSelectedBD is None:
            return torch.mm(features, self.Wb[:-1,:])+self.Wb[-1,:]
        return torch.mm(features, self.Wb[:-1,isSelectedBD])+self.Wb[-1,isSelectedBD]
        
#     def computeConfigs(self, features):
#         values = self.computeVal(features)
#         configs = torch.zeros(values.shape, dtype=torch.bool, device='cuda')
#         configs[values>0] = True
#         return configs
    
    def computeConfigs(self, features, isSelectedBD=None):
        if isSelectedBD is None:
            n_boundaries = self.Wb.shape[1]
            isSelectedBD = torch.ones(n_boundaries, dtype=torch.bool, device='cuda')
            
        values = self.computeVal(features, isSelectedBD)
        configs = torch.zeros(values.shape, dtype=torch.bool, device='cuda')
        configs[values>0] = True
        return configs
        
        
        
    
        
class RepresentativeInterpretation:
    def __init__(self, model, dataset, target_label, feature_layer=-1):
        self.model = model
        self.feature_layer = feature_layer
        self.dataset = dataset 
        self.pred_labels = self.dataset.getTensorLabels()
        self.target_label = target_label
        
        self.pos_indices = torch.where(self.pred_labels == target_label)[0].detach().numpy()
        self.neg_indices = torch.where(self.pred_labels != target_label)[0].detach().numpy()
        
        '''lsyer start?'''
    
    def save(self, path):
        ri_info = dict()
        ri_info['neg sample indices'] = self.random_indices
        ri_info['configuration matrix'] = self.configs
        ri_info['is covered'] = self.isCovered
        ri_info['selected boundaries'] = self.boundary_list
        ri_info['neg sample labels'] = self.neg_label_list
        # save
        with open(path, 'wb') as f:
            pickle.dump(ri_info, f, pickle.HIGHEST_PROTOCOL)
        
        
    def load(self, path):
        # load
        with open(path, 'rb') as f:
            ri_info = pickle.load(f)
        self.random_indices = ri_info['neg sample indices']
        self.configs = ri_info['configuration matrix']
        self.isCovered = ri_info['is covered']
        self.boundary_list = ri_info['selected boundaries']
        self.neg_label_list = ri_info['neg sample labels']
    
    
    def getUpdatedBD(self, boundary_pt):
        '''BBofLastLayer 대용 함수'''
        '''return bdinfo corresponding to delta G'''
    
        bdinfo = self.getBD(boundary_pt)
        logits = bdinfo.computeVal(boundary_pt.flatten().unsqueeze(0).cuda())
        logits = logits.squeeze()
        
        _, largest_indices = torch.topk(logits, 2)
        max1_idx, max2_idx = largest_indices
        
        bdinfo.updateDeltaG(max1_idx, max2_idx)
        
        return bdinfo
        
    def getBD(self, boundary_pt):
        boundary_pt.requires_grad_(True)
        
        ''' G : feature -> logit '''
        logit = self.model.feature2logit(boundary_pt, self.feature_layer)
#         logit = logit.reshape(logit.size(0), -1) # logit shape이 어떻길래...?

        self.model.zero_grad()
        self.model.eval()
        
        W_list = []
        b_list = []
        ''' logit.shape[1]: # class '''
        for idx in range(logit.shape[1]):
            logit[:, idx].backward(retain_graph=True)
            
            W = boundary_pt.grad.clone().detach().reshape(boundary_pt.size(0), -1)
            W_list.append(W)
            
            Wx = torch.sum(boundary_pt.clone().detach()*boundary_pt.grad.clone().detach(), dim=(1,2,3))
            b = logit[:,idx].clone().detach() - Wx
            b_list.append(b)
            
            self.model.zero_grad()
            boundary_pt.grad.data.zero_()
        
        boundary_pt.requires_grad_(False)
        
        ''' batch가 마지막 차원으로? dim=2,1이 이상하다 '''
        W_stacked = torch.stack(W_list, dim=2).detach().squeeze().cuda()
        b_stacked = torch.stack(b_list, dim=1).detach().squeeze().cuda()
        
        return BoundaryInfo(W_stacked, b_stacked)
        
        
    
    def sampleBoundaries(self, feature, sample_size=50, randomness=False, bs_weight = 0.9):
        self.bdinfo = BoundaryInfo()
        self.model.eval()
        
        if randomness:
            self.random_indices = np.random.choice(self.neg_indices, sample_size, replace=False)
        neg_features = self.dataset.getCNNFeatures(self.random_indices).cuda()
        
        boundary_list = []
        neg_label_list = []
        
        for i in range(sample_size):
            
            pos = feature
            neg = neg_features[i]
            
            while True:
                ''' adjusted binary search '''
                boundary_pt = bs_weight*pos + (1-bs_weight)*neg
                boundary_logit = self.model.feature2logit(boundary_pt, self.feature_layer)[0]
                
                _, largest_indices = torch.topk(boundary_logit, 2)
                max1_idx, max2_idx = largest_indices
                
                isAligned = (boundary_logit[max1_idx]-boundary_logit[max2_idx])**2 < 0.00001
                
                if isAligned and max1_idx==self.target_label:
                    break
                    
                if max1_idx == self.target_label:
                    pos = boundary_pt
                else:
                    neg = boundary_pt
            
            bd_buf = self.getUpdatedBD(boundary_pt)
            self.bdinfo.extendWb(bd_buf)

            boundary_list.append(boundary_pt.cpu())
            neg_label_list.append(max2_idx.cpu())
        
        self.configs = self.bdinfo.computeConfigs(torch.flatten(self.dataset.features, start_dim=1).cuda())
               
        '''initial list 필요?'''
        self.boundary_list = boundary_list
        self.neg_label_list = neg_label_list
    
    def selectBoundaries(self, target_feature, pred_target_label, delta=0, n_step=40):
        configs = self.configs
        
        feature = torch.flatten(target_feature, start_dim=1)
        target_config = self.bdinfo.computeConfigs(feature).squeeze(0)
        
        pred_labels = self.pred_labels
        target_label = pred_target_label
        
        idx_same = (pred_labels==target_label) 
        idx_else = ~idx_same
        
        match_mat_same = (configs[idx_same, :]==target_config)
        match_mat_else = (configs[idx_else, :]==target_config)
        
        '''Submodular Optimization'''
        submodular = SubmodularAG(match_mat_same, match_mat_else)
        isSelectedBD, n_survivor_same, n_survivor_else = submodular.submodularOpt(delta, n_step)
        
        isCovered, _ = self.getCoveredIdx(isSelectedBD, target_config[isSelectedBD])
        coveredLabels = self.dataset.data[1].detach().numpy()[isCovered]
        
        ''' Glb '''
        glb = 0
        ''' Glb '''
#         print(isSelectedBD)
        
        boundary_list = np.array(self.boundary_list)[isSelectedBD.cpu()]
        neg_label_list = np.array(self.neg_label_list)[isSelectedBD.cpu()]
        
        self.isCovered = isCovered
        self.boundary_list = boundary_list
        self.neg_label_list = neg_label_list
        
        return isCovered, boundary_list, neg_label_list
    
        
    def getCoveredIdx(self, isSelectedBD, tar_config):
        
#         n_BD = torch.sum(isSelectedBD)

        features = torch.flatten(self.dataset.features, start_dim=1).to(device='cuda')
        
        subconfigs = self.bdinfo.computeConfigs(features, isSelectedBD)
        unmatched = subconfigs ^ tar_config
        
        ''' 
        - check_sum: given the target and samples, how different their configs are
        - empirically reveal the correlation btw check_sum and instance similarity
        '''
        check_sum = torch.sum(unmatched, axis=1)
        
        return (check_sum==0).cpu(), check_sum.cpu()
        

        

class SubmodularAG:
    def __init__(self, match_mat_same, match_mat_else):
        self.match_mat_same = match_mat_same
        self.match_mat_else = match_mat_else
        
        '''Oracle init...'''
        self.config_dim = match_mat_same.shape[1]
        self.same_len = match_mat_same.shape[0]  # number of samples in all positive data
        self.else_len = match_mat_else.shape[0]  # number of samples in all negative data
        self.total_len = self.same_len + self.else_len  # number of samples in all training data
        
        # Weighted
#         self._f_N_weights = (torch.sum(self.match_mat_same, axis=1) / self._D) ** 4
#         self._g_N_weights = (torch.sum(self.match_mat_else, axis=1) / self._D) ** 4
#         self._f_N_weighted = torch.sum(self._f_N_weights)
#         self._g_N_weighted = torch.sum(self._g_N_weights)
        
    def submodularOpt(self, delta=0, n_step=40):
#         self. = self.else_len - delta
        
        isSelectedBD = self.greedyAG(delta, n_step)
        n_survivor_same = len(self.well_classified_idx)
        n_survivor_else = len(self.mis_classified_idx)
        
        return isSelectedBD, n_survivor_same, n_survivor_else
        
        
    def greedyAG(self, delta, n_step=40):
        print(delta)
        print(n_step)
        
#         same_config_pos_idx
#         same_config_neg_idx
        self.well_classified_idx = torch.tensor(range(self.same_len), dtype=torch.long, device='cuda')
        self.mis_classified_idx = torch.tensor(range(self.else_len), dtype=torch.long, device='cuda')
        
        steps = 0
        
        isSelectedBD = torch.zeros(self.config_dim, dtype=torch.bool, device='cuda')
        
        while True:
            
            steps += 1
            sel_h = self.selectOptBD(isSelectedBD)
            isSelectedBD[sel_h] = True
            self.updateSurvivors(sel_h)
            
            '''Stop conditions'''
            cond1 = (len(self.mis_classified_idx)<=delta)
            cond2 = (sel_h<0)
            cond3 = (steps>n_step)
            
            if cond1 or cond2 or cond3:
                break
        
        return isSelectedBD
    
    def selectOptBD(self, isSelectedBD):
        
        if isSelectedBD.all()==True:
            return -1
        
        '''paper: (nom+denom)/denom -> nom/denom + const.'''
        nom = torch.sum(self.match_mat_same[self.well_classified_idx, :] == False, axis=0) 
        denom = torch.sum(self.match_mat_else[self.mis_classified_idx, :] == False, axis=0)
        if torch.max(denom)<=0:
            return -1

        objective = (nom + 1e-5)/(denom + 1e-10)
        
        '''assign penalty to already selected BDs'''
        objective[isSelectedBD] = 1e10
        
        sel_h = torch.argmin(objective)

        return sel_h
    
    def updateSurvivors(self, sel_h):
        
        '''True in Decision Region'''
        match_mat_same_h = self.match_mat_same[:, sel_h]
        still_true = torch.where(match_mat_same_h[self.well_classified_idx])[0]
        self.well_classified_idx = self.well_classified_idx[still_true] 
        
        '''False but in Decision Region'''
        match_mat_else_h = self.match_mat_else[:, sel_h]
        still_true = torch.where(match_mat_else_h[self.mis_classified_idx])[0]
        self.mis_classified_idx = self.mis_classified_idx[still_true]
        

        
        
        
        
            
        
      
        
    