import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

class EvalModel():
    def __init__(self, dir_path, datacase):
        self.datacase = datacase
        self.dir_path = dir_path
        self.mixformer_alphas = None
        self.scores = None
        self.N = 0
        self.num_class = 155
        self.load_scores()
        
    def load_scores(self):
        self.scores = []
        for dc_name in os.listdir(self.dir_path):
            if dc_name[-5:] == self.datacase:
                for m in os.listdir(os.path.join(self.dir_path, dc_name)):
                    pkl_path = os.path.join(self.dir_path, dc_name, m, 'epoch1_test_score.pkl')
                    with open(pkl_path, 'rb') as f:
                        a = list(pickle.load(f).items())
                        b = []
                        for i in a:
                            b.append(i[1])
                        self.scores.append(np.array(b))
        self.scores = np.array(self.scores)
        self.N = self.scores.shape[1]
        self.mixformer_alphas = np.array([1] * 6)

    def adjust_alphas(self, mix_alphas, mixformer_alphas=None):
        assert len(mix_alphas) == len(self.mixformer_alphas)
        self.mixformer_alphas = np.array(mix_alphas)

    def forward(self, ):
        pred_score = np.zeros_like(self.scores)
        
        for i, _ in enumerate(self.mixformer_alphas):
            pred_score += self.scores[i] * self.mixformer_alphas[i]
            
        pred_score = pred_score.sum(axis=0)
        pred = pred_score.argmax(axis=-1)
        return pred

    def evaluate(self, label):
        pre = self.forward()
        acc = accuracy_score(label, pre)
        print(f'{self.datacase} acc{acc}')
        return acc

if __name__ == '__main__':
    npz_data_v1 = np.load('./data/uav/UAV-cv1.npz')
    npz_data_v2 = np.load('./data/uav/UAV-cv2.npz')
    label_v1 = np.where(npz_data_v1['y_test'] > 0)[1]
    label_v2 = np.where(npz_data_v2['y_test'] > 0)[1]
    
    evalModel_v1 = EvalModel('ensemble_results', 'CSv1')
    evalModel_v2 = EvalModel('ensemble_results', 'CSv2')

    evalModel_v1.adjust_alphas([1.5, 1.5, 0.5, 0, 0, 0])
    evalModel_v2.adjust_alphas([1, 1.5, 1, 0, 0, 0])
    
    evalModel_v1.evaluate(label_v1)
    evalModel_v2.evaluate(label_v2)
    
"""
evalModel_v1.adjust_alphas([1.5, 1.5, 0.5, 0.4, 0.7, 0.4])
evalModel_v2.adjust_alphas([1, 1.5, 1, 0.5, 0.5, 0.5])
CSv1 acc0.46075788806088475
CSv2 acc0.48879150514073827

evalModel_v1.adjust_alphas([0.5, 1, 0.5, 0.5, 1, 0.5])
evalModel_v2.adjust_alphas([1, 1.5, 1, 0.5, 0.5, 0.5])
CSv1 acc0.45631837640716666
CSv2 acc0.48879150514073827

evalModel_v1.adjust_alphas([0.5, 1, 0.5, 1, 0.5, 1])
evalModel_v2.adjust_alphas([0.5, 1, 0.5, 1, 0.5, 1])
CSv1 acc0.4523545267163469
CSv2 acc0.47429630878139223

evalModel_v1.adjust_alphas([0.2, 1.2, 0.2, 1.2, 0.2, 1.2])
evalModel_v2.adjust_alphas([0.2, 1.2, 0.2, 1.2, 0.2, 1.2])
CSv1 acc0.44458538132234027
CSv2 acc0.4658688690375864
"""